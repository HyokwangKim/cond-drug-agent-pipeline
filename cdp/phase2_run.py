"""Phase 2: 제약 JSON을 읽고 Training-Free 확산 경로로 SMILES 후보를 생성합니다."""

from __future__ import annotations

import importlib
import logging
import pickle
import re
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from cdp.config import PipelineConfig
from cdp.io_trace import TraceEvent, write_trace_event
from phase2_diffusion.real_custom_sampler import (
    ConditionalGuidance,
    guided_sampling,
)

logger = logging.getLogger(__name__)

LDMOL_WEIGHTS_NAME = "checkpoint_ldmol.pt"
SMILES_PATTERN = re.compile(r"^[A-Za-z0-9@\+\-\[\]\(\)=#$\\/\.]+$")


class DummyDiffusionModel(nn.Module):
    """가중치가 없을 때 파이프라인 연결만 검증하는 최소 구현."""

    def predict_x0(self, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return z * 0.9

    def step(self, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return z * 0.95

    def decode(self, z: torch.Tensor) -> list[str]:
        # 제약과 정합되는 예시 SMILES (Fasudil 계열)
        return ["O=S(=O)(c1cccc2cnccc12)N3CCNCC3"]


class DeterministicSurrogate(nn.Module):
    """학습 없이 사용하는 고정 surrogate: 잠재벡터 노름 기반 점수."""

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return z.norm(dim=1, keepdim=True) * 0.01


def _load_model_class(class_path: str) -> type[nn.Module]:
    """`module:ClassName` 형식의 모델 클래스를 동적으로 로드합니다."""
    if ":" not in class_path:
        raise ValueError("phase2_model_class_path 형식은 `module:ClassName` 이어야 합니다.")
    module_name, class_name = class_path.split(":", 1)
    module = importlib.import_module(module_name.strip())
    model_cls = getattr(module, class_name.strip(), None)
    if model_cls is None:
        raise ValueError(f"클래스를 찾을 수 없습니다: {class_path}")
    if not isinstance(model_cls, type) or not issubclass(model_cls, nn.Module):
        raise TypeError(f"nn.Module 하위 클래스가 아닙니다: {class_path}")
    return model_cls


def _build_checkpoint_model(cfg: PipelineConfig, checkpoint_payload: Any, device: torch.device) -> nn.Module | None:
    """체크포인트 `model/ema` state_dict를 실제 모델 클래스에 로드합니다."""
    class_path = cfg.phase2_model_class_path
    if class_path is None:
        return None
    if not isinstance(checkpoint_payload, dict):
        raise ValueError("checkpoint payload가 dict 형식이 아닙니다.")

    state_key = "ema" if cfg.phase2_use_ema_weights and "ema" in checkpoint_payload else "model"
    state_dict = checkpoint_payload.get(state_key)
    if not isinstance(state_dict, dict):
        raise ValueError(f"checkpoint에 state_dict 키가 없습니다: {state_key}")

    model_cls = _load_model_class(class_path)
    try:
        model = model_cls().to(device)
    except TypeError as e:
        raise TypeError(
            "모델 기본 생성자 호출에 실패했습니다. 인자가 필요한 클래스면 래퍼 클래스를 만들어 주세요."
        ) from e

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    logger.info(
        "checkpoint state_dict loaded class=%s state_key=%s missing=%s unexpected=%s",
        class_path,
        state_key,
        len(missing),
        len(unexpected),
    )
    print(
        f"[Phase2] checkpoint state loaded -> {class_path} ({state_key}), "
        f"missing={len(missing)}, unexpected={len(unexpected)}"
    )
    required_methods = ("predict_x0", "step", "decode")
    missing_methods = [name for name in required_methods if not callable(getattr(model, name, None))]
    if missing_methods:
        raise TypeError(f"체크포인트 모델에 필수 메서드가 없습니다: {missing_methods}")
    return model.eval()


def _build_initial_noise(cfg: PipelineConfig, model: nn.Module, device: torch.device) -> torch.Tensor:
    """모델 입력 요구 shape에 맞춰 초기 노이즈를 생성합니다."""
    in_channels = getattr(model, "in_channels", None)
    input_size = getattr(model, "input_size", None)
    if isinstance(in_channels, int) and isinstance(input_size, int):
        return torch.randn(1, in_channels, input_size, 1, device=device, requires_grad=False)
    return torch.randn(1, cfg.latent_dim, device=device, requires_grad=False)


def _extract_smiles_candidates(checkpoint_payload: Any) -> list[str]:
    """체크포인트에서 추론 가능한 SMILES 후보를 안전하게 추출합니다."""
    def _is_smiles_like(text: str) -> bool:
        token = text.strip()
        return bool(token) and len(token) >= 4 and bool(SMILES_PATTERN.match(token))

    def _normalize_candidates(raw_values: list[Any]) -> list[str]:
        return [str(item).strip() for item in raw_values if isinstance(item, str) and _is_smiles_like(item)]

    def _collect_recursive(node: Any) -> list[str]:
        if isinstance(node, dict):
            out: list[str] = []
            for key, value in node.items():
                key_norm = str(key).lower()
                if key_norm in {"smiles", "smiles_list", "smiles_candidates", "generated_smiles", "samples"}:
                    if isinstance(value, list):
                        out.extend(_normalize_candidates(value))
                    elif isinstance(value, str) and _is_smiles_like(value):
                        out.append(value.strip())
                out.extend(_collect_recursive(value))
            return out
        if isinstance(node, list):
            if all(isinstance(item, str) for item in node):
                return _normalize_candidates(node)
            out: list[str] = []
            for item in node:
                out.extend(_collect_recursive(item))
            return out
        return []

    if isinstance(checkpoint_payload, dict):
        if isinstance(checkpoint_payload.get("smiles_candidates"), list):
            parsed = _normalize_candidates(checkpoint_payload["smiles_candidates"])
            if parsed:
                return parsed
        if isinstance(checkpoint_payload.get("smiles_list"), list):
            parsed = _normalize_candidates(checkpoint_payload["smiles_list"])
            if parsed:
                return parsed
        recursive_candidates = _collect_recursive(checkpoint_payload)
        if recursive_candidates:
            # 순서 유지 중복 제거
            return list(dict.fromkeys(recursive_candidates))
    return []


def _load_checkpoint_payload(checkpoint_path: Path, device: torch.device) -> Any:
    """체크포인트 파일을 로드합니다."""
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"checkpoint 파일이 없습니다: {checkpoint_path}")
    return torch.load(checkpoint_path, map_location=device, weights_only=True)


def run_phase2_diffusion(cfg: PipelineConfig) -> list[dict[str, str]]:
    """
    Phase 2 실행.

    Returns:
        Phase 3와 동일한 키를 갖는 후보 목록: ``name``, ``smiles``.
    """
    constraints_path = cfg.constraints_path
    if not constraints_path.is_file():
        raise FileNotFoundError(f"제약 JSON이 없습니다: {constraints_path}")

    guidance = ConditionalGuidance(str(constraints_path))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Phase2 device=%s", device)

    ldmol_path = cfg.weights_dir / LDMOL_WEIGHTS_NAME
    checkpoint_payload: Any | None = None
    checkpoint_smiles: list[str] = []
    checkpoint_model: nn.Module | None = None
    if ldmol_path.is_file():
        try:
            checkpoint_payload = _load_checkpoint_payload(ldmol_path, device)
            checkpoint_smiles = _extract_smiles_candidates(checkpoint_payload)
            checkpoint_model = _build_checkpoint_model(cfg, checkpoint_payload, device)
            logger.info(
                "LDMol checkpoint 로드 성공 path=%s smiles_candidates=%s",
                ldmol_path,
                len(checkpoint_smiles),
            )
            print(f"[Phase2] checkpoint loaded: {ldmol_path}")
        except (RuntimeError, OSError, ValueError, pickle.UnpicklingError) as e:
            if cfg.trust_checkpoint_source:
                logger.warning("weights_only 로드 실패. trust 모드로 재시도 path=%s err=%s", ldmol_path, e)
                try:
                    checkpoint_payload = torch.load(ldmol_path, map_location=device, weights_only=False)
                    checkpoint_smiles = _extract_smiles_candidates(checkpoint_payload)
                    checkpoint_model = _build_checkpoint_model(cfg, checkpoint_payload, device)
                    logger.info(
                        "trust 모드 checkpoint 로드 성공 path=%s smiles_candidates=%s",
                        ldmol_path,
                        len(checkpoint_smiles),
                    )
                    print(f"[Phase2] checkpoint loaded in trust mode: {ldmol_path}")
                except (RuntimeError, OSError, ValueError, pickle.UnpicklingError) as e2:
                    logger.warning("trust 모드 checkpoint 로드 실패 path=%s err=%s", ldmol_path, e2)
                    print(f"[Phase2] checkpoint load failed, fallback to dummy: {e2}")
            else:
                logger.warning("checkpoint 로드 실패 path=%s err=%s", ldmol_path, e)
                print(f"[Phase2] checkpoint load failed, fallback to dummy: {e}")
    else:
        logger.warning("LDMol 코어 가중치 없음 (%s). DummyDiffusionModel로 디코딩합니다.", ldmol_path)
        print(f"[Phase2] checkpoint not found, fallback to dummy: {ldmol_path}")

    model = checkpoint_model if checkpoint_model is not None else DummyDiffusionModel().to(device)
    surrogate = DeterministicSurrogate().to(device)
    initial_noise = _build_initial_noise(cfg, model, device)
    if cfg.io_trace_enabled:
        write_trace_event(
            trace_dir=cfg.io_trace_dir,
            event=TraceEvent(
                phase="phase2",
                stage="input",
                payload={
                    "constraints_path": str(constraints_path),
                    "mw_limit": guidance.mw_limit,
                    "excluded_substructure_count": len(guidance.excluded_smarts),
                    "initial_noise_shape": list(initial_noise.shape),
                    "diffusion_num_steps": cfg.diffusion_num_steps,
                    "guidance_scale": cfg.diffusion_guidance_scale,
                },
            ),
            max_chars=cfg.io_trace_max_chars,
        )

    uses_upstream_sampling = callable(getattr(model, "sample_latent", None))
    if uses_upstream_sampling:
        logger.info("Phase2 uses upstream p_sample_loop path")
        print("[Phase2] upstream diffusion.p_sample_loop 경로로 샘플링합니다.")
        final_latent = model.sample_latent(initial_noise=initial_noise, num_steps=cfg.diffusion_num_steps)
    else:
        final_latent = guided_sampling(
            model=model,
            surrogate=surrogate,
            initial_noise=initial_noise,
            num_steps=cfg.diffusion_num_steps,
            guidance_scale=cfg.diffusion_guidance_scale,
        )

    smiles_list = checkpoint_smiles if checkpoint_smiles else model.decode(final_latent)
    candidates = [{"name": f"generated_{i + 1}", "smiles": sm} for i, sm in enumerate(smiles_list)]
    if cfg.io_trace_enabled:
        write_trace_event(
            trace_dir=cfg.io_trace_dir,
            event=TraceEvent(
                phase="phase2",
                stage="decoded_output",
                payload={
                    "final_latent_shape": list(final_latent.shape),
                    "checkpoint_path": str(ldmol_path),
                    "checkpoint_loaded": checkpoint_payload is not None,
                    "checkpoint_smiles_count": len(checkpoint_smiles),
                    "uses_upstream_sampling": uses_upstream_sampling,
                    "candidates": candidates,
                },
            ),
            max_chars=cfg.io_trace_max_chars,
        )
    return candidates
