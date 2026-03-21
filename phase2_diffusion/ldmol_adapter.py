"""LDMol checkpoint inference adapter for Phase2 pipeline.

Usage:
  1) Clone LDMol repo locally.
  2) Set env var: LDMOL_REPO_PATH=/abs/path/to/ldmol
  3) Run with:
     --phase2-model-class-path phase2_diffusion.ldmol_adapter:LDMolAdapter
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


DEFAULT_LDMOL_REPO = Path("external/ldmol")
DEFAULT_VAE_CHECKPOINT = "checkpoint_autoencoder.ckpt"
DEFAULT_TOKENIZER_VOCAB = "vocab_bpe_300_sc.txt"
DEFAULT_DECODER_CONFIG = "config_decoder.json"
DEFAULT_ENCODER_CONFIG = "config_encoder.json"
DEFAULT_NUM_STEPS = 1000


def _resolve_ldmol_repo_path() -> Path:
    env_path = os.getenv("LDMOL_REPO_PATH", "").strip()
    repo_path = Path(env_path) if env_path else DEFAULT_LDMOL_REPO
    repo_path = repo_path.resolve()
    if not repo_path.is_dir():
        raise FileNotFoundError(
            f"LDMol repo 경로를 찾을 수 없습니다: {repo_path}. "
            "환경변수 LDMOL_REPO_PATH를 설정하거나 external/ldmol 경로에 배치하세요."
        )
    return repo_path


def _register_repo_path(repo_path: Path) -> None:
    repo_str = str(repo_path)
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)


class LDMolAdapter(nn.Module):
    """Adapts upstream LDMol model to Phase2 expected interface.

    Required methods for pipeline:
      - predict_x0(z, t)
      - step(z, t)
      - decode(z)
    """

    def __init__(self) -> None:
        super().__init__()
        self.repo_path = _resolve_ldmol_repo_path()
        _register_repo_path(self.repo_path)

        from diffusion import create_diffusion  # type: ignore
        from models import DiT_models  # type: ignore
        from train_autoencoder import ldmol_autoencoder  # type: ignore
        from utils import AE_SMILES_decoder, regexTokenizer  # type: ignore

        self._ae_decoder_fn = AE_SMILES_decoder
        self._create_diffusion = create_diffusion
        self._diffusion = create_diffusion(str(DEFAULT_NUM_STEPS))

        # Checkpoint args에 맞춘 LDMol 기본 구성.
        self.input_size = 127
        self.in_channels = 64
        self.cross_attn = 768
        self.condition_dim = 1024

        self.core = DiT_models["DiT-XL/2"](
            input_size=self.input_size,
            in_channels=self.in_channels,
            cross_attn=self.cross_attn,
            condition_dim=self.condition_dim,
        )

        decoder_cfg = {
            "bert_config_decoder": str(self.repo_path / DEFAULT_DECODER_CONFIG),
            "bert_config_encoder": str(self.repo_path / DEFAULT_ENCODER_CONFIG),
            "embed_dim": 256,
        }
        tokenizer = regexTokenizer(vocab_path=str(self.repo_path / DEFAULT_TOKENIZER_VOCAB), max_len=127)
        self.ae_model = ldmol_autoencoder(config=decoder_cfg, no_train=True, tokenizer=tokenizer)
        self._load_autoencoder_checkpoint_if_exists()
        self.eval()

    def _load_autoencoder_checkpoint_if_exists(self) -> None:
        ckpt = os.getenv("LDMOL_VAE_CKPT", "").strip()
        if ckpt:
            ckpt_path = Path(ckpt).resolve()
        else:
            ckpt_path = self.repo_path / "Pretrain" / DEFAULT_VAE_CHECKPOINT
        if not ckpt_path.is_file():
            return

        payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state_dict = payload.get("model", payload.get("state_dict", payload))
        if not isinstance(state_dict, dict):
            return
        self.ae_model.load_state_dict(state_dict, strict=False)

    def predict_x0(self, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # Upstream model outputs eps(, sigma). 여기서는 eps 기준으로 x0 근사.
        model_out = self.core(z, t)
        eps = model_out[:, : self.in_channels]
        return z - eps

    def step(self, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # 간이 DDPM step: epsilon 추정 기반 한 스텝 감소.
        model_out = self.core(z, t)
        eps = model_out[:, : self.in_channels]
        return z - (0.1 * eps)

    @torch.no_grad()
    def sample_latent(self, initial_noise: torch.Tensor, num_steps: int) -> torch.Tensor:
        """Upstream diffusion.p_sample_loop 경로로 latent를 샘플링합니다."""
        if num_steps < 1:
            raise ValueError("num_steps는 1 이상이어야 합니다.")
        diffusion = self._create_diffusion(str(num_steps))
        model_kwargs: dict[str, Any] = {"y": None, "pad_mask": None}
        return diffusion.p_sample_loop(
            self.core.forward,
            initial_noise.shape,
            initial_noise,
            clip_denoised=False,
            model_kwargs=model_kwargs,
            progress=False,
            device=initial_noise.device,
        )

    @torch.no_grad()
    def decode(self, z: torch.Tensor) -> list[str]:
        latent = z.squeeze(-1).permute((0, 2, 1))
        smiles = self._ae_decoder_fn(latent, self.ae_model, stochastic=False, k=1)
        return [str(item).strip() for item in smiles if str(item).strip()]

