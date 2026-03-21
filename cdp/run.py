"""CLI 진입점: 저장소 루트에서 `python -m cdp.run --mode mock|real` 로 실행합니다."""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import replace
from pathlib import Path

# `python cdp/run.py`로 직접 실행할 때도 절대 import가 동작하도록
# 저장소 루트를 `sys.path`에 보장합니다.
if __package__ in {None, ""}:
    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from cdp.config import PipelineConfig, RunMode
from cdp.orchestrator import run_pipeline
from cdp.paths import get_project_root


def _configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


def main(argv: list[str] | None = None) -> int:
    # Windows 콘솔(cp949)에서 유니코드 기호 출력 실패 방지
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
        except (OSError, ValueError, AttributeError):
            pass

    parser = argparse.ArgumentParser(description="조건부 신약 파이프라인 (통합 실행)")
    parser.add_argument(
        "--mode",
        choices=[RunMode.MOCK.value, RunMode.REAL.value],
        default=RunMode.MOCK.value,
        help="mock: 고정 데이터·메시지 검증 / real: NOTEEVENTS·Chroma·LLM·확산",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=None,
        help="기본값: cdp 패키지의 부모(저장소 루트)",
    )
    parser.add_argument(
        "--batch-limit",
        type=int,
        default=None,
        help="real 모드 배치 적재 시 처리할 Discharge summary 상한 (미지정 시 config 기본값)",
    )
    parser.add_argument(
        "--trust-checkpoint-source",
        action="store_true",
        default=True,
        help="신뢰 가능한 checkpoint에 한해 torch.load(weights_only=False)를 허용합니다.",
    )
    parser.add_argument(
        "--phase2-model-class-path",
        type=str,
        default=None,
        help="체크포인트 state_dict를 로드할 모델 클래스 경로 (예: pkg.module:ModelClass)",
    )
    parser.add_argument(
        "--phase2-no-ema",
        action="store_true",
        help="체크포인트 로드 시 ema 대신 model 가중치를 사용합니다.",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    root = (args.project_root or get_project_root()).resolve()
    if not root.is_dir():
        print(f"유효하지 않은 project root: {root}", file=sys.stderr)
        return 2

    _configure_logging(args.verbose)

    mode = RunMode(args.mode)
    cfg = replace(PipelineConfig.default(mode), project_root=root)
    if args.batch_limit is not None:
        cfg = replace(cfg, batch_limit_samples=args.batch_limit)
    if args.trust_checkpoint_source:
        cfg = replace(cfg, trust_checkpoint_source=True)
    if args.phase2_model_class_path is not None:
        cfg = replace(cfg, phase2_model_class_path=args.phase2_model_class_path)
    if args.phase2_no_ema:
        cfg = replace(cfg, phase2_use_ema_weights=False)

    try:
        run_pipeline(cfg)
    except FileNotFoundError as e:
        print(str(e), file=sys.stderr)
        return 1
    except ValueError as e:
        print(str(e), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
