"""실행 모드·경로·배치 한도 등 파이프라인 파라미터를 한곳에서 관리합니다."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from cdp.paths import get_project_root


class RunMode(str, Enum):
    """mock: 고정 픽스처로 단계 연결만 검증. real: NOTEEVENTS·Chroma·LLM·Training-Free 실행."""

    MOCK = "mock"
    REAL = "real"


@dataclass(frozen=True)
class PipelineConfig:
    """파이프라인 전역 설정. 하드코딩 대신 이 클래스만 수정하면 동작 조건이 바뀌도록 합니다."""

    mode: RunMode
    project_root: Path
    data_subdir: str = "data"
    note_events_filename: str = "NOTEEVENTS.csv.gz"
    chroma_subdir: str = "chroma_db_real"
    clinical_constraints_filename: str = "clinical_constraints.json"
    processed_ids_filename: str = "processed_ids.txt"
    weights_subdir: str = "weights"
    mock_runs_subdir: str = "runs/mock"
    io_trace_subdir: str = "runs/io_trace"
    batch_chunk_size: int = 1000
    batch_limit_samples: int | None = 50
    phase1_query: str | None = None
    diffusion_num_steps: int = 20
    diffusion_guidance_scale: float = 0.15
    latent_dim: int = 256
    io_trace_enabled: bool = True
    io_trace_max_chars: int = 1000
    trust_checkpoint_source: bool = False
    phase2_model_class_path: str | None = None
    phase2_use_ema_weights: bool = True

    @classmethod
    def default(cls, mode: RunMode) -> PipelineConfig:
        return cls(mode=mode, project_root=get_project_root())

    @property
    def data_dir(self) -> Path:
        return self.project_root / self.data_subdir

    @property
    def note_events_path(self) -> Path:
        return self.data_dir / self.note_events_filename

    @property
    def chroma_path(self) -> Path:
        return self.data_dir / self.chroma_subdir

    @property
    def constraints_path(self) -> Path:
        return self.data_dir / self.clinical_constraints_filename

    @property
    def processed_ids_path(self) -> Path:
        return self.data_dir / self.processed_ids_filename

    @property
    def weights_dir(self) -> Path:
        return self.data_dir / self.weights_subdir

    @property
    def mock_run_dir(self) -> Path:
        return self.data_dir / self.mock_runs_subdir

    @property
    def io_trace_dir(self) -> Path:
        return self.data_dir / self.io_trace_subdir
