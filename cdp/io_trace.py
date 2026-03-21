"""파이프라인 단계별 입출력 샘플을 JSON으로 기록하는 유틸리티."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class TraceEvent:
    """단계별 I/O 기록 이벤트."""

    phase: str
    stage: str
    payload: dict[str, Any]


def _truncate_text(text: str, max_chars: int) -> str:
    if max_chars < 1:
        raise ValueError("max_chars는 1 이상이어야 합니다.")
    return text if len(text) <= max_chars else f"{text[:max_chars]}...(truncated)"


def summarize_value(value: Any, max_chars: int) -> Any:
    """가독성을 위해 긴 값은 요약합니다."""
    if isinstance(value, str):
        return _truncate_text(value, max_chars)
    if isinstance(value, list):
        return [summarize_value(v, max_chars) for v in value[:5]]
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for idx, key in enumerate(value):
            if idx >= 15:
                out["..."] = "truncated"
                break
            out[key] = summarize_value(value[key], max_chars)
        return out
    return value


def write_trace_event(trace_dir: Path, event: TraceEvent, max_chars: int) -> Path:
    """
    단계별 I/O 이벤트를 타임스탬프 파일로 저장합니다.

    Args:
        trace_dir: 이벤트 저장 디렉터리.
        event: 기록할 이벤트 데이터.
        max_chars: 문자열 요약 최대 길이.

    Returns:
        생성된 JSON 파일 경로.
    """
    if not event.phase.strip() or not event.stage.strip():
        raise ValueError("phase/stage는 비어 있으면 안 됩니다.")

    trace_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = f"{timestamp}_{event.phase}_{event.stage}.json".replace(" ", "_")
    output_path = trace_dir / filename

    payload = asdict(event)
    payload["payload"] = summarize_value(event.payload, max_chars)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return output_path
