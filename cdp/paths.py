"""저장소 루트 해석: 모든 데이터/산출물 경로는 이 루트에 대한 상대 경로로만 조합합니다."""

from pathlib import Path


def get_project_root() -> Path:
    """`cdp` 패키지의 부모 디렉터리(저장소 루트)를 반환합니다."""
    return Path(__file__).resolve().parent.parent
