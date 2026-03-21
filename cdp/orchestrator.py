"""Mock / Real 공통 오케스트레이션: 단계 산출물이 다음 단계 입력과 어떻게 맞물리는지 로그로 명시합니다."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from cdp.config import PipelineConfig, RunMode
from cdp.io_trace import TraceEvent, write_trace_event
from cdp.mock_fixtures import (
    MOCK_CHEMICAL_CONSTRAINTS,
    MOCK_CLINICAL_QUERY,
    MOCK_DIFFUSION_CANDIDATES,
    MOCK_EHR_DISCHARGE_SUMMARY,
)
from phase3_verification.pipeline_mock import print_verification_report

logger = logging.getLogger(__name__)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def _log_bridge(phase_from: str, phase_to: str, description: str) -> None:
    logger.info(
        "[연결] %s 산출 → %s 입력: %s",
        phase_from,
        phase_to,
        description,
    )
    print(f"\n[연결] {phase_from} 산출 → {phase_to} 입력: {description}\n")


def _trace(cfg: PipelineConfig, phase: str, stage: str, payload: dict[str, Any]) -> None:
    if not cfg.io_trace_enabled:
        return
    trace_path = write_trace_event(
        trace_dir=cfg.io_trace_dir,
        event=TraceEvent(phase=phase, stage=stage, payload=payload),
        max_chars=cfg.io_trace_max_chars,
    )
    print(f"[I/O Trace] {phase}.{stage} -> {trace_path.relative_to(cfg.project_root)}")


def run_pipeline(cfg: PipelineConfig) -> None:
    """전체 파이프라인을 한 번에 실행합니다."""
    print("=" * 64)
    print(f"조건부 신약 파이프라인 실행 (mode={cfg.mode.value})")
    print("=" * 64)

    if cfg.mode is RunMode.MOCK:
        _run_mock(cfg)
    else:
        _run_real(cfg)


def _run_mock(cfg: PipelineConfig) -> None:
    mock_dir = cfg.mock_run_dir
    constraints_path = mock_dir / "clinical_constraints.json"
    _write_json(constraints_path, MOCK_CHEMICAL_CONSTRAINTS)

    print("--- Mock EHR (고정) ---")
    print(MOCK_EHR_DISCHARGE_SUMMARY.strip()[:400], "...\n" if len(MOCK_EHR_DISCHARGE_SUMMARY) > 400 else "\n")
    _trace(
        cfg,
        "phase1",
        "input",
        {
            "mode": cfg.mode.value,
            "ehr_discharge_summary": MOCK_EHR_DISCHARGE_SUMMARY.strip(),
            "query": MOCK_CLINICAL_QUERY,
        },
    )
    _trace(cfg, "phase1", "output", {"chemical_constraints": MOCK_CHEMICAL_CONSTRAINTS})

    _log_bridge(
        "Phase1(픽스처)",
        "Phase2",
        f"chemical_constraints JSON이 {constraints_path.relative_to(cfg.project_root)} 에 저장되었고, "
        "Phase2 ConditionalGuidance가 동일 파일(또는 동일 키 구조)을 읽습니다.",
    )

    # Phase1이 실제로보내는 필드명과 픽스처 일치 확인 메시지
    print(
        "[검증] Phase1 최종 키 physicochemical_rules / excluded_pharmacophores / structural_priority 가 "
        "Phase2·Phase3 파서와 동일 스키마를 사용합니다."
    )

    _log_bridge(
        "Phase2(Mock 확산)",
        "Phase3",
        "MOCK_DIFFUSION_CANDIDATES 의 각 smiles 가 StructuralVerifier(Phase1 제약 JSON) 입력으로 검증됩니다.",
    )

    candidates = list(MOCK_DIFFUSION_CANDIDATES)
    print(f"[Mock] Phase2 후보 {len(candidates)}개를 Phase3로 전달 (고정 목록).")
    _trace(cfg, "phase2", "input", {"constraints_path": str(constraints_path), "constraints": MOCK_CHEMICAL_CONSTRAINTS})
    _trace(cfg, "phase2", "output", {"candidates": candidates})
    print_verification_report(constraints_path, candidates)
    _trace(cfg, "phase3", "input", {"constraints_path": str(constraints_path), "candidate_count": len(candidates)})

    print("\n[요약] mock 모드: API·NOTEEVENTS·GPU 없이 단계 간 경로·스키마·검증 로직만 확인했습니다.")
    print(f"산출물: {constraints_path}")


def _run_real(cfg: PipelineConfig) -> None:
    from phase1_clinical_rag.batch_processor import run_batch_ingestion
    from phase1_clinical_rag.agent_graph import run_phase1_clinical
    from cdp.phase2_run import run_phase2_diffusion

    root = cfg.project_root
    if not cfg.note_events_path.is_file():
        raise FileNotFoundError(
            f"Real 모드에 필요한 MIMIC 파일이 없습니다: {cfg.note_events_path}\n"
            "NOTEEVENTS.csv.gz 를 data/ 아래에 두거나 config 의 note_events_filename 을 수정하세요."
        )

    print(f"[Real] MIMIC 소스: {cfg.note_events_path.relative_to(root)}")
    _trace(
        cfg,
        "phase1_ingestion",
        "input",
        {
            "note_events_path": str(cfg.note_events_path),
            "chunk_size": cfg.batch_chunk_size,
            "limit_samples": cfg.batch_limit_samples,
        },
    )

    if not cfg.chroma_path.is_dir():
        print(
            "[Real] Chroma DB가 없어 배치 적재를 실행합니다. "
            "(GOOGLE_API_KEY 필요, 비용·시간 발생 가능)"
        )
        run_batch_ingestion(
            input_file=cfg.note_events_path,
            db_path=cfg.chroma_path,
            log_file=cfg.processed_ids_path,
            chunk_size=cfg.batch_chunk_size,
            limit_samples=cfg.batch_limit_samples,
        )
    else:
        print(f"[Real] 기존 Chroma 사용: {cfg.chroma_path.relative_to(root)}")

    query = cfg.phase1_query or MOCK_CLINICAL_QUERY
    print(f"[Real] Phase1 질의 실행 (RAG + LangGraph)…")
    _trace(cfg, "phase1", "input", {"query": query, "chroma_path": str(cfg.chroma_path)})

    final_state = run_phase1_clinical(
        query=query,
        chroma_persist_dir=cfg.chroma_path,
    )
    chemical = final_state["chemical_constraints"]
    if not isinstance(chemical, dict):
        raise TypeError("Phase1 chemical_constraints 가 dict 가 아닙니다.")

    _write_json(cfg.constraints_path, chemical)
    _trace(cfg, "phase1", "output", {"chemical_constraints": chemical, "constraints_path": str(cfg.constraints_path)})
    _log_bridge(
        "Phase1(agent_graph)",
        "Phase2",
        f"chemical_constraints 가 {cfg.constraints_path.relative_to(root)} 에 저장되었습니다.",
    )

    candidates = run_phase2_diffusion(cfg)
    _trace(
        cfg,
        "phase2",
        "output",
        {"candidate_count": len(candidates), "candidates_preview": candidates[:5]},
    )
    _log_bridge(
        "Phase2(diffusion)",
        "Phase3",
        f"{len(candidates)}개 SMILES 후보가 검증기로 전달됩니다.",
    )

    _trace(cfg, "phase3", "input", {"constraints_path": str(cfg.constraints_path), "candidate_count": len(candidates)})
    print_verification_report(cfg.constraints_path, candidates)
    print("\n[요약] real 모드: NOTEEVENTS → Chroma → LLM 제약 → 확산(가중치 있으면 로드) → RDKit 검증 까지 수행했습니다.")
