"""Mock 모드용 고정 EHR·제약·생성 SMILES. API/대용량 데이터 없이 단계 간 I/O 계약을 검증합니다."""

from __future__ import annotations

from typing import Any, TypedDict


class MoleculeCandidate(TypedDict):
    name: str
    smiles: str


# Phase 1 입력으로 쓰는 가상 퇴원 요약 (RAG 컨텍스트가 비어도 스키마·Phase2 연결 설명용)
MOCK_EHR_DISCHARGE_SUMMARY: str = """
Patient is a 68-year-old with hypertension and documented angioedema to ACE inhibitors.
Discharge meds: amlodipine 5 mg daily. Avoid ARB with tetrazole-like motifs in de novo design.
ICU course: brief hypotension resolved; no beta-lactam antibiotics in last 48h.
"""

# 에이전트 그래프 질의 (Mock에서도 동일 문자열로 "입력"을 고정)
MOCK_CLINICAL_QUERY: str = (
    "Amlodipine-tolerant patient with ACE-inhibitor angioedema history. "
    "Derive chemical constraints for a ROCK-biased small molecule avoiding tetrazole and beta-lactam motifs."
)

# Phase 1 최종 산출물 형태 (Phase 2 ConditionalGuidance / Phase 3 StructuralVerifier와 키 호환)
# 금기: β-내酰胺, 테트라졸(로사르탄류). 술폰아미드는 Fasudil 계열 목표와 충돌하므로 mock 에서는 제외합니다.
MOCK_CHEMICAL_CONSTRAINTS: dict[str, Any] = {
    "physicochemical_rules": {
        "MWT": "300-550",
        "LogP": "1.0-4.5",
        "HBD": "<=4",
        "HBA": "<=8",
    },
    "excluded_pharmacophores": [
        "C1C(=O)NC1",  # beta-lactam core
        "c1nnnn1",  # tetrazole (Losartan SMILES 와 RDKit 부분구조 일치 확인됨)
    ],
    "structural_priority": (
        "Nitrogen heterocycle suitable for hinge binding; avoid tetrazole and beta-lactam warheads."
    ),
}

# 고분자량 탈락용: MWT 상한 550 초과 (~C45 지방산)
_MOCK_HEAVY_MW_SMILES: str = "C" * 45 + "(=O)O"

# Phase 2 -> Phase 3 후보 (의도적으로 PASS/REJECT 혼합)
MOCK_DIFFUSION_CANDIDATES: list[MoleculeCandidate] = [
    {"name": "Fasudil-like (목표 경로)", "smiles": "O=S(=O)(c1cccc2cnccc12)N3CCNCC3"},
    {"name": "Losartan-like (tetrazole 금기)", "smiles": "CCCCC1=NC(=C(N1CC2=CC=C(C=C2)C3=CC=CC=C3C4=NNN=N4)CO)Cl"},
    {"name": "Aspirin (소분자 통과)", "smiles": "CC(=O)Oc1ccccc1C(=O)O"},
    {"name": "Amoxicillin-like (beta-lactam)", "smiles": "CC1(C(N2C(S1)C(C2=O)NC(=O)C(C3=CC=C(C=C3)O)N)C(=O)O)C"},
    {"name": "Heavy chain acid (MW 초과)", "smiles": _MOCK_HEAVY_MW_SMILES},
]
