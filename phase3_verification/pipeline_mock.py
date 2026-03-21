from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Iterable, Mapping

from rdkit import Chem
from rdkit.Chem import Descriptors

from cdp.io_trace import TraceEvent, write_trace_event

logger = logging.getLogger(__name__)

# --- 1. 더미 확산 모델 (Mock Diffusion Model) ---
def mock_diffusion_generation():
    """
    Phase 2의 확산 모델이 생성했다고 가정하는 가상의 분자 리스트입니다.
    의도적으로 합격할 물질과 불합격할 물질을 섞어두었습니다.
    """
    print("🤖 [Phase 2: Mock Diffusion] 가상의 분자 생성 중...")
    return [
        {"name": "Fasudil (목표 약물)", "smiles": "O=S(=O)(c1cccc2cnccc12)N3CCNCC3"},
        {"name": "Losartan (ARB 계열 금기)", "smiles": "CCCCC1=NC(=C(N1CC2=CC=C(C=C2)C3=CC=CC=C3C4=NNN=N4)CO)Cl"},
        {"name": "Aspirin (분자량 통과/작은 분자)", "smiles": "CC(=O)Oc1ccccc1C(=O)O"},
        {"name": "Amoxicillin (Beta-lactam 금기)", "smiles": "CC1(C(N2C(S1)C(C2=O)NC(=O)C(C3=CC=C(C=C3)O)N)C(=O)O)C"},
        {"name": "Heavy_Molecule (분자량 초과)", "smiles": "CCCCCCCCCCCCCCCCCCCCCCCCCCCCCC(=O)O"}
    ]

# --- 2. 검증 에이전트 (Verification Agent) ---
class StructuralVerifier:
    def __init__(self, json_path: str | Path) -> None:
        path = Path(json_path)
        if not path.is_file():
            raise FileNotFoundError(f"JSON 제약 조건 파일이 없습니다: {path}")

        with path.open("r", encoding="utf-8") as f:
            self.constraints = json.load(f)
            
        # 분자량 파싱
        rules = self.constraints.get("physicochemical_rules", {})
        mw_str = rules.get("molecular_weight", rules.get("MWT", rules.get("MW", "500")))
        if isinstance(mw_str, str) and "-" in mw_str:
            mw_str = mw_str.split("-")[1]
        self.mw_limit = float(re.sub(r"[^\d.]", "", str(mw_str)))
        
        # 금기 구조 파싱 (custom_sampler의 로직 재사용)
        self.excluded_smarts = []
        raw_list = self.constraints.get("excluded_pharmacophores", [])
        for item in raw_list:
            match = re.search(r"\((?:SMILES|SMARTS):\s*([^)]+)\)", item, re.IGNORECASE)
            pattern = match.group(1).strip() if match else item.strip()
            mol = Chem.MolFromSmarts(pattern) or Chem.MolFromSmiles(pattern)
            if mol: self.excluded_smarts.append((pattern, mol))

    def verify(self, smiles: str) -> tuple[bool, str]:
        """단일 SMILES에 대해 제약 조건 통과 여부를 검증합니다."""
        mol = Chem.MolFromSmiles(smiles)
        if not mol:
            return False, "유효하지 않은 화학식"

        # 1. 분자량 검증
        mw = Descriptors.MolWt(mol)
        if mw > self.mw_limit:
            return False, f"분자량 초과 (제한: {self.mw_limit}, 실제: {mw:.1f})"

        # 2. 금기 구조 검증
        for pattern, smarts in self.excluded_smarts:
            if mol.HasSubstructMatch(smarts):
                return False, f"금기 구조 포함 적발 ({pattern})"

        return True, "모든 제약 조건 통과 (PASS)"

def print_verification_report(
    constraints_path: Path,
    candidates: Iterable[Mapping[str, str]],
) -> None:
    """Phase 3: 제약 JSON과 후보 SMILES 목록에 대해 PASS/REJECT 로그를 출력합니다."""
    verifier = StructuralVerifier(constraints_path)
    logger.info("Phase3 loaded constraints MW_limit=%s", verifier.mw_limit)
    print(f"✅ Phase 3: 제약 로드 (기준 분자량: {verifier.mw_limit})")

    print("\n🔬 [Phase 3: Hybrid Verification] 후보 물질 검증 시작...")
    print("-" * 60)

    cand_list = list(candidates)
    passed: list[Mapping[str, str]] = []
    rejected: list[dict[str, str]] = []
    for candidate in cand_list:
        name = candidate["name"]
        smiles = candidate["smiles"]
        is_pass, reason = verifier.verify(smiles)
        if is_pass:
            print(f"🟢 [PASS] {name}")
            print(f"   - SMILES: {smiles}")
            passed.append(candidate)
        else:
            print(f"🔴 [REJECT] {name}")
            print(f"   - 사유: {reason}")
            rejected.append({"name": name, "smiles": smiles, "reason": reason})

    print("-" * 60)
    print(f"🏆 최종 요약: 총 {len(cand_list)}개 후보 중 {len(passed)}개 물질이 검증을 통과했습니다.")
    print("=" * 60)
    trace_dir = constraints_path.resolve().parent / "runs" / "io_trace"
    write_trace_event(
        trace_dir=trace_dir,
        event=TraceEvent(
            phase="phase3",
            stage="verification_output",
            payload={
                "constraints_path": str(constraints_path),
                "candidate_count": len(cand_list),
                "passed": list(passed),
                "rejected": rejected,
            },
        ),
        max_chars=500,
    )


# --- 3. 파이프라인 통합 실행 (레거시 단독 스크립트) ---
def run_pipeline() -> None:
    print("=" * 60)
    print("💊 조건부 신약 후보 물질 생성 통합 파이프라인 검증 💊")
    print("=" * 60)

    root = Path(__file__).resolve().parents[1]
    json_path = root / "data" / "clinical_constraints.json"
    candidates = mock_diffusion_generation()
    print_verification_report(json_path, candidates)

if __name__ == "__main__":
    run_pipeline()