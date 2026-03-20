# phase3_verification/pipeline_mock.py
import json
from rdkit import Chem
from rdkit.Chem import Descriptors

# ---------------------------------------------------------
# [Phase 1] 에이전트가 도출한 제약 조건 (이전 단계의 결과 모사)
# ---------------------------------------------------------
phase1_constraints = {
    "max_molecular_weight": 500,
    "avoid_substructures": ["beta-lactam ring"]
}

# ---------------------------------------------------------
# [Phase 2] 더미 확산 모델 (Dummy Diffusion Model)
# ---------------------------------------------------------
def dummy_diffusion_model(constraints):
    print("\n🚀 [Phase 2] 더미(Mock) 확산 모델 가동 중...")
    print(" -> LDMol 모델이 제약 조건에 맞춰 분자를 생성하는 척합니다.")
    
    # 테스트를 위해 고의로 2개의 분자를 생성합니다.
    # 1. 조건을 잘 지킨 안전한 분자 (예: Ibuprofen)
    safe_molecule = "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O"
    
    # 2. 고의로 'beta-lactam' 구조를 포함시킨 실패 유도 분자 (예: Ampicillin)
    # 페니실린계 항생제로, 환자의 알레르기를 유발하는 구조입니다.
    toxic_molecule = "CC1(C(N2C(S1)C(C2=O)NC(=O)C(C3=CC=CC=C3)N)C(=O)O)C"
    
    generated_candidates = [
        {"name": "Candidate_A (Ibuprofen-like)", "smiles": safe_molecule},
        {"name": "Candidate_B (Ampicillin-like)", "smiles": toxic_molecule}
    ]
    
    return generated_candidates

# ---------------------------------------------------------
# [Phase 3] RDKit 기반 화학적 타당성 및 제약 조건 검증 (Verifier)
# ---------------------------------------------------------
# 자연어로 된 화학 구조 이름을 RDKit이 이해할 수 있는 SMARTS 패턴으로 매핑
SMARTS_DICTIONARY = {
    "beta-lactam ring": "C1CC(=O)N1"  # 4원환 락탐(Lactam) 구조
}

def verify_molecule_with_rdkit(smiles, constraints):
    # 1. SMILES 문자열을 RDKit 분자 객체로 변환
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return False, "유효하지 않은 화학 구조 (SMILES 파싱 실패)"

    # 2. 분자량(Molecular Weight) 검증
    mw = Descriptors.MolWt(mol)
    max_mw = constraints.get("max_molecular_weight", 1000)
    if mw > max_mw:
        return False, f"분자량 초과 (계산값: {mw:.2f} > 제한: {max_mw})"

    # 3. 회피 구조(Avoid Substructures) 검증 (알레르기 필터링)
    avoid_list = constraints.get("avoid_substructures", [])
    for sub_name in avoid_list:
        smarts_pattern = SMARTS_DICTIONARY.get(sub_name)
        if smarts_pattern:
            pattern_mol = Chem.MolFromSmarts(smarts_pattern)
            # 분자 내에 해당 금지 구조가 포함되어 있는지 매칭 테스트 (SubstructMatch)
            if mol.HasSubstructMatch(pattern_mol):
                return False, f"환자 알레르기 유발 구조 발견 ({sub_name})"

    return True, f"모든 제약 조건 통과 (분자량: {mw:.2f})"

# ---------------------------------------------------------
# 파이프라인 실행 메인 블록
# ---------------------------------------------------------
if __name__ == "__main__":
    print("==================================================")
    print(" 에이전트 기반 신약 후보 물질 생성 파이프라인 (통합 테스트)")
    print("==================================================")
    print(f"[입력된 제약 조건]:\n{json.dumps(phase1_constraints, indent=4)}\n")
    
    # 1. 가짜 모델을 통해 후보 물질 생성
    candidates = dummy_diffusion_model(phase1_constraints)
    
    print("\n🔬 [Phase 3] RDKit 하이브리드 검증 시스템 가동...")
    
    # 2. 생성된 분자들을 하나씩 RDKit으로 깐깐하게 검증
    for idx, candidate in enumerate(candidates, 1):
        print(f"\n[{idx}번 후보 물질 검토]: {candidate['name']}")
        print(f" - 구조식(SMILES): {candidate['smiles']}")
        
        # RDKit 검증 함수 호출
        is_valid, reason = verify_molecule_with_rdkit(candidate['smiles'], phase1_constraints)
        
        if is_valid:
            print(f" -> ✅ [PASS] {reason}")
            print(" -> [최종 결과]: 이 물질은 다음 단계(임상/합성)로 넘어갑니다.")
        else:
            print(f" -> ❌ [REJECT] 사유: {reason}")
            print(" -> [최종 결과]: 폐기 및 확산 모델에 재생성 요청 (Rejection Sampling)")