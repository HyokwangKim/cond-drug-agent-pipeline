import os
import json
import re
import torch
from rdkit import Chem
from rdkit.Chem import Descriptors

class ConditionalGuidance:
    CHEMICAL_DICT = {
        "beta-lactam": "C1C(=O)NC1",
        "sulfonamide": "S(=O)(=O)N",
        "4-hydroxycoumarin": "c1ccccc1C2=C(O)C(=O)OC=C2",
        "aniline": "c1ccccc1N",
        "coumarin": "O=C1OC2=CC=CC=C2C=C1",
        "penicillin": "CC1(C(N2C(S1)C(C2=O)NC(=O)CC3=CC=CC=C3)C(=O)O)C"
    }

    def __init__(self, json_path):
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"❌ 파일을 찾을 수 없습니다: {json_path}")
            
        with open(json_path, "r", encoding="utf-8") as f:
            self.constraints = json.load(f)
            
        print(f"\n✅ 제약 조건 로드 완료: {json_path}")
        
        rules = self.constraints.get("physicochemical_rules", {})
        # MW, MWT, molecular_weight 등 다양한 키워드에 대응
        mw_str = rules.get("molecular_weight", rules.get("MWT", rules.get("MW", "500")))
        
        # '300-550' 처럼 범위로 나올 경우 최댓값을 취함
        if isinstance(mw_str, str) and "-" in mw_str:
            mw_str = mw_str.split("-")[1]
            
        self.mw_limit = float(re.sub(r"[^\d.]", "", str(mw_str)))
        print(f"   - 분자량 제한(MW Limit): {self.mw_limit}")
        
        self.excluded_smarts = []
        raw_list = self.constraints.get("excluded_pharmacophores", [])
        
        print("🔍 화학 제약 조건 분석 및 매핑 중...")
        for item in raw_list:
            pattern = None
            
            # 1. 괄호 안의 코드 추출 시도 (예: "명칭 (SMILES: C1...)")
            match = re.search(r"\((?:SMILES|SMARTS):\s*([^)]+)\)", item, re.IGNORECASE)
            
            if match:
                pattern = match.group(1).strip()
            else:
                # 2. 괄호가 없다면, 문자열 자체가 코드인지 의심해봄
                item_clean = item.strip()
                # 공백이 없는 긴 문자열이라면 보통 SMILES 코드일 확률이 높음
                if " " not in item_clean and len(item_clean) > 3:
                    pattern = item_clean
                else:
                    # 3. 그것도 아니라면 사전에 있는지 확인
                    pattern = self.CHEMICAL_DICT.get(item_clean.lower())

            # 패턴이 존재하면 RDKit 객체화 시도
            if pattern:
                mol = Chem.MolFromSmarts(pattern) or Chem.MolFromSmiles(pattern)
                if mol:
                    self.excluded_smarts.append(mol)
                    print(f"   ✅ 구조 등록 성공: {pattern}")
                    continue
            
            print(f"   ⚠️ 무시됨 (코드 없음/해석 불가): '{item}'")

        print(f"\n🚀 총 {len(self.excluded_smarts)}개의 제약 조건이 가이드에 반영되었습니다.")

    def calculate_loss(self, smiles):
        mol = Chem.MolFromSmiles(smiles)
        if not mol: return 20.0
        loss = 0.0
        mw = Descriptors.MolWt(mol)
        if mw > self.mw_limit:
            loss += (mw - self.mw_limit) * 0.5
        for smarts in self.excluded_smarts:
            if mol.HasSubstructMatch(smarts):
                loss += 10.0
        return loss

if __name__ == "__main__":
    JSON_PATH = "./data/clinical_constraints.json"
    try:
        guidance = ConditionalGuidance(JSON_PATH)
    except Exception as e:
        print(f"❌ 에러: {e}")
        
    try:
        guidance = ConditionalGuidance(JSON_PATH)
        
        print("\n" + "="*50)
        print("🧪 [가상 약물 Loss 검증 테스트] 🧪")
        print("="*50)

        # 1. [Bad Drug] 로사르탄 (Losartan) 
        # - ARB 계열 고혈압 약. (금기 구조인 Tetrazole 포함)
        bad_smiles = "CCCCC1=NC(=C(N1CC2=CC=C(C=C2)C3=CC=CC=C3C4=NNN=N4)CO)Cl"
        bad_loss = guidance.calculate_loss(bad_smiles)
        print(f"🚫 [Losartan (ARB 계열)] Loss 점수: {bad_loss}")
        if bad_loss >= 10.0:
            print("   -> 훌륭합니다! 금기 구조를 정확히 적발하여 강력한 패널티를 부여했습니다.")

        print("-" * 50)

        # 2. [Good Drug] 파수딜 (Fasudil)
        # - ROCK 저해제. (우리가 원하는 타겟, 금기 구조 없음)
        good_smiles = "O=S(=O)(c1cccc2cnccc12)N3CCNCC3"
        good_loss = guidance.calculate_loss(good_smiles)
        print(f"✅ [Fasudil (ROCK 저해제)] Loss 점수: {good_loss}")
        if good_loss == 0.0:
            print("   -> 완벽합니다! 제약 조건을 모두 통과하여 패널티가 0입니다.")
            
    except Exception as e:
        print(f"❌ 에러: {e}")