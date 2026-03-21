import os
import json
import re
import torch
import torch.nn as nn
from rdkit import Chem
from rdkit.Chem import Descriptors
from tqdm import tqdm

# =====================================================================
# 1. 제약 조건 파싱 모듈 (Phase 1 결과물 연동)
# =====================================================================
class ConditionalGuidance:
    CHEMICAL_DICT = {
        "beta-lactam": "C1C(=O)NC1",
        "sulfonamide": "S(=O)(=O)N",
        "4-hydroxycoumarin": "c1ccccc1C2=C(O)C(=O)OC=C2",
        "aniline": "c1ccccc1N",
        "coumarin": "O=C1OC2=CC=CC=C2C=C1"
    }

    def __init__(self, json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            self.constraints = json.load(f)
            
        rules = self.constraints.get("physicochemical_rules", {})
        mw_str = rules.get("molecular_weight", rules.get("MWT", rules.get("MW", "500")))
        if isinstance(mw_str, str) and "-" in mw_str:
            mw_str = mw_str.split("-")[1]
        self.mw_limit = float(re.sub(r"[^\d.]", "", str(mw_str)))
        
        self.excluded_smarts = []
        for item in self.constraints.get("excluded_pharmacophores", []):
            match = re.search(r"\((?:SMILES|SMARTS):\s*([^)]+)\)", item, re.IGNORECASE)
            pattern = match.group(1).strip() if match else item.strip()
            if " " not in pattern and len(pattern) > 3: pass
            else: pattern = self.CHEMICAL_DICT.get(pattern.lower())
                
            if pattern:
                mol = Chem.MolFromSmarts(pattern) or Chem.MolFromSmiles(pattern)
                if mol: self.excluded_smarts.append(mol)

    def calculate_rdkit_loss(self, smiles):
        """(평가용) RDKit 기반 하드 룰 Loss 계산"""
        mol = Chem.MolFromSmiles(smiles)
        if not mol: return 20.0
        loss = 0.0
        if Descriptors.MolWt(mol) > self.mw_limit: loss += 5.0
        for smarts in self.excluded_smarts:
            if mol.HasSubstructMatch(smarts): loss += 10.0
        return loss

# =====================================================================
# 2. 미분 가능한 대리 모델 (Differentiable Surrogate Model)
# =====================================================================
class PropertySurrogate(nn.Module):
    """
    잠재 벡터(z0)를 입력받아 제약 조건 위반 페널티를 미분 가능한 형태로 출력하는 네트워크.
    (Training-Free 실행에서는 랜덤 초기화 또는 고정 surrogate로 사용할 수 있습니다.)
    """
    def __init__(self, latent_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.SiLU(),
            nn.Linear(512, 256),
            nn.SiLU(),
            nn.Linear(256, 1) # 제약 조건 위반 점수 (Loss)
        )
        
    def forward(self, z):
        return self.net(z)

# =====================================================================
# 3. 확산 모델 역방향 샘플링 로직 (Guided Reverse Sampling)
# =====================================================================
def guided_sampling(model, surrogate, initial_noise, num_steps, guidance_scale=0.1):
    """
    Tweedie's Formula를 이용해 생성 궤적을 제어하는 실전 샘플링 함수
    """
    device = initial_noise.device
    z_t = initial_noise.clone()
    
    # 시간에 따른 스케줄러 (예시용 선형 스케줄러)
    betas = torch.linspace(0.0001, 0.02, num_steps).to(device)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    
    print("\n🔬 [Phase 2] 확산 모델 가이던스 샘플링 시작...")
    
    for t in tqdm(reversed(range(num_steps)), total=num_steps, desc="Denoising"):
        t_tensor = torch.tensor([t], device=device)
        
        # 1. 계산 그래프 연결 (Gradient 계산을 위함)
        z_t = z_t.detach().requires_grad_(True)
        
        with torch.enable_grad():
            # 2. 현재 노이즈에서 깨끗한 분자 잠재 벡터(z0) 예측
            # model.predict_x0는 사용하시는 LDMol의 구조에 맞게 메서드명을 변경해야 합니다.
            pred_z0 = model.predict_x0(z_t, t_tensor) 
            
            # 3. Surrogate 모델을 통한 미분 가능한 Loss 계산
            # (우리가 Phase 1에서 설정한 제약 조건을 위반할수록 값이 커짐)
            loss = surrogate(pred_z0).mean()
            
            # 4. 잠재 공간(z_t)에 대한 그래디언트 추출
            grad = torch.autograd.grad(loss, z_t)[0]
            
        # 5. 노이즈 제거 스텝 진행 (무조건 grad 없이 진행)
        with torch.no_grad():
            # 표준 DDPM/DDIM 스텝 (가상의 step 함수)
            z_prev = model.step(z_t, t_tensor) 
            
            # 6. 제약 조건 그래디언트 주입 (Guidance 주입)
            # Loss가 줄어드는 방향(-grad)으로 잠재 벡터를 밀어냅니다.
            z_t = z_prev - (guidance_scale * grad)
            
    return z_t.detach()

# =====================================================================
# 4. 메인 실행 블록
# =====================================================================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ 사용 장치: {device}")
    
    # 1. Phase 1 제약 조건 로드
    JSON_PATH = "./data/clinical_constraints.json"
    guidance = ConditionalGuidance(JSON_PATH)
    
    try:
        print("\n⏳ (가상) 확산 모델 및 Surrogate 네트워크 초기화 완료.")
        
        # 더미 모델 및 데이터 (실제 실행 시 위 주석의 실제 모델로 교체)
        class DummyModel(nn.Module):
            def predict_x0(self, z, t): return z * 0.9
            def step(self, z, t): return z * 0.95
            def decode(self, z): return ["O=S(=O)(c1cccc2cnccc12)N3CCNCC3"] # Fasudil (더미 출력)
            
        model = DummyModel().to(device)
        surrogate = PropertySurrogate(latent_dim=256).to(device)
        
        # 2. 초기 노이즈 생성 및 샘플링
        initial_noise = torch.randn(1, 256).to(device) # Batch=1, Latent=256
        
        final_latent = guided_sampling(
            model=model, 
            surrogate=surrogate, 
            initial_noise=initial_noise, 
            num_steps=100, 
            guidance_scale=0.15  # 가이드 강도
        )
        
        # 3. 디코딩 (Latent -> SMILES)
        generated_smiles_list = model.decode(final_latent)
        print("\n✨ 최종 생성된 분자 (SMILES):")
        for idx, sm in enumerate(generated_smiles_list):
            print(f"  [{idx+1}] {sm}")
            print(f"  -> RDKit 검증 Loss: {guidance.calculate_rdkit_loss(sm)}")
            
    except Exception as e:
        print(f"❌ 실행 중 오류 발생: {e}")