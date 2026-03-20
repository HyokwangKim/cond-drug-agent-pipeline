# Conditional Drug Candidate Generation Agent Pipeline

본 프로젝트는 환자의 임상 텍스트(EHR) 분석, 대형 언어 모델(LLM) 기반의 화학적 구조 추론, 그리고 사전 학습된 분자 확산 모델(Diffusion Model)의 Training-Free 제어 기술을 융합한 **조건부 신약 후보 물질 생성 에이전트 파이프라인**입니다.

## 📌 Architecture / 핵심 단계

본 파이프라인은 3개의 주요 Phase로 구성됩니다.

1. **Phase 1: Clinical RAG & Multi-Agent Reasoning**
   - MIMIC-III 퇴원 요약지를 분석하여 질환 정보와 분자적 제약 조건(예: 분자량, 통과 장벽 등)을 추출합니다.
   - LangChain을 활용한 다중 에이전트(Clinical Analysis, Pharmacological Mapping, Chemical Constraint) 시스템을 구축합니다.

2. **Phase 2: Training-Free Diffusion Control**
   - 파인튜닝 없이 사전 학습된 LDMol(Latent Diffusion Model)의 생성 궤적을 제어합니다.
   - Tweedie's formula 및 FlowAlign 기술을 적용하여 제약 조건을 수학적으로 강제합니다.

3. **Phase 3: Hybrid Verification & Structural Reasoning**
   - RDKit의 비미분성 한계를 극복하기 위한 대리 모델(Surrogate Model)을 활용합니다.
   - 명시적 분자 구조 추론(MSR) 프레임워크를 결합하여 생성된 후보 물질의 화학적 타당성과 무결성을 검증합니다.

## 🚀 Getting Started

### 1. 환경 설정

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt