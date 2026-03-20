# 💊 Conditional Drug Candidate Generation Agent Pipeline

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

---

## 📂 Project Structure & Code Description

프로젝트의 각 모듈은 독립적으로 작동하며, 이전 단계의 출력을 다음 단계의 입력으로 사용합니다.

### 📍 `phase1_clinical_rag/` (임상 데이터 분석 및 제약 조건 생성)
* **`data_ingestion.py`**
  * **역할:** 임상 기록(MIMIC-III)에 포함된 극심한 의료 약어와 오탈자를 LLM(Zero-shot CoT)을 통해 해소하고, 문맥이 유지되도록 청킹(Chunking)하여 Vector DB에 저장합니다.
  * **Input:** 원본 임상 텍스트 (Raw text)
  * **Output:** 정제된 텍스트 및 ChromaDB 벡터 데이터
* **`batch_processor.py`**
  * **역할:** 1.1GB에 달하는 실제 MIMIC-III 파일(`NOTEEVENTS.csv.gz`)을 메모리 초과 없이 청크 단위로 불러와 '퇴원 요약지'만 필터링한 후 `data_ingestion.py`의 함수를 호출하여 대량 적재를 수행합니다.
  * **Input:** `NOTEEVENTS.csv.gz` 파일
  * **Output:** 환자별로 분리 및 정제된 `chroma_db_real` 디렉토리
* **`agent_graph.py`**
  * **역할:** LangGraph로 구축된 3명의 AI 에이전트가 순차적으로 추론을 진행합니다. (1) 환자 알레르기 분석 -> (2) 타겟 단백질 매핑 -> (3) 화학적 제약 조건 도출.
  * **Input:** 의사/사용자의 자연어 질의 (예: "이 환자의 기록을 분석해 줘")
  * **Output:** 확산 모델을 제어하기 위한 구조화된 JSON 데이터 (예: 최대 분자량, 회피해야 할 화학 구조 등)

### 📍 `phase2_diffusion/` (분자 생성)
* **`custom_sampler.py`**
  * **역할:** 사전 학습된 딥러닝 확산 모델(Diffusion Model)의 역방향 샘플링 과정에 외부 그래디언트를 주입하여, 생성되는 분자의 구조를 Phase 1의 제약 조건에 맞게 수학적으로 틀어줍니다. (현재 로컬용 뼈대 코드)
  * **Input:** Phase 1의 제약 조건(JSON) 및 초기 노이즈 잠재 벡터(Latent Vector)
  * **Output:** 최적화된 분자 구조 데이터(SMILES)

### 📍 `phase3_verification/` (검증 시스템)
* **`pipeline_mock.py`**
  * **역할:** 전체 파이프라인의 S/W 아키텍처를 검증하기 위한 통합 테스트 스크립트입니다. Phase 2를 더미 모델로 대체하고, RDKit을 이용해 최종 생성된 분자(SMILES)가 환자의 제약 조건(예: 특정 알레르기 유발 구조 포함 여부)을 위반하지 않았는지 깐깐하게 검증(Pass/Reject)합니다.
  * **Input:** 확산 모델이 생성한 분자 SMILES 문자열 및 Phase 1의 제약 조건(JSON)
  * **Output:** 후보 물질별 최종 합격(PASS) 및 폐기(REJECT) 사유 로그

---

## 🚀 Getting Started & Usage

### 1. 환경 설정 및 설치
본 프로젝트는 Python 가상환경 사용을 권장합니다.

```bash
# 저장소 클론 및 이동
git clone [repository_url]
cd cond-drug-agent-pipeline

# 가상환경 생성 및 활성화
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 필수 패키지 설치
pip install -r requirements.txt
pip install rdkit  # Phase 3 검증용 화학 라이브러리
```
### 2. API 키 설정
프로젝트 최상위 디렉토리에 .env 파일을 생성하고 아래와 같이 API 키를 입력합니다.

```Bash
GOOGLE_API_KEY="당신의-구글-AI-STUDIO-API-키"
```
### 3. 실행 순서 (How to Run)
Step 1. Vector DB 구축 (지식 기반 생성)

```Bash
# 샘플 데이터로 테스트할 경우
python phase1_clinical_rag/data_ingestion.py

# 실제 MIMIC-III 데이터셋으로 적재할 경우 (data/ 폴더에 NOTEEVENTS.csv.gz 필요)
python phase1_clinical_rag/batch_processor.py
Step 2. 멀티 에이전트 추론 (제약 조건 도출)
```
```Bash
python phase1_clinical_rag/agent_graph.py
# -> 실행 후 화학적 제약 조건이 담긴 JSON 결과물이 터미널에 출력됩니다.
Step 3. 파이프라인 통합 및 화학 구조 검증 (End-to-End 테스트)
```
```Bash
python phase3_verification/pipeline_mock.py
# -> 생성된 분자들이 RDKit을 통해 물리화학적 룰(Hard-rule)을 통과하는지 검증합니다.
```