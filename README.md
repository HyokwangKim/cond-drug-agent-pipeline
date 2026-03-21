# Conditional Drug Candidate Generation Agent Pipeline

환자 임상 텍스트(EHR), LLM 기반 화학 제약 추론, 확산 모델 가이던스, RDKit 검증을 묶은 **Training-Free 조건부 신약 후보 생성** 파이프라인입니다.

## 통합 실행

저장소 루트에서 실행합니다.

```bash
# Mock: 고정 입력으로 단계 간 I/O 계약만 검증
python -m cdp.run --mode mock

# Real: data/NOTEEVENTS.csv.gz -> Chroma(없으면 생성) -> Phase1 -> Phase2 -> Phase3
python -m cdp.run --mode real
```

- Mock 산출물: `data/runs/mock/clinical_constraints.json`
- Real 산출물: `data/clinical_constraints.json`

## 핵심 설계

1. `cdp.run` 단일 진입점으로 phase 연결 순서를 고정했습니다.
2. 모든 내부 경로는 `project_root / data / ...` 상대 조합만 사용합니다.
3. Mock/Real 분리로, 빠른 구조 검증과 실제 데이터 실행을 분리했습니다.
4. Phase2는 학습 없이 동작하는 deterministic surrogate를 사용합니다.

## 필요 파일 / 환경

```bash
python -m venv venv
venv\Scripts\activate   # Windows
pip install -r requirements.txt
pip install rdkit
```

`.env`에 `GOOGLE_API_KEY`를 넣어야 Real 모드 Phase1/배치 전처리가 동작합니다.

## 모델 PT 관련

- **필수 아님**: 기본은 DummyDiffusionModel로 끝까지 실행됩니다.
- **선택**: 실제 확산 본체를 붙이려면 `data/weights/checkpoint_ldmol.pt`를 준비하고 `cdp/phase2_run.py`의 Dummy 모델 구간을 실제 모델 로더로 교체하세요.
- Surrogate 학습/학습 코드/학습 체크포인트 경로는 모두 제거되었습니다.
