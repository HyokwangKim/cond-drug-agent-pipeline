# phase1_clinical_rag/agent_graph.py
import os
from dotenv import load_dotenv
from typing import Dict, TypedDict, List
from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field

# 1. 환경변수 로드
load_dotenv()

# 2. LLM 및 임베딩 모델 (최신 버전 적용 완료)
llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro", temperature=0) # flash
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-2-preview")

# 3. Vector DB 로드 (방금 만든 chroma_db 연결)
db_path = "./data/chroma_db_real"
if os.path.exists(db_path):
    vectorstore = Chroma(persist_directory=db_path, embedding_function=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2}) # 가장 관련성 높은 2개 청크 검색
else:
    print("경고: Chroma DB를 찾을 수 없습니다. data_ingestion.py를 먼저 실행하세요.")
    retriever = None

GraphState = dict
# 4. 출력 스키마 고도화 (이유/근거 필드 추가)
class ClinicalInfo(BaseModel):
    primary_diagnosis: str = Field(description="환자의 주 질환 및 현재 상태")
    clinical_context: str = Field(description="병력 및 현재 증상의 의학적 요약")
    critical_contraindications: List[str] = Field(description="반드시 피해야 할 약물 성분 또는 계열")
    reasoning: str = Field(description="추출된 정보의 의학적 근거")

class TargetInfo(BaseModel):
    proposed_target_protein: str = Field(description="제안하는 약물 타겟 단백질 (공식 명칭)")
    mechanism_of_action: str = Field(description="해당 타겟을 조절했을 때의 치료 기전")
    rationale: str = Field(description="환자의 임상 상태를 고려한 타겟 선정 이유")

class ChemicalConstraints(BaseModel):
    physicochemical_rules: Dict[str, str] = Field(description="분자량, LogP, HBD/HBA 등 물리화학적 제약")
    excluded_pharmacophores: List[str] = Field(description="알레르기/독성 우려로 제외해야 할 부분 구조(SMILES 또는 명칭)")
    structural_priority: str = Field(description="확산 모델이 집중해야 할 구조적 특징 (예: Hydrophobicity 강화)")

# 5. 에이전트 노드 구현 (프롬프트 고도화)

def clinical_analysis_agent(state: GraphState):
    print("\n🔍 [Agent 1] 중환자 임상 데이터 심층 분석 중...")
    print("[입력 예시] clinical_analysis_agent state:")
    print(state)
    docs = retriever.invoke(state["query"]) if retriever else []
    context = "\n".join([doc.page_content for doc in docs])
    
    parser = JsonOutputParser(pydantic_object=ClinicalInfo)
    prompt = PromptTemplate(
        template="""당신은 세계 최고의 임상 의학 전문가입니다. 
제공된 중환자실(ICU) 퇴원 요약지를 분석하여 신약 설계의 기초가 될 환자 프로필을 작성하세요.

[분석 지침]:
1. 복합적인 증상 중에서 '신약 치료가 필요한 핵심 질환'을 식별하세요.
2. 과거 약물 반응 기록을 추적하여 알레르기뿐만 아니라 '부작용 우려 계열'까지 확장하여 식별하세요.
3. ICU 기록의 특성상 수치적 변화(혈압, 염증 수치 등)가 암시하는 병태생리학적 상태를 요약하세요.

[임상 기록]:
{context}

{format_instructions}
""",
        input_variables=["context"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )
    
    result = (prompt | llm | parser).invoke({"context": context})
    print("[출력 예시] clinical_analysis_agent result:")
    print(result)
    return {**state, "clinical_analysis": result}

def pharmacological_mapping_agent(state: GraphState):
    print("\n🧬 [Agent 2] 분자 표적 및 치료 기전 매핑 중...")
    print("[입력 예시] pharmacological_mapping_agent state:")
    print(state)
    clinical = state["clinical_analysis"]
    
    parser = JsonOutputParser(pydantic_object=TargetInfo)
    prompt = PromptTemplate(
        template="""당신은 신약 개발 초기의 타겟 발굴(Target Identification) 전문가입니다.
임상 데이터 분석 결과를 바탕으로, 환자에게 가장 효과적이면서 안전한 분자적 타겟을 선정하세요.

[전략 가이드]:
1. 환자의 질환({diagnosis})에 대한 최신 표준 치료법을 고려하세요.
2. 환자가 피해야 할 약물({contra})과 교차 반응이 없는 완전히 새로운 기전 또는 안전한 기전의 타겟을 선정하세요.
3. 선정된 타겟이 왜 이 환자의 병태생리에 적합한지 논리적으로 설명하세요.



[환자 상태 요약]: {context}

{format_instructions}
""",
        input_variables=["diagnosis", "contra", "context"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )
    
    result = (prompt | llm | parser).invoke({
        "diagnosis": clinical["primary_diagnosis"],
        "contra": ", ".join(clinical["critical_contraindications"]),
        "context": clinical["clinical_context"]
    })
    print("[출력 예시] pharmacological_mapping_agent result:")
    print(result)
    return {**state, "target_protein": result}

def chemical_constraint_agent(state: GraphState):
    print("\n🧪 [Agent 3] 확산 모델 제어용 화학 제약 조건 생성 중...")
    print("[입력 예시] chemical_constraint_agent state:")
    print(state)
    target = state["target_protein"]
    clinical = state["clinical_analysis"]
    
    parser = JsonOutputParser(pydantic_object=ChemicalConstraints)
    prompt = PromptTemplate(
        template="""당신은 계산 화학자입니다. 확산 모델 제어용 제약 조건을 JSON으로 생성하세요.

    [Constraint Generation Rule]:
    1. 모든 화학 구조는 RDKit으로 파싱 가능한 유효한 SMARTS 또는 SMILES 형식이어야 합니다.
    2. 괄호 '(', ')'와 고리 번호(1, 2...)의 쌍이 반드시 맞아야 합니다.
    3. 부분 구조(Substructure)를 정의할 때는 반드시 완성된 형태의 SMARTS를 사용하세요. 
    - 잘못된 예: S(=O  (괄호 미포함)
    - 올바른 예: S(=O)=O, [N+](=O)[O-]
    4. 확신이 없는 구조는 생략하고, 확실히 유효한 코드만 JSON 배열에 담으세요.

    타겟 단백질: {target_name}
    환자 알레르기/금기: {allergies}

    {format_instructions}""",
        input_variables=["target_name", "allergies"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
        )
    
    result = (prompt | llm | parser).invoke({
        "target_name": target["proposed_target_protein"],
        "allergies": ", ".join(clinical["critical_contraindications"])
    })
    print("[출력 예시] chemical_constraint_agent result:")
    print(result)
    return {**state, "chemical_constraints": result}

# ... (나머지 그래프 연결 로직은 동일)

# 6. 워크플로우(LangGraph) 연결
workflow = StateGraph(GraphState)

workflow.add_node("clinical_analyzer", clinical_analysis_agent)
workflow.add_node("pharmacological_mapper", pharmacological_mapping_agent)
workflow.add_node("chemical_constrainer", chemical_constraint_agent)

workflow.set_entry_point("clinical_analyzer")
workflow.add_edge("clinical_analyzer", "pharmacological_mapper")
workflow.add_edge("pharmacological_mapper", "chemical_constrainer")
workflow.add_edge("chemical_constrainer", END)

app = workflow.compile()

if __name__ == "__main__":
    # 쿼리를 구체화하여 RAG가 알레르기 기록을 찾아오도록 유도합니다.
    initial_state = {
        "query": "Amlodipine에 알레르기가 있는 환자의 기록을 찾아 분석하고, 이를 대체하여 투여할 수 있는 새로운 기전의 신약 후보 물질 제약 조건을 도출해줘."
    }
    print("[초기 입력 예시] initial_state:")
    print(initial_state)
    print("\n🚀 [Phase 1] 멀티 에이전트 파이프라인 가동 (Target: CCB Allergy Patient) 🚀")
    final_state = app.invoke(initial_state)
    print("\n==================================================")
    print("✅ [최종 도출된 분자 확산 모델 제어용 JSON 조건] ✅")
    print("==================================================")
    import json
    print(json.dumps(final_state["chemical_constraints"], indent=4, ensure_ascii=False))
    # --- [수정] 결과 저장 로직 추가 ---
    output_dir = "./data"
    os.makedirs(output_dir, exist_ok=True) # 폴더가 없으면 생성
    
    output_path = os.path.join(output_dir, "clinical_constraints.json")
    
    with open(output_path, "w", encoding="utf-8") as f:
        # Agent 3의 최종 결과물(chemical_constraints)을 저장
        json.dump(final_state["chemical_constraints"], f, indent=4, ensure_ascii=False)
        
    print(f"\n✅ 분석 완료! 결과가 다음 경로에 저장되었습니다: {os.path.abspath(output_path)}")