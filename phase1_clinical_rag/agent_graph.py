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
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-2-preview")

# 3. Vector DB 로드 (방금 만든 chroma_db 연결)
db_path = "./data/chroma_db"
if os.path.exists(db_path):
    vectorstore = Chroma(persist_directory=db_path, embedding_function=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2}) # 가장 관련성 높은 2개 청크 검색
else:
    print("경고: Chroma DB를 찾을 수 없습니다. data_ingestion.py를 먼저 실행하세요.")
    retriever = None

# 4. 에이전트 상태(State) 및 JSON 출력 스키마(Pydantic) 정의
class GraphState(TypedDict):
    query: str
    clinical_analysis: dict
    target_protein: dict
    chemical_constraints: dict

class ClinicalInfo(BaseModel):
    clinical_phenotype: str = Field(description="환자의 주 질환 (예: Treatment-resistant infection)")
    allergies: List[str] = Field(description="약물 알레르기 또는 부작용 목록 (예: Ceftriaxone)")

class TargetInfo(BaseModel):
    target_protein: str = Field(description="추천되는 약리학적 표적 단백질 이름")
    uniprot_id: str = Field(description="가상의 UniProt ID")

class ChemicalConstraints(BaseModel):
    max_molecular_weight: int = Field(description="최대 분자량 (일반적으로 500~600)")
    avoid_substructures: List[str] = Field(description="알레르기 기반으로 피해야 할 화학 구조 (예: beta-lactam ring)")

# 5. 에이전트 노드 구현
def clinical_analysis_agent(state: GraphState):
    print("\n--- [Agent 1] 임상 기록 검색(RAG) 및 분석 중 ---")
    
    # DB에서 관련된 환자 기록 검색
    docs = retriever.invoke(state["query"]) if retriever else []
    context = "\n".join([doc.page_content for doc in docs])
    
    parser = JsonOutputParser(pydantic_object=ClinicalInfo)
    prompt = PromptTemplate(
        template="당신은 임상 분석가입니다. 다음 환자 기록을 보고 주 질환(phenotype)과 부작용/알레르기(allergies)를 추출하세요.\n\n[기록]:\n{context}\n\n{format_instructions}",
        input_variables=["context"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )
    
    result = (prompt | llm | parser).invoke({"context": context})
    print(f" -> 분석 완료: {result}")
    return {"clinical_analysis": result}

def pharmacological_mapping_agent(state: GraphState):
    print("\n--- [Agent 2] 약리학적 타겟 단백질 매핑 중 ---")
    
    clinical_data = state["clinical_analysis"]
    parser = JsonOutputParser(pydantic_object=TargetInfo)
    prompt = PromptTemplate(
        template="당신은 약리학자입니다. 환자의 질환({phenotype})을 치료하기 위한 최적의 타겟 단백질을 제안하세요. 단, 환자의 부작용({allergies})과 겹치지 않는 기전이어야 합니다.\n\n{format_instructions}",
        input_variables=["phenotype", "allergies"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )
    
    result = (prompt | llm | parser).invoke({
        "phenotype": clinical_data["clinical_phenotype"], 
        "allergies": ", ".join(clinical_data["allergies"])
    })
    print(f" -> 타겟 도출: {result}")
    return {"target_protein": result}

def chemical_constraint_agent(state: GraphState):
    print("\n--- [Agent 3] 화학적 제약 조건(JSON) 생성 중 ---")
    
    target_data = state["target_protein"]
    clinical_data = state["clinical_analysis"]
    parser = JsonOutputParser(pydantic_object=ChemicalConstraints)
    
    prompt = PromptTemplate(
        template="당신은 화학정보학자입니다. 타겟 단백질({target})과 환자의 알레르기({allergies})를 고려하여, 신약이 가져야 할 화학적 제약 조건을 설정하세요.\n알레르기를 유발할 수 있는 구조(예: 항생제 알레르기면 beta-lactam 등)를 회피 구조에 포함하세요.\n\n{format_instructions}",
        input_variables=["target", "allergies"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )
    
    result = (prompt | llm | parser).invoke({
        "target": target_data["target_protein"],
        "allergies": ", ".join(clinical_data["allergies"])
    })
    return {"chemical_constraints": result}

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
    initial_state = {"query": "이 환자의 기록을 분석해서 대체 투여할 수 있는 신약 후보 물질의 조건을 알려줘."}
    
    print("🚀 조건부 신약 후보 물질 생성 파이프라인 가동 🚀")
    final_state = app.invoke(initial_state)
    
    print("\n==================================================")
    print("✅ [최종 도출된 분자 확산 모델 제어용 JSON 조건] ✅")
    print("==================================================")
    import json
    print(json.dumps(final_state["chemical_constraints"], indent=4, ensure_ascii=False))