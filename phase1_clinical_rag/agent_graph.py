# phase1_clinical_rag/agent_graph.py
from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
try:
    from langchain_chroma import Chroma  # pyright: ignore[reportMissingImports]
except ImportError:  # pragma: no cover - fallback for older envs
    from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field
from rdkit import Chem

from cdp.io_trace import TraceEvent, write_trace_event

logger = logging.getLogger(__name__)
PHASE1_LLM_MODEL = "gemma-3-27b-it"
NOISY_SUBSTRUCTURE_PATTERNS: tuple[str, ...] = ("[C(=O)O]", "[COOH]", "[NH2]", "[OH]")

GraphState = dict[str, Any]


class ClinicalInfo(BaseModel):
    primary_diagnosis: str = Field(description="환자의 주 질환 및 현재 상태")
    clinical_context: str = Field(description="병력 및 현재 증상의 의학적 요약")
    critical_contraindications: list[str] = Field(description="반드시 피해야 할 약물 성분 또는 계열")
    reasoning: str = Field(description="추출된 정보의 의학적 근거")


class TargetInfo(BaseModel):
    proposed_target_protein: str = Field(description="제안하는 약물 타겟 단백질 (공식 명칭)")
    mechanism_of_action: str = Field(description="해당 타겟을 조절했을 때의 치료 기전")
    rationale: str = Field(description="환자의 임상 상태를 고려한 타겟 선정 이유")


class ChemicalConstraints(BaseModel):
    physicochemical_rules: dict[str, str] = Field(description="분자량, LogP, HBD/HBA 등 물리화학적 제약")
    excluded_pharmacophores: list[str] = Field(description="알레르기/독성 우려로 제외해야 할 부분 구조(SMILES 또는 명칭)")
    structural_priority: str = Field(description="확산 모델이 집중해야 할 구조적 특징 (예: Hydrophobicity 강화)")


def _extract_llm_text(content: Any) -> str:
    """LLM 응답 content를 사람이 읽을 수 있는 문자열로 정규화합니다."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(str(item) for item in content)
    return str(content)


def _trace_prompt_exchange(
    trace_dir: Path,
    phase: str,
    prompt_input: dict[str, Any],
    rendered_prompt: str,
    llm_output_text: str,
    parsed_output: dict[str, Any],
) -> None:
    write_trace_event(
        trace_dir=trace_dir,
        event=TraceEvent(
            phase=phase,
            stage="prompt_exchange",
            payload={
                "prompt_input": prompt_input,
                "rendered_prompt": rendered_prompt,
                "llm_output_text": llm_output_text,
                "parsed_output": parsed_output,
            },
        ),
        max_chars=3000,
    )


def _print_prompt_preview(phase: str, rendered_prompt: str, llm_output_text: str) -> None:
    """콘솔에서 프롬프트 입출력 원문 전체를 확인합니다."""
    prompt_preview = rendered_prompt.replace("\n", " ")
    output_preview = llm_output_text.replace("\n", " ")
    print(f"[Prompt Preview][{phase}] rendered_prompt[full]: {prompt_preview}")
    print(f"[Prompt Preview][{phase}] llm_output_text[full]: {output_preview}")


def _is_valid_substructure_pattern(pattern: str) -> bool:
    if not pattern or not pattern.strip():
        return False
    token = pattern.strip()
    if token in NOISY_SUBSTRUCTURE_PATTERNS:
        return False
    # bracket 안에 괄호/등호가 섞인 축약 표기는 RDKit 파싱 오류를 자주 유발합니다.
    if re.search(r"\[[^\]]*[()=][^\]]*\]", token):
        return False
    return bool(Chem.MolFromSmarts(token) or Chem.MolFromSmiles(token))


def _sanitize_chemical_constraints(result: dict[str, Any]) -> dict[str, Any]:
    """Phase2/3에서 RDKit 파싱 오류를 내는 금기 패턴을 제거합니다."""
    excluded = result.get("excluded_pharmacophores", [])
    if not isinstance(excluded, list):
        return result

    filtered = [str(item).strip() for item in excluded if _is_valid_substructure_pattern(str(item))]
    removed_count = len(excluded) - len(filtered)
    if removed_count > 0:
        logger.warning("invalid excluded_pharmacophores removed=%s", removed_count)

    return {
        **result,
        "excluded_pharmacophores": filtered,
    }


def _make_clinical_analysis_agent(llm: ChatGoogleGenerativeAI, retriever: Any | None, trace_dir: Path):
    def clinical_analysis_agent(state: GraphState) -> GraphState:
        print("\n🔍 [Agent 1] 중환자 임상 데이터 심층 분석 중...")
        logger.info("clinical_analysis_agent entry query=%s", state.get("query", "")[:120])
        docs = retriever.invoke(state["query"]) if retriever is not None else []
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

        prompt_input = {"context": context}
        rendered_prompt = prompt.format(**prompt_input)
        llm_message = (prompt | llm).invoke(prompt_input)
        llm_output_text = _extract_llm_text(getattr(llm_message, "content", llm_message))
        _print_prompt_preview("agent1_clinical_analysis", rendered_prompt, llm_output_text)
        result = parser.parse(llm_output_text)
        _trace_prompt_exchange(
            trace_dir=trace_dir,
            phase="phase1_agent1_clinical_analysis",
            prompt_input=prompt_input,
            rendered_prompt=rendered_prompt,
            llm_output_text=llm_output_text,
            parsed_output=result,
        )
        logger.info("clinical_analysis_agent exit primary=%s", result.get("primary_diagnosis", "")[:80])
        return {**state, "clinical_analysis": result}

    return clinical_analysis_agent


def _make_pharmacological_mapping_agent(llm: ChatGoogleGenerativeAI, trace_dir: Path):
    def pharmacological_mapping_agent(state: GraphState) -> GraphState:
        print("\n🧬 [Agent 2] 분자 표적 및 치료 기전 매핑 중...")
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

        prompt_input = {
            "diagnosis": clinical["primary_diagnosis"],
            "contra": ", ".join(clinical["critical_contraindications"]),
            "context": clinical["clinical_context"],
        }
        rendered_prompt = prompt.format(**prompt_input)
        llm_message = (prompt | llm).invoke(prompt_input)
        llm_output_text = _extract_llm_text(getattr(llm_message, "content", llm_message))
        _print_prompt_preview("agent2_target_mapping", rendered_prompt, llm_output_text)
        result = parser.parse(llm_output_text)
        _trace_prompt_exchange(
            trace_dir=trace_dir,
            phase="phase1_agent2_target_mapping",
            prompt_input=prompt_input,
            rendered_prompt=rendered_prompt,
            llm_output_text=llm_output_text,
            parsed_output=result,
        )
        return {**state, "target_protein": result}

    return pharmacological_mapping_agent


def _make_chemical_constraint_agent(llm: ChatGoogleGenerativeAI, trace_dir: Path):
    def chemical_constraint_agent(state: GraphState) -> GraphState:
        print("\n🧪 [Agent 3] 확산 모델 제어용 화학 제약 조건 생성 중...")
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

        prompt_input = {
            "target_name": target["proposed_target_protein"],
            "allergies": ", ".join(clinical["critical_contraindications"]),
        }
        rendered_prompt = prompt.format(**prompt_input)
        llm_message = (prompt | llm).invoke(prompt_input)
        llm_output_text = _extract_llm_text(getattr(llm_message, "content", llm_message))
        _print_prompt_preview("agent3_chemical_constraints", rendered_prompt, llm_output_text)
        result = _sanitize_chemical_constraints(parser.parse(llm_output_text))
        _trace_prompt_exchange(
            trace_dir=trace_dir,
            phase="phase1_agent3_chemical_constraints",
            prompt_input=prompt_input,
            rendered_prompt=rendered_prompt,
            llm_output_text=llm_output_text,
            parsed_output=result,
        )
        return {**state, "chemical_constraints": result}

    return chemical_constraint_agent


def build_clinical_agent_app(retriever: Any | None, trace_dir: Path) -> Any:
    """RAG 리트리버를 주입하여 LangGraph 앱을 컴파일합니다."""
    load_dotenv()
    llm = ChatGoogleGenerativeAI(model=PHASE1_LLM_MODEL, temperature=0)

    workflow = StateGraph(GraphState)
    workflow.add_node("clinical_analyzer", _make_clinical_analysis_agent(llm, retriever, trace_dir))
    workflow.add_node("pharmacological_mapper", _make_pharmacological_mapping_agent(llm, trace_dir))
    workflow.add_node("chemical_constrainer", _make_chemical_constraint_agent(llm, trace_dir))

    workflow.set_entry_point("clinical_analyzer")
    workflow.add_edge("clinical_analyzer", "pharmacological_mapper")
    workflow.add_edge("pharmacological_mapper", "chemical_constrainer")
    workflow.add_edge("chemical_constrainer", END)
    return workflow.compile()


def run_phase1_clinical(query: str, chroma_persist_dir: Path) -> GraphState:
    """
    Phase 1 전체 그래프를 실행하고 최종 state를 반환합니다.

    Args:
        query: RAG 검색 및 에이전트에 전달할 자연어 질의.
        chroma_persist_dir: Chroma persist 디렉터리 (없으면 컨텍스트 없이 LLM만 동작).

    Returns:
        LangGraph 최종 state (``chemical_constraints`` 포함).
    """
    if not query or not query.strip():
        raise ValueError("query는 비어 있으면 안 됩니다.")

    chroma_persist_dir = chroma_persist_dir.resolve()
    load_dotenv()
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-2-preview")

    retriever = None
    retriever_preview: list[str] = []
    if chroma_persist_dir.is_dir():
        vectorstore = Chroma(persist_directory=str(chroma_persist_dir), embedding_function=embeddings)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
        logger.info("Chroma 로드 완료 path=%s", chroma_persist_dir)
        try:
            docs = retriever.invoke(query)
            retriever_preview = [str(doc.page_content)[:300] for doc in docs[:2]]
        except (ValueError, RuntimeError, OSError) as e:
            logger.warning("retriever preview 실패 err=%s", e)
    else:
        logger.warning("Chroma 경로 없음 — RAG 없이 진행합니다: %s", chroma_persist_dir)

    project_root = Path(__file__).resolve().parents[1]
    trace_dir = project_root / "data" / "runs" / "io_trace"
    write_trace_event(
        trace_dir=trace_dir,
        event=TraceEvent(
            phase="phase1",
            stage="retrieval_input",
            payload={
                "query": query,
                "chroma_persist_dir": str(chroma_persist_dir),
                "retrieved_doc_preview": retriever_preview,
            },
        ),
        max_chars=500,
    )

    app = build_clinical_agent_app(retriever, trace_dir)
    initial_state: GraphState = {"query": query}
    logger.info("Phase1 invoke start")
    final_state = app.invoke(initial_state)
    logger.info("Phase1 invoke done keys=%s", list(final_state.keys()))
    write_trace_event(
        trace_dir=trace_dir,
        event=TraceEvent(
            phase="phase1",
            stage="graph_output",
            payload={
                "keys": list(final_state.keys()),
                "clinical_analysis": final_state.get("clinical_analysis"),
                "target_protein": final_state.get("target_protein"),
                "chemical_constraints": final_state.get("chemical_constraints"),
            },
        ),
        max_chars=500,
    )
    return final_state


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    db = root / "data" / "chroma_db_real"
    q = (
        "Find ICU discharge summaries for a patient with chronic kidney disease and hyperkalemia risk, "
        "especially with ACE inhibitor or ARB intolerance. Analyze the case and derive safe chemical "
        "constraints for a new mechanism antihypertensive candidate."
    )
    print("\n🚀 [Phase 1] 멀티 에이전트 파이프라인 가동 🚀")
    final = run_phase1_clinical(query=q, chroma_persist_dir=db)

    print("\n==================================================")
    print("✅ [최종 도출된 분자 확산 모델 제어용 JSON 조건] ✅")
    print("==================================================")
    print(json.dumps(final["chemical_constraints"], indent=4, ensure_ascii=False))

    out_dir = root / "data"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "clinical_constraints.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(final["chemical_constraints"], f, indent=4, ensure_ascii=False)
    print(f"\n✅ 분석 완료! 저장: {out_path.resolve()}")
