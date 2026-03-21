# phase1_clinical_rag/data_ingestion.py
from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv
try:
    from langchain_chroma import Chroma
except ImportError:  # pragma: no cover - fallback for older envs
    from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


def preprocess_clinical_text(raw_text: str) -> str:
    """
    Zero-Shot Chain-of-Thought (CoT) 기법을 사용하여
    임상 기록의 모호한 약어(Abbreviations)를 LLM을 통해 문맥에 맞게 풀어서 변환합니다.
    """
    if raw_text is None or not str(raw_text).strip():
        raise ValueError("raw_text는 비어 있으면 안 됩니다.")

    load_dotenv()
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

    print("--- [전처리] Gemini가 의료 약어를 문맥 기반으로 해소(Disambiguation) 중... ---")

    prompt = PromptTemplate(
        input_variables=["clinical_text"],
        template="""
        당신은 최고 수준의 임상 데이터 분석가입니다.
        다음 중환자실(ICU) 임상 텍스트를 읽고, 극심한 의료 약어와 오탈자를 문맥(투여된 약물, 검사 수치 등)에 맞게 추론하여 완전한 단어로 치환하세요.
        Let's think step-by-step.

        [원본 텍스트]:
        {clinical_text}

        추론 과정은 생략하고, 오직 '약어가 해소되고 정제된 영문 텍스트'만 결과로 출력하세요.
        """,
    )

    chain = prompt | llm | StrOutputParser()
    return str(chain.invoke({"clinical_text": raw_text}))


def build_vector_db(cleaned_text: str, db_path: str | Path) -> Chroma:
    """
    정제된 텍스트를 청킹(Chunking)하여 Chroma Vector DB에 임베딩 및 저장합니다.
    """
    if not cleaned_text or not cleaned_text.strip():
        raise ValueError("cleaned_text는 비어 있으면 안 됩니다.")

    load_dotenv()
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-2-preview")

    print("--- [DB 저장] 텍스트 청킹 및 Vector DB(Chroma) 구축 중... ---")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", ".", " "],
    )

    chunks = text_splitter.split_text(cleaned_text)
    documents = [
        Document(page_content=chunk, metadata={"source": "MIMIC-III_discharge_summary", "chunk_index": i})
        for i, chunk in enumerate(chunks)
    ]

    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=str(path),
    )

    print(f"성공! 총 {len(chunks)}개의 청크가 {path}에 저장되었습니다.")
    return vectorstore


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]

    sample_raw_mimic_text = """
    Pt is a 65 yo M with hx of HTN, DM2, admitted for SOB and CP.
    CXR showed b/l infiltrates. Started on IV CTX and azithro for CAP.
    Pt developed severe rash and wheezing after CTX. Transferred to ICU.
    Current meds: ASA 81mg PO daily, Lisinopril 10mg PO daily.
    """

    print("원본 텍스트:\n", sample_raw_mimic_text)
    cleaned_text = preprocess_clinical_text(sample_raw_mimic_text)
    print("\n정제된 텍스트:\n", cleaned_text)

    _ = build_vector_db(cleaned_text, db_path=root / "data" / "chroma_db")
