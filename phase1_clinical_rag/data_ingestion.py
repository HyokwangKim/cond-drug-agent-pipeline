# phase1_clinical_rag/data_ingestion.py
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

# 1. 환경변수 로드 (.env의 GOOGLE_API_KEY 자동 인식)
load_dotenv()

# 2. Gemini LLM 및 임베딩 모델 초기화
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-2-preview")

def preprocess_clinical_text(raw_text: str) -> str:
    """
    Zero-Shot Chain-of-Thought (CoT) 기법을 사용하여
    임상 기록의 모호한 약어(Abbreviations)를 문맥에 맞게 풀어서 변환합니다.
    """
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
        """
    )
    
    # LangChain LCEL 문법으로 파이프라인 구성
    chain = prompt | llm | StrOutputParser()
    cleaned_text = chain.invoke({"clinical_text": raw_text})
    
    return cleaned_text

def build_vector_db(cleaned_text: str, db_path: str = "./chroma_db"):
    """
    정제된 텍스트를 청킹(Chunking)하여 Chroma Vector DB에 임베딩 및 저장합니다.
    """
    print("--- [DB 저장] 텍스트 청킹 및 Vector DB(Chroma) 구축 중... ---")
    
    # 계층적 청킹을 위한 설정 (문맥이 끊기지 않도록 overlap 부여)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", ".", " "]
    )
    
    chunks = text_splitter.split_text(cleaned_text)
    
    # 메타데이터 주입 (실제 프로젝트에서는 섹션 정보나 시간 순서를 여기에 넣습니다)
    documents = [Document(page_content=chunk, metadata={"source": "MIMIC-III_discharge_summary", "chunk_index": i}) for i, chunk in enumerate(chunks)]
    
    # ChromaDB에 임베딩 저장
    vectorstore = Chroma.from_documents(
        documents=documents, 
        embedding=embeddings, 
        persist_directory=db_path
    )
    
    print(f"성공! 총 {len(chunks)}개의 청크가 {db_path}에 저장되었습니다.")
    return vectorstore

if __name__ == "__main__":
    # 테스트용 가상의 MIMIC-III 원본 텍스트 (의도적으로 약어 포함)
    sample_raw_mimic_text = """
    Pt is a 65 yo M with hx of HTN, DM2, admitted for SOB and CP.
    CXR showed b/l infiltrates. Started on IV CTX and azithro for CAP.
    Pt developed severe rash and wheezing after CTX. Transferred to ICU.
    Current meds: ASA 81mg PO daily, Lisinopril 10mg PO daily.
    """
    
    print("원본 텍스트:\n", sample_raw_mimic_text)
    
    # 1. 전처리 실행
    cleaned_text = preprocess_clinical_text(sample_raw_mimic_text)
    print("\n정제된 텍스트:\n", cleaned_text)
    
    # 2. Vector DB 구축 실행
    db = build_vector_db(cleaned_text, db_path="./data/chroma_db")