import pandas as pd
import os
import time
from tqdm import tqdm
from data_ingestion import preprocess_clinical_text
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma

# --- 설정 ---
INPUT_FILE = "./data/NOTEEVENTS.csv.gz"
DB_PATH = "./data/chroma_db_real"
LOG_FILE = "./data/processed_ids.txt"  # 처리 완료된 ID 저장용
CHUNK_SIZE = 1000 
LIMIT_SAMPLES = 100 # 이제 속도가 빠르니 더 크게 잡으셔도 됩니다.

def load_processed_ids():
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, "r") as f:
            return set(line.strip() for line in f)
    return set()

def save_processed_id(row_id):
    with open(LOG_FILE, "a") as f:
        f.write(f"{row_id}\n")

def process_mimic_batch():
    processed_ids = load_processed_ids()
    print(f"--- [시작] 분석 재개 (기존 처리: {len(processed_ids)}건) ---")
    
    reader = pd.read_csv(
        INPUT_FILE, 
        compression='gzip', 
        chunksize=CHUNK_SIZE,
        usecols=['ROW_ID', 'CATEGORY', 'TEXT'] # ROW_ID를 고유 식별자로 사용
    )
    

    # Embedding 및 Vectorstore 초기화

    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-2-preview")
    vectorstore = None
    total_processed = 0

    for i, chunk in enumerate(reader):
        # Discharge summary만 필터링
        discharge_summaries = chunk[chunk['CATEGORY'] == 'Discharge summary']
        
        for _, row in discharge_summaries.iterrows():
            row_id = str(row['ROW_ID'])
            
            # [핵심] 이미 처리된 ID라면 패스
            if row_id in processed_ids:
                continue
            
            raw_text = row['TEXT']
            
            try:
                print(f"\n{'='*30} [ID: {row_id} 분석 중] {'='*30}")
                print(f"🔍 [Original]: {raw_text[:150].strip()}...")
                

                # Gemini 전처리 (Paid Tier 1의 속도로!)
                cleaned_text = preprocess_clinical_text(raw_text)
                if vectorstore is None:
                    # DB가 없을 때 첫 데이터로 생성
                    vectorstore = Chroma.from_texts(
                        texts=[cleaned_text],
                        embedding=embeddings,
                        metadatas=[{"row_id": row_id, "category": "Discharge summary"}],
                        persist_directory=DB_PATH
                    )
                else:
                    vectorstore.add_texts(
                        texts=[cleaned_text],
                        metadatas=[{"row_id": row_id, "category": "Discharge summary"}]
                    )
                vectorstore.persist()  # 각 케이스마다 즉시 저장
                save_processed_id(row_id)
                print(f"✨ [Cleaned]: {cleaned_text[:150].strip()}...")
                total_processed += 1
                time.sleep(0.1)
                if LIMIT_SAMPLES and total_processed >= LIMIT_SAMPLES:
                    break
                    
            except Exception as e:
                print(f"🚨 오류 발생 (ID {row_id}): {e}")
                # 혹시 모를 일시적 에러 시에만 약간 대기
                time.sleep(5)
                continue
        
        if LIMIT_SAMPLES and total_processed >= LIMIT_SAMPLES:
            break

    print(f"🎉 성공! 이번 세션에서 {total_processed}개를 새로 처리했습니다.")

if __name__ == "__main__":
    process_mimic_batch()