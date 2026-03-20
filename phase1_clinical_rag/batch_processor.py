# phase1_clinical_rag/batch_processor.py
import pandas as pd
import os
from data_ingestion import preprocess_clinical_text, build_vector_db

def process_mimic_notes(file_path, chunk_size=1000, target_category="Discharge summary", max_patients=5):
    """
    거대한 MIMIC-III NOTEEVENTS 파일을 메모리 안전하게 청크 단위로 읽어와
    특정 카테고리(퇴원 요약지)만 필터링한 후 RAG DB에 적재합니다.
    """
    print(f"📦 [{file_path}] 파일에서 데이터 추출을 시작합니다...")
    
    processed_count = 0
    combined_text = ""
    
    # 압축된 csv.gz 파일을 chunk 단위로 조금씩 읽어옵니다.
    chunk_iterator = pd.read_csv(
        file_path, 
        compression='gzip', 
        chunksize=chunk_size, 
        low_memory=False,
        usecols=['SUBJECT_ID', 'CATEGORY', 'TEXT'] # 메모리 절약을 위해 필요한 열만 가져옴
    )
    
    for chunk in chunk_iterator:
        # 1. 'Discharge summary(퇴원 요약지)'만 필터링
        discharge_summaries = chunk[chunk['CATEGORY'] == target_category]
        
        if discharge_summaries.empty:
            continue
            
        # 2. 지정한 환자 수(max_patients)만큼만 순회하며 텍스트 추출
        for index, row in discharge_summaries.iterrows():
            if processed_count >= max_patients:
                break
                
            patient_id = row['SUBJECT_ID']
            raw_text = row['TEXT']
            
            print(f"\n✅ [환자 ID: {patient_id}] 퇴원 요약지 발견! (길이: {len(raw_text)}자)")
            
            # 여기서 앞서 만든 Gemini 약어 해소 모듈을 호출합니다.
            # (비용과 시간을 고려하여 실제 텍스트의 앞부분 1000자만 잘라서 테스트합니다)
            truncated_text = raw_text[:1000] 
            cleaned_text = preprocess_clinical_text(truncated_text)
            
            # 환자 구분선을 넣어 하나의 큰 텍스트로 합칩니다.
            combined_text += f"\n\n[Patient ID: {patient_id} Record]\n{cleaned_text}"
            processed_count += 1
            
        if processed_count >= max_patients:
            break

    if combined_text:
        print("\n🚀 지정된 환자 수의 데이터 전처리가 완료되었습니다. Vector DB 구축을 시작합니다.")
        # 합쳐진 깨끗한 텍스트를 통째로 DB에 밀어 넣습니다.
        db_path = "./data/chroma_db_real"
        build_vector_db(combined_text, db_path=db_path)
        print(f"\n🎉 실제 MIMIC 데이터가 {db_path}에 성공적으로 적재되었습니다!")
    else:
        print("조건에 맞는 데이터를 찾지 못했습니다.")

if __name__ == "__main__":
    # 다운로드 받은 NOTEEVENTS.csv.gz 파일의 경로를 지정해주세요.
    # (data 폴더 안에 파일을 넣었다고 가정합니다)
    mimic_file_path = "./data/NOTEEVENTS.csv.gz"
    
    if os.path.exists(mimic_file_path):
        # 테스트를 위해 딱 2명의 환자 데이터만 뽑아서 DB에 넣어봅니다.
        process_mimic_notes(mimic_file_path, max_patients=2)
    else:
        print(f"❌ 파일을 찾을 수 없습니다. 경로를 확인해주세요: {mimic_file_path}")