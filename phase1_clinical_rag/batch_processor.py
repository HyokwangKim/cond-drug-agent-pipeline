from __future__ import annotations

import logging
import time
from pathlib import Path

import pandas as pd
try:
    from langchain_chroma import Chroma
except ImportError:  # pragma: no cover - fallback for older envs
    from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from tqdm import tqdm

from phase1_clinical_rag.data_ingestion import preprocess_clinical_text

logger = logging.getLogger(__name__)


def load_processed_ids(log_file: Path) -> set[str]:
    if log_file.is_file():
        with log_file.open("r", encoding="utf-8") as f:
            return {line.strip() for line in f if line.strip()}
    return set()


def save_processed_id(log_file: Path, row_id: str) -> None:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with log_file.open("a", encoding="utf-8") as f:
        f.write(f"{row_id}\n")


def run_batch_ingestion(
    input_file: Path,
    db_path: Path,
    log_file: Path,
    chunk_size: int = 1000,
    limit_samples: int | None = 50,
) -> None:
    """
    NOTEEVENTS.csv.gz 에서 Discharge summary 만 골라 전처리 후 Chroma에 적재합니다.

    Args:
        input_file: MIMIC NOTEEVENTS gzip CSV 경로.
        db_path: Chroma persist 디렉터리.
        log_file: 처리된 ROW_ID 누적 로그.
        chunk_size: pandas chunksize.
        limit_samples: None 이면 상한 없음(전체 스캔).
    """
    if not input_file.is_file():
        raise FileNotFoundError(f"입력 파일이 없습니다: {input_file}")

    processed_ids = load_processed_ids(log_file)
    logger.info("배치 적재 시작 기처리=%s건 input=%s", len(processed_ids), input_file)

    reader = pd.read_csv(
        input_file,
        compression="gzip",
        chunksize=chunk_size,
        usecols=["ROW_ID", "CATEGORY", "TEXT"],
    )

    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-2-preview")
    vectorstore: Chroma | None = None
    total_processed = 0

    db_path.mkdir(parents=True, exist_ok=True)

    for chunk in reader:
        discharge_summaries = chunk[chunk["CATEGORY"] == "Discharge summary"]

        for _, row in discharge_summaries.iterrows():
            row_id = str(row["ROW_ID"])
            if row_id in processed_ids:
                continue

            raw_text = row["TEXT"]
            try:
                print(f"\n{'=' * 30} [ID: {row_id} 분석 중] {'=' * 30}")
                print(f"🔍 [Original]: {str(raw_text)[:150].strip()}...")

                cleaned_text = preprocess_clinical_text(str(raw_text))
                if vectorstore is None:
                    vectorstore = Chroma.from_texts(
                        texts=[cleaned_text],
                        embedding=embeddings,
                        metadatas=[{"row_id": row_id, "category": "Discharge summary"}],
                        persist_directory=str(db_path),
                    )
                else:
                    vectorstore.add_texts(
                        texts=[cleaned_text],
                        metadatas=[{"row_id": row_id, "category": "Discharge summary"}],
                    )
                vectorstore.persist()
                save_processed_id(log_file, row_id)
                print(f"✨ [Cleaned]: {cleaned_text[:150].strip()}...")
                total_processed += 1
                processed_ids.add(row_id)
                time.sleep(0.1)
                if limit_samples is not None and total_processed >= limit_samples:
                    break
            except (OSError, ValueError, RuntimeError) as e:
                logger.warning("행 처리 실패 row_id=%s err=%s", row_id, e)
                print(f"🚨 오류 발생 (ID {row_id}): {e}")
                time.sleep(5)
                continue

        if limit_samples is not None and total_processed >= limit_samples:
            break

    print(f"🎉 성공! 이번 세션에서 {total_processed}개를 새로 처리했습니다.")


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    data = root / "data"
    run_batch_ingestion(
        input_file=data / "NOTEEVENTS.csv.gz",
        db_path=data / "chroma_db_real",
        log_file=data / "processed_ids.txt",
        chunk_size=1000,
        limit_samples=100,
    )
