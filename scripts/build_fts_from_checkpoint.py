"""
Script: build_fts_from_checkpoint.py
Description: Build SQLite FTS5 database from current checkpoint JSONL file.
"""

import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.ocr.indexer import OCRIndexer

def main():
    checkpoint_file = "data/ocr_results/.ocr_records_checkpoint.jsonl"
    if not os.path.exists(checkpoint_file):
        print(f"Checkpoint file not found: {checkpoint_file}")
        return
    
    records = []
    with open(checkpoint_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    
    print(f"Loaded {len(records):,} records from checkpoint. Building FTS5 database...")
    indexer = OCRIndexer(output_dir="data/ocr_results")
    db_path = indexer.build_sqlite_fts5(records, db_filename="ocr_fts.db")
    print(f"✅ Successfully built FTS5 DB at: {db_path} ({os.path.getsize(db_path)/(1024*1024):.2f} MB)")

if __name__ == "__main__":
    main()
