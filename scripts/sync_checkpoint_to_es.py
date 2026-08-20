"""
Script: sync_checkpoint_to_es.py
Description: Reads current OCR checkpoint, exports frames_ocr_bulk.jsonl, and ingests to Elasticsearch.
"""

import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.ocr.indexer import OCRIndexer
from scripts.ingest_to_backend import ingest_jsonl_to_elasticsearch

def main():
    checkpoint_file = "data/ocr_results/.ocr_records_checkpoint.jsonl"
    if not os.path.exists(checkpoint_file):
        print(f"❌ Checkpoint file not found: {checkpoint_file}")
        return

    records = []
    print(f"📖 Reading checkpoint file: {checkpoint_file}...")
    with open(checkpoint_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    print(f"✅ Loaded {len(records):,} records.")
    indexer = OCRIndexer(output_dir="data/ocr_results")
    bulk_path = indexer.export_elasticsearch_bulk(records, filename="frames_ocr_bulk.jsonl", index_name="frames_ocr")
    print(f"✅ Exported Elasticsearch bulk format to: {bulk_path}")

    # Ingest to Elasticsearch
    es_url = "http://localhost:9200"
    print(f"🚀 Ingesting to Elasticsearch ({es_url})...")
    ingest_jsonl_to_elasticsearch(
        jsonl_path=bulk_path,
        es_url=es_url,
        index_name="frames_ocr",
        batch_size=1000
    )
    print("🎉 All done!")

if __name__ == "__main__":
    main()
