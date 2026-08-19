"""
Script: ingest_to_backend.py
Description: CLI utility to ingest extracted OCR results (JSON, JSONL, Parquet) 
             directly into Backend services (Elasticsearch 'frames_ocr' index and SQLite FTS5).
"""

import os
import sys
import json
import argparse
import requests
from typing import Dict, Any, List
from tqdm import tqdm


def create_es_index_if_not_exists(es_url: str, index_name: str = "frames_ocr"):
    """Tạo Elasticsearch index với mapping chuẩn cho tìm kiếm tiếng Việt."""
    url = f"{es_url.rstrip('/')}/{index_name}"
    check_res = requests.head(url)
    if check_res.status_code == 200:
        print(f"✓ Elasticsearch index '{index_name}' đã tồn tại.")
        return

    mapping = {
        "settings": {
            "number_of_shards": 1,
            "number_of_replicas": 0,
            "analysis": {
                "analyzer": {
                    "vietnamese_standard": {
                        "tokenizer": "standard",
                        "filter": ["lowercase", "asciifolding"]
                    }
                }
            }
        },
        "mappings": {
            "properties": {
                "frame_id": {"type": "keyword"},
                "video_id": {"type": "keyword"},
                "frame_idx": {"type": "integer"},
                "timestamp": {"type": "float"},
                "timestamp_ms": {"type": "integer"},
                "timestamp_str": {"type": "keyword"},
                "youtube_url": {"type": "keyword"},
                "text": {
                    "type": "text",
                    "analyzer": "standard",
                    "fields": {
                        "folded": {
                            "type": "text",
                            "analyzer": "vietnamese_standard"
                        }
                    }
                },
                "text_unsigned": {"type": "text"},
                "confidence": {"type": "float"},
                "bbox_norm": {"type": "float"},
                "entities": {"type": "keyword"},
                "frame_span": {"type": "integer"},
                "occurrences": {"type": "integer"}
            }
        }
    }

    resp = requests.put(url, json=mapping)
    if resp.status_code in [200, 201]:
        print(f"✓ Đã tạo thành công Elasticsearch index '{index_name}'.")
    else:
        print(f"⚠️ Cảnh báo tạo index ({resp.status_code}): {resp.text}")


def ingest_jsonl_to_elasticsearch(jsonl_path: str, es_url: str, index_name: str = "frames_ocr", batch_size: int = 1000):
    """Nạp file frames_ocr_bulk.jsonl trực tiếp vào Elasticsearch bằng Bulk API."""
    if not os.path.exists(jsonl_path):
        print(f"❌ Lỗi: File '{jsonl_path}' không tồn tại!")
        return False

    create_es_index_if_not_exists(es_url, index_name)

    bulk_url = f"{es_url.rstrip('/')}/_bulk"
    headers = {"Content-Type": "application/x-ndjson"}

    total_lines = sum(1 for _ in open(jsonl_path, 'r', encoding='utf-8'))
    print(f"🚀 Bắt đầu nạp {total_lines:,} dòng dữ liệu vào Elasticsearch ({es_url})...")

    batch_payload = []
    indexed_count = 0

    with open(jsonl_path, 'r', encoding='utf-8') as f:
        with tqdm(total=total_lines, desc="Ingesting to Elasticsearch", unit="doc") as pbar:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                batch_payload.append(line)
                if len(batch_payload) >= batch_size * 2: # Mỗi doc gồm 2 dòng: action + document
                    data_str = "\n".join(batch_payload) + "\n"
                    resp = requests.post(bulk_url, data=data_str.encode('utf-8'), headers=headers)
                    if resp.status_code == 200:
                        res_json = resp.json()
                        if res_json.get("errors"):
                            print("⚠️ Một số bản ghi bị lỗi khi bulk insert.")
                    else:
                        print(f"❌ Lỗi HTTP Bulk Insert ({resp.status_code}): {resp.text}")
                    indexed_count += len(batch_payload) // 2
                    pbar.update(len(batch_payload))
                    batch_payload = []

            if batch_payload:
                data_str = "\n".join(batch_payload) + "\n"
                requests.post(bulk_url, data=data_str.encode('utf-8'), headers=headers)
                indexed_count += len(batch_payload) // 2
                pbar.update(len(batch_payload))

    print(f"✅ Hoàn tất nạp {indexed_count:,} bản ghi vào Elasticsearch index '{index_name}'!")
    return True


def main():
    parser = argparse.ArgumentParser(description="Ingest OCR Results to Backend Services (Elasticsearch/SQLite).")
    parser.add_argument("--input_file", default="data/ocr_results/frames_ocr_bulk.jsonl", help="Path to frames_ocr_bulk.jsonl or ocr.json")
    parser.add_argument("--es_url", default="http://localhost:9200", help="Elasticsearch HTTP Endpoint")
    parser.add_argument("--index_name", default="frames_ocr", help="Target Elasticsearch index name")
    parser.add_argument("--batch_size", type=int, default=1000, help="Batch size for bulk indexing")

    args = parser.parse_args()
    ingest_jsonl_to_elasticsearch(
        jsonl_path=args.input_file,
        es_url=args.es_url,
        index_name=args.index_name,
        batch_size=args.batch_size
    )


if __name__ == "__main__":
    main()
