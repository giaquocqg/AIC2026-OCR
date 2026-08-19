"""
Module: indexer.py
Description: High-speed indexing and retrieval engine using SQLite FTS5 (Full-Text Search)
             and Apache Parquet for downstream multimodal RAG and competition submission formatting.
"""

import os
import json
import sqlite3
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from .postprocessor import VietnamesePostProcessor


class OCRIndexer:
    """Quản lý xuất và lưu trữ dữ liệu OCR sang Parquet, SQLite FTS5, và định dạng JSON/Elasticsearch cho Backend."""

    def __init__(self, output_dir: str = "data/ocr_results"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def export_parquet(self, records: List[Dict[str, Any]], filename: str = "ocr_index.parquet") -> str:
        """
        Xuất danh sách detection đã tiền xử lý & gộp sang file Parquet chuẩn schema.
        Phục vụ team Multimodal Retrieval / Vector Search (T04) và kết hợp Milvus theo frame_id.
        """
        output_path = os.path.join(self.output_dir, filename)

        if not records:
            empty_df = pd.DataFrame(columns=[
                "frame_id", "video_id", "frame_idx", "timestamp", "timestamp_ms", "timestamp_str",
                "youtube_url", "frame_start", "frame_end", "occurrences", "text", "text_unsigned",
                "confidence", "bbox_ymin", "bbox_xmin", "bbox_ymax", "bbox_xmax", "polygon_json", "entities_json"
            ])
            table = pa.Table.from_pandas(empty_df)
            pq.write_table(table, output_path, compression='snappy')
            return output_path

        flat_data = []
        for r in records:
            bbox = r.get("bbox_norm", [0.0, 0.0, 0.0, 0.0])
            t_ms = int(r.get("timestamp_ms", 0))
            t_sec = float(r.get("timestamp", round(t_ms / 1000.0, 3)))
            flat_data.append({
                "frame_id": str(r.get("frame_id", f"{r.get('video_id', '')}_{r.get('frame_idx', 0)}")),
                "video_id": str(r.get("video_id", "")),
                "frame_idx": int(r.get("frame_idx", 0)),
                "timestamp": t_sec,
                "timestamp_ms": t_ms,
                "timestamp_str": str(r.get("timestamp_str", "00:00:00.000")),
                "youtube_url": str(r.get("youtube_url", "")),
                "frame_start": int(r.get("frame_start", r.get("frame_idx", 0))),
                "frame_end": int(r.get("frame_end", r.get("frame_idx", 0))),
                "occurrences": int(r.get("occurrences", 1)),
                "text": str(r.get("text", "")),
                "text_unsigned": str(r.get("text_unsigned", "")),
                "confidence": float(r.get("confidence", 0.0)),
                "bbox_ymin": float(bbox[0]),
                "bbox_xmin": float(bbox[1]),
                "bbox_ymax": float(bbox[2]),
                "bbox_xmax": float(bbox[3]),
                "polygon_json": json.dumps(r.get("polygon", []), ensure_ascii=False),
                "entities_json": json.dumps(r.get("entities", []), ensure_ascii=False)
            })

        df = pd.DataFrame(flat_data)
        table = pa.Table.from_pandas(df)
        pq.write_table(table, output_path, compression='snappy')
        return output_path

    def build_sqlite_fts5(self, records: List[Dict[str, Any]], db_filename: str = "ocr_fts.db") -> str:
        """
        Khởi tạo cơ sở dữ liệu SQLite và đánh chỉ mục Full-Text Search (FTS5).
        Cung cấp tốc độ tìm kiếm < 3ms cho giao diện thi đấu và LLM Agent Tool.
        """
        db_path = os.path.join(self.output_dir, db_filename)
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Tạo bảng dữ liệu chính
        cursor.execute("DROP TABLE IF EXISTS ocr_records")
        cursor.execute("DROP TABLE IF EXISTS ocr_fts")

        cursor.execute("""
            CREATE TABLE ocr_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                frame_id TEXT,
                video_id TEXT,
                frame_idx INTEGER,
                timestamp REAL,
                timestamp_ms INTEGER,
                timestamp_str TEXT,
                youtube_url TEXT,
                frame_start INTEGER,
                frame_end INTEGER,
                occurrences INTEGER,
                text TEXT,
                text_unsigned TEXT,
                confidence REAL,
                bbox_ymin REAL,
                bbox_xmin REAL,
                bbox_ymax REAL,
                bbox_xmax REAL,
                polygon TEXT,
                entities TEXT
            )
        """)

        # Tạo chỉ mục tìm kiếm thông thường
        cursor.execute("CREATE INDEX idx_frame_id ON ocr_records (frame_id)")
        cursor.execute("CREATE INDEX idx_video_frame ON ocr_records (video_id, frame_idx)")
        cursor.execute("CREATE INDEX idx_confidence ON ocr_records (confidence)")

        # Tạo bảng Full-Text Search FTS5
        cursor.execute("""
            CREATE VIRTUAL TABLE ocr_fts USING fts5(
                text,
                text_unsigned,
                frame_id UNINDEXED,
                video_id UNINDEXED,
                entities UNINDEXED,
                content='ocr_records',
                content_rowid='id'
            )
        """)

        # Insert dữ liệu vào ocr_records
        insert_sql = """
            INSERT INTO ocr_records (
                frame_id, video_id, frame_idx, timestamp, timestamp_ms, timestamp_str, youtube_url,
                frame_start, frame_end, occurrences, text, text_unsigned, confidence,
                bbox_ymin, bbox_xmin, bbox_ymax, bbox_xmax, polygon, entities
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """

        rows = []
        for r in records:
            bbox = r.get("bbox_norm", [0.0, 0.0, 0.0, 0.0])
            t_ms = int(r.get("timestamp_ms", 0))
            t_sec = float(r.get("timestamp", round(t_ms / 1000.0, 3)))
            rows.append((
                str(r.get("frame_id", f"{r.get('video_id', '')}_{r.get('frame_idx', 0)}")),
                str(r.get("video_id", "")),
                int(r.get("frame_idx", 0)),
                t_sec,
                t_ms,
                str(r.get("timestamp_str", "00:00:00.000")),
                str(r.get("youtube_url", "")),
                int(r.get("frame_start", r.get("frame_idx", 0))),
                int(r.get("frame_end", r.get("frame_idx", 0))),
                int(r.get("occurrences", 1)),
                str(r.get("text", "")),
                str(r.get("text_unsigned", "")),
                float(r.get("confidence", 0.0)),
                float(bbox[0]),
                float(bbox[1]),
                float(bbox[2]),
                float(bbox[3]),
                json.dumps(r.get("polygon", []), ensure_ascii=False),
                json.dumps(r.get("entities", []), ensure_ascii=False)
            ))

        cursor.executemany(insert_sql, rows)

        # Đẩy dữ liệu vào bảng FTS5
        cursor.execute("INSERT INTO ocr_fts(ocr_fts) VALUES('rebuild')")

        conn.commit()
        conn.close()
        return db_path

    def export_backend_ocr_json(self, records: List[Dict[str, Any]], filename: str = "ocr.json") -> str:
        """
        Xuất file ocr.json cấu trúc theo frame_id tương thích trực tiếp với Backend Video Retrieval:
        {
          "0": {
            "ocr_text": "CƠM TẤM BA GHIỀN 45K",
            "video_id": "L21_V001",
            "frame_idx": 0,
            "timestamp": 0.0,
            "youtube_url": "https://...",
            "entities": [...]
          }
        }
        """
        output_path = os.path.join(self.output_dir, filename)
        frame_dict: Dict[str, Dict[str, Any]] = {}

        for r in records:
            fid = str(r.get("frame_id", f"{r.get('video_id', '')}_{r.get('frame_idx', 0)}"))
            t_ms = int(r.get("timestamp_ms", 0))
            t_sec = float(r.get("timestamp", round(t_ms / 1000.0, 3)))
            text = str(r.get("text", "")).strip()

            if fid not in frame_dict:
                frame_dict[fid] = {
                    "ocr_text": text,
                    "video_id": str(r.get("video_id", "")),
                    "frame_idx": int(r.get("frame_idx", 0)),
                    "timestamp": t_sec,
                    "youtube_url": str(r.get("youtube_url", "")),
                    "entities": list(r.get("entities", []))
                }
            else:
                # Ghép text nếu có nhiều phát hiện trong cùng một frame
                if text:
                    existing_text = frame_dict[fid]["ocr_text"]
                    if existing_text:
                        frame_dict[fid]["ocr_text"] = f"{existing_text} | {text}"
                    else:
                        frame_dict[fid]["ocr_text"] = text
                
                # Gộp entities
                for ent in r.get("entities", []):
                    if ent not in frame_dict[fid]["entities"]:
                        frame_dict[fid]["entities"].append(ent)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(frame_dict, f, ensure_ascii=False, indent=2)

        return output_path

    def export_elasticsearch_bulk(self, records: List[Dict[str, Any]], filename: str = "frames_ocr_bulk.jsonl", index_name: str = "frames_ocr") -> str:
        """
        Xuất file JSONL theo chuẩn Elasticsearch Bulk API cho index `frames_ocr`.
        """
        output_path = os.path.join(self.output_dir, filename)
        # Đầu tiên nhóm theo frame_id
        grouped: Dict[str, Dict[str, Any]] = {}
        for r in records:
            fid = str(r.get("frame_id", f"{r.get('video_id', '')}_{r.get('frame_idx', 0)}"))
            t_ms = int(r.get("timestamp_ms", 0))
            t_sec = float(r.get("timestamp", round(t_ms / 1000.0, 3)))
            text = str(r.get("text", "")).strip()

            if fid not in grouped:
                grouped[fid] = {
                    "frame_id": fid,
                    "ocr_text": text,
                    "video_id": str(r.get("video_id", "")),
                    "frame_idx": int(r.get("frame_idx", 0)),
                    "timestamp": t_sec
                }
            else:
                if text:
                    existing = grouped[fid]["ocr_text"]
                    grouped[fid]["ocr_text"] = f"{existing} {text}" if existing else text

        with open(output_path, "w", encoding="utf-8") as f:
            for fid, doc in grouped.items():
                action = {"index": {"_index": index_name, "_id": fid}}
                f.write(json.dumps(action, ensure_ascii=False) + "\n")
                f.write(json.dumps({
                    "ocr_text": doc["ocr_text"],
                    "video_id": doc["video_id"],
                    "frame_idx": doc["frame_idx"],
                    "timestamp": doc["timestamp"]
                }, ensure_ascii=False) + "\n")

        return output_path


# Common query stop words in Vietnamese Q&A and KIS prompts
QUESTION_STOPWORDS = {
    "tim", "video", "canh", "hinh", "anh", "co", "bao", "nhieu", "o", "dau", "mau",
    "gi", "la", "ai", "nao", "trong", "khoang", "thoi", "gian", "dia", "diem", "xem",
    "cho", "biet", "ten", "the", "nao", "do", "va", "voi", "mot", "nhung", "cac"
}


class OCRIndexSearcher:
    """Công cụ tìm kiếm văn bản tốc độ cao trên SQLite FTS5."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        if not os.path.exists(db_path):
            raise FileNotFoundError(f"Database FTS không tồn tại: {db_path}")

    def get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def search(self, 
               query: str, 
               top_k: int = 100, 
               video_filter: Optional[str] = None, 
               entity_type: Optional[str] = None,
               min_confidence: float = 0.30) -> List[Dict[str, Any]]:
        """
        Tìm kiếm truy vấn text trên cơ sở dữ liệu FTS5.
        Hỗ trợ tìm kiếm cả tiếng Việt có dấu, không dấu, và câu hỏi tự nhiên (Q&A).
        """
        if not query or not query.strip():
            return []

        query_clean = VietnamesePostProcessor.clean_text(query)
        query_unsigned = VietnamesePostProcessor.to_unsigned(query_clean)

        raw_tokens = query_unsigned.split()
        if not raw_tokens:
            return []

        # Lọc bớt từ dừng / câu hỏi phụ để lấy các từ khóa thực thể quan trọng
        content_tokens = [t for t in raw_tokens if t not in QUESTION_STOPWORDS]
        if not content_tokens:
            content_tokens = raw_tokens

        # Xây dựng biểu thức tìm kiếm FTS5 thông minh
        exact_phrase = f'"{query_unsigned.replace(chr(34), "")}"'
        or_terms = " OR ".join([f'"{t.replace(chr(34), "")}"*' for t in content_tokens])
        and_terms = " AND ".join([f'"{t.replace(chr(34), "")}"*' for t in content_tokens])

        fts_query_str = f"({exact_phrase}) OR ({and_terms}) OR ({or_terms})"

        conn = self.get_connection()
        cursor = conn.cursor()

        sql = f"""
            SELECT 
                r.id,
                r.frame_id,
                r.video_id,
                r.frame_idx,
                r.timestamp,
                r.timestamp_ms,
                r.timestamp_str,
                r.youtube_url,
                r.frame_start,
                r.frame_end,
                r.occurrences,
                r.text,
                r.text_unsigned,
                r.confidence,
                r.bbox_ymin,
                r.bbox_xmin,
                r.bbox_ymax,
                r.bbox_xmax,
                r.polygon,
                r.entities,
                bm25(ocr_fts) AS fts_score
            FROM ocr_fts f
            JOIN ocr_records r ON f.rowid = r.id
            WHERE ocr_fts MATCH ?
              AND r.confidence >= ?
        """

        params: List[Any] = [f"text_unsigned: ({fts_query_str}) OR text: ({fts_query_str})", min_confidence]

        if video_filter:
            sql += " AND r.video_id = ?"
            params.append(video_filter)

        sql += " ORDER BY fts_score ASC, r.confidence DESC LIMIT ?"
        params.append(top_k * 2)

        try:
            cursor.execute(sql, params)
            rows = cursor.fetchall()
        except sqlite3.OperationalError:
            # Fallback nếu FTS syntax error: dùng LIKE search
            fallback_sql = """
                SELECT 
                    id, frame_id, video_id, frame_idx, timestamp, timestamp_ms, timestamp_str,
                    youtube_url, frame_start, frame_end, occurrences, text, text_unsigned,
                    confidence, bbox_ymin, bbox_xmin, bbox_ymax, bbox_xmax,
                    polygon, entities, 1.0 AS fts_score
                FROM ocr_records
                WHERE (text_unsigned LIKE ? OR text LIKE ?)
                  AND confidence >= ?
            """
            like_param = f"%{query_unsigned}%"
            fallback_params = [like_param, f"%{query_clean}%", min_confidence]
            if video_filter:
                fallback_sql += " AND video_id = ?"
                fallback_params.append(video_filter)
            fallback_sql += " ORDER BY confidence DESC LIMIT ?"
            fallback_params.append(top_k * 2)
            cursor.execute(fallback_sql, fallback_params)
            rows = cursor.fetchall()

        results = []
        seen_candidates = set()

        for row in rows:
            entities = json.loads(row["entities"]) if row["entities"] else []
            
            # Lọc theo entity nếu có yêu cầu
            if entity_type:
                matching_ents = [e for e in entities if e.get("type", "").upper() == entity_type.upper()]
                if not matching_ents:
                    continue

            polygon = json.loads(row["polygon"]) if row["polygon"] else []
            vid = row["video_id"]
            f_idx = row["frame_idx"]
            fid = row["frame_id"] if row["frame_id"] else f"{vid}_{f_idx}"
            
            # Deduplicate kết quả top để đa dạng hóa
            candidate_key = (vid, f_idx)
            if candidate_key in seen_candidates:
                continue
            seen_candidates.add(candidate_key)

            # Tính điểm xếp hạng kết hợp (BM25 ranking + exact match boost + confidence)
            bm25_val = float(row["fts_score"])
            relevance_score = (1.0 / (1.0 + abs(bm25_val))) * 0.7 + float(row["confidence"]) * 0.3
            if query_unsigned in row["text_unsigned"]:
                relevance_score += 0.5  # Boost khi khớp trọn vẹn cụm từ

            # Tạo chuỗi nộp bài chuẩn quy chế AIC 2026
            kis_submission = f"{vid}, {f_idx}"
            
            # Gợi ý answer tốt nhất cho Visual Q&A
            suggested_answer = row["text"]
            if entities:
                suggested_answer = entities[0].get("value", row["text"])
            qa_submission = f"{vid}, {f_idx}, {suggested_answer}"

            results.append({
                "frame_id": fid,
                "video_id": vid,
                "frame_idx": f_idx,
                "timestamp": float(row["timestamp"]) if row["timestamp"] is not None else 0.0,
                "timestamp_ms": row["timestamp_ms"],
                "timestamp_str": row["timestamp_str"],
                "youtube_url": row["youtube_url"] if row["youtube_url"] else "",
                "frame_start": row["frame_start"],
                "frame_end": row["frame_end"],
                "occurrences": row["occurrences"],
                "text": row["text"],
                "text_unsigned": row["text_unsigned"],
                "confidence": round(float(row["confidence"]), 4),
                "bbox_norm": [
                    round(float(row["bbox_ymin"]), 4),
                    round(float(row["bbox_xmin"]), 4),
                    round(float(row["bbox_ymax"]), 4),
                    round(float(row["bbox_xmax"]), 4)
                ],
                "polygon": polygon,
                "entities": entities,
                "score": round(relevance_score, 4),
                "submission_kis": kis_submission,
                "submission_qa": qa_submission,
                "submission_trake": f"{vid}, {f_idx}"
            })

            if len(results) >= top_k:
                break

        conn.close()
        # Sắp xếp kết quả theo điểm tổng hợp cao nhất
        results.sort(key=lambda x: x["score"], reverse=True)
        return results

    def get_frame_detections(self, video_id: str, frame_idx: int) -> List[Dict[str, Any]]:
        """Lấy tất cả các đoạn text phát hiện được trong 1 frame cụ thể."""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT * FROM ocr_records
            WHERE video_id = ? AND frame_idx = ?
            ORDER BY bbox_ymin ASC, bbox_xmin ASC
        """, (video_id, frame_idx))
        rows = cursor.fetchall()
        conn.close()

        detections = []
        for r in rows:
            detections.append({
                "frame_id": r["frame_id"],
                "text": r["text"],
                "text_unsigned": r["text_unsigned"],
                "confidence": r["confidence"],
                "timestamp": r["timestamp"],
                "timestamp_str": r["timestamp_str"],
                "youtube_url": r["youtube_url"],
                "bbox_norm": [r["bbox_ymin"], r["bbox_xmin"], r["bbox_ymax"], r["bbox_xmax"]],
                "polygon": json.loads(r["polygon"]) if r["polygon"] else [],
                "entities": json.loads(r["entities"]) if r["entities"] else []
            })
        return detections

