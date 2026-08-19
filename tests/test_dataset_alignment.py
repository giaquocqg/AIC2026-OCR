"""
Tests for AIC 2026 Dataset Alignment & Metadata Mapping:
- fps.json
- index.json (190k keyframes mapping)
- Youtube_URL.json
- WebP keyframe naming parsing (keyframe_XX.webp)
- Backend ocr.json export & Elasticsearch bulk JSONL export
"""

import os
import json
import tempfile
import unittest
from src.ocr.preprocessor import FrameMetadataMapper
from src.ocr.indexer import OCRIndexer, OCRIndexSearcher
from src.ocr.pipeline import OCRPipeline


class TestDatasetAlignment(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        
        # Tạo mock fps.json
        self.fps_file = os.path.join(self.temp_dir, "fps.json")
        with open(self.fps_file, "w", encoding="utf-8") as f:
            json.dump({"L21_V001": 30.0, "L21_V002": 25.0}, f)

        # Tạo mock index.json
        self.index_file = os.path.join(self.temp_dir, "index.json")
        with open(self.index_file, "w", encoding="utf-8") as f:
            json.dump({
                "0": "L21_V001/keyframe_0.webp",
                "1": "L21_V001/keyframe_75.webp",
                "2": "L21_V002/keyframe_100.webp"
            }, f)

        # Tạo mock Youtube_URL.json
        self.yt_file = os.path.join(self.temp_dir, "Youtube_URL.json")
        with open(self.yt_file, "w", encoding="utf-8") as f:
            json.dump([
                {"video_id": "L21_V001", "watch_url": "https://youtube.com/watch?v=TEST001"},
                {"video_id": "L21_V002", "watch_url": "https://youtube.com/watch?v=TEST002"}
            ], f)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_frame_metadata_mapper_webp_and_index(self):
        mapper = FrameMetadataMapper(
            fps_file=self.fps_file,
            index_file=self.index_file,
            youtube_url_file=self.yt_file
        )

        # Test Frame 1: L21_V001 / keyframe_75.webp
        info = mapper.get_frame_info("L21_V001", "keyframe_75.webp")
        self.assertEqual(info["frame_id"], "1")
        self.assertEqual(info["video_id"], "L21_V001")
        self.assertEqual(info["frame_idx"], 75)
        self.assertEqual(info["timestamp"], 2.5)  # 75 frames / 30 fps = 2.5s
        self.assertEqual(info["timestamp_ms"], 2500)
        self.assertEqual(info["timestamp_str"], "00:02.500")
        self.assertEqual(info["youtube_url"], "https://youtube.com/watch?v=TEST001&t=2s")

        # Test Frame 0: L21_V001 / keyframe_0.webp
        info0 = mapper.get_frame_info("L21_V001", "keyframe_0.webp")
        self.assertEqual(info0["frame_id"], "0")
        self.assertEqual(info0["frame_idx"], 0)
        self.assertEqual(info0["timestamp"], 0.0)

    def test_indexer_exports_and_search(self):
        indexer = OCRIndexer(output_dir=self.temp_dir)
        
        sample_records = [
            {
                "frame_id": "1",
                "video_id": "L21_V001",
                "frame_idx": 75,
                "timestamp": 2.5,
                "timestamp_ms": 2500,
                "timestamp_str": "00:02.500",
                "youtube_url": "https://youtube.com/watch?v=TEST001&t=2s",
                "frame_start": 75,
                "frame_end": 75,
                "occurrences": 1,
                "text": "CƠM TẤM SƯỜN NƯỚNG 45K",
                "text_unsigned": "com tam suon nuong 45k",
                "confidence": 0.95,
                "bbox_norm": [0.1, 0.2, 0.3, 0.8],
                "polygon": [[10, 10], [50, 10], [50, 50], [10, 50]],
                "entities": [{"type": "PRICE", "value": "45K", "bbox": [0.1, 0.2, 0.3, 0.8]}]
            }
        ]

        # 1. Parquet
        parquet_path = indexer.export_parquet(sample_records, "test.parquet")
        self.assertTrue(os.path.exists(parquet_path))

        # 2. SQLite FTS5
        db_path = indexer.build_sqlite_fts5(sample_records, "test.db")
        self.assertTrue(os.path.exists(db_path))

        # 3. Backend ocr.json
        json_path = indexer.export_backend_ocr_json(sample_records, "ocr.json")
        self.assertTrue(os.path.exists(json_path))
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            self.assertIn("1", data)
            self.assertEqual(data["1"]["video_id"], "L21_V001")
            self.assertEqual(data["1"]["timestamp"], 2.5)

        # 4. Elasticsearch bulk JSONL
        es_bulk_path = indexer.export_elasticsearch_bulk(sample_records, "es_bulk.jsonl")
        self.assertTrue(os.path.exists(es_bulk_path))

        # 5. Searcher test
        searcher = OCRIndexSearcher(db_path)
        hits = searcher.search("com tam 45k")
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0]["frame_id"], "1")
        self.assertEqual(hits[0]["submission_kis"], "L21_V001, 75")
        self.assertEqual(hits[0]["submission_qa"], "L21_V001, 75, 45K")
        self.assertEqual(hits[0]["youtube_url"], "https://youtube.com/watch?v=TEST001&t=2s")


if __name__ == "__main__":
    unittest.main()
