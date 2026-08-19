"""
Unit Tests: OCR Indexer & Searcher (Parquet + SQLite FTS5)
"""

import os
import shutil
import unittest
import pandas as pd
from src.ocr.indexer import OCRIndexer, OCRIndexSearcher


class TestOCRIndexer(unittest.TestCase):

    def setUp(self):
        self.test_dir = "tests/temp_ocr_results"
        os.makedirs(self.test_dir, exist_ok=True)
        self.indexer = OCRIndexer(output_dir=self.test_dir)
        self.sample_records = [
            {
                "video_id": "L01_V001",
                "frame_idx": 150,
                "timestamp_ms": 6000,
                "timestamp_str": "00:06.000",
                "frame_start": 140,
                "frame_end": 160,
                "occurrences": 3,
                "text": "Cơm Tấm Ba Ghiền 45k",
                "text_unsigned": "com tam ba ghien 45k",
                "confidence": 0.95,
                "bbox_norm": [0.1, 0.2, 0.3, 0.8],
                "polygon": [[100, 100], [200, 100], [200, 200], [100, 200]],
                "entities": [{"type": "PRICE", "value": "45k", "span": [17, 20]}]
            },
            {
                "video_id": "L02_V005",
                "frame_idx": 880,
                "timestamp_ms": 35200,
                "timestamp_str": "00:35.200",
                "frame_start": 880,
                "frame_end": 890,
                "occurrences": 2,
                "text": "Xe Honda biển số 59-X3 123.45",
                "text_unsigned": "xe honda bien so 59-x3 123.45",
                "confidence": 0.92,
                "bbox_norm": [0.4, 0.5, 0.6, 0.9],
                "polygon": [],
                "entities": [{"type": "LICENSE_PLATE", "value": "59-X3 123.45", "span": [17, 29]}]
            },
            {
                "video_id": "L03_V010",
                "frame_idx": 2450,
                "timestamp_ms": 98000,
                "timestamp_str": "01:38.000",
                "frame_start": 2450,
                "frame_end": 2450,
                "occurrences": 1,
                "text": "Cửa hàng tiện lợi Circle K",
                "text_unsigned": "cua hang tien loi circle k",
                "confidence": 0.89,
                "bbox_norm": [0.2, 0.1, 0.4, 0.5],
                "polygon": [],
                "entities": []
            }
        ]

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_parquet_export(self):
        parquet_path = self.indexer.export_parquet(self.sample_records, "test_index.parquet")
        self.assertTrue(os.path.exists(parquet_path))
        df = pd.read_parquet(parquet_path)
        self.assertEqual(len(df), 3)
        self.assertIn("text_unsigned", df.columns)
        self.assertIn("bbox_ymin", df.columns)

    def test_sqlite_fts5_search(self):
        db_path = self.indexer.build_sqlite_fts5(self.sample_records, "test_fts.db")
        self.assertTrue(os.path.exists(db_path))

        searcher = OCRIndexSearcher(db_path)

        # 1. Tìm kiếm không dấu
        res1 = searcher.search("ba ghien")
        self.assertGreaterEqual(len(res1), 1)
        self.assertEqual(res1[0]["video_id"], "L01_V001")
        self.assertEqual(res1[0]["submission_kis"], "L01_V001, 150")

        # 2. Tìm kiếm có dấu
        res2 = searcher.search("biển số")
        self.assertGreaterEqual(len(res2), 1)
        self.assertEqual(res2[0]["video_id"], "L02_V005")

        # 3. Lọc theo thực thể
        res3 = searcher.search("45k", entity_type="PRICE")
        self.assertEqual(len(res3), 1)
        self.assertEqual(res3[0]["entities"][0]["value"], "45k")

        # 4. Lấy toàn bộ detections trong frame
        frame_dets = searcher.get_frame_detections("L01_V001", 150)
        self.assertEqual(len(frame_dets), 1)
        self.assertEqual(frame_dets[0]["text"], "Cơm Tấm Ba Ghiền 45k")


if __name__ == "__main__":
    unittest.main()
