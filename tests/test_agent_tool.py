"""
Unit Tests: Agentic AI Spatial OCR Tool Card
"""

import os
import shutil
import unittest
from src.ocr.indexer import OCRIndexer
from src.ocr.agent_tool import SpatialOCRTool


class TestSpatialOCRTool(unittest.TestCase):

    def setUp(self):
        self.test_dir = "tests/temp_agent_tool_db"
        os.makedirs(self.test_dir, exist_ok=True)
        self.indexer = OCRIndexer(output_dir=self.test_dir)
        self.records = [
            {
                "video_id": "L01_V001",
                "frame_idx": 500,
                "timestamp_ms": 20000,
                "timestamp_str": "00:20.000",
                "frame_start": 500,
                "frame_end": 500,
                "occurrences": 1,
                "text": "QUÁN PHỞ THÌN 60k",
                "text_unsigned": "quan pho thin 60k",
                "confidence": 0.94,
                "bbox_norm": [0.1, 0.1, 0.3, 0.6],
                "polygon": [],
                "entities": [{"type": "PRICE", "value": "60k"}]
            }
        ]
        self.db_path = self.indexer.build_sqlite_fts5(self.records, "agent_fts.db")
        self.tool = SpatialOCRTool(self.db_path)

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_tool_search_ocr(self):
        hits = self.tool.tool_search_ocr("pho thin", top_k=5)
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0]["video_id"], "L01_V001")
        self.assertEqual(hits[0]["frame_idx"], 500)

    def test_tool_read_frame_text(self):
        info = self.tool.tool_read_frame_text("L01_V001", 500)
        self.assertEqual(info["total_detections"], 1)
        self.assertIn("PHỞ THÌN", info["full_transcription"])

    def test_tool_inspect_roi(self):
        roi_match = self.tool.tool_inspect_roi("L01_V001", 500, [0.0, 0.0, 0.5, 0.8])
        self.assertIn("PHỞ THÌN", roi_match["matched_text"])

        roi_miss = self.tool.tool_inspect_roi("L01_V001", 500, [0.7, 0.7, 0.9, 0.9])
        self.assertEqual(roi_miss["matched_text"], "")

    def test_tool_schemas(self):
        schemas = SpatialOCRTool.get_tool_schemas()
        self.assertEqual(len(schemas), 3)
        self.assertEqual(schemas[0]["name"], "search_ocr_text")


if __name__ == "__main__":
    unittest.main()
