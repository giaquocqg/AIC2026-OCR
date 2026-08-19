"""
Unit Tests: Temporal Deduplicator
"""

import unittest
from src.ocr.deduplicator import TemporalDeduplicator


class TestTemporalDeduplicator(unittest.TestCase):

    def setUp(self):
        self.dedup = TemporalDeduplicator(
            enable_temporal_merge=True,
            similarity_threshold=0.80,
            max_frame_gap=10
        )

    def test_text_similarity(self):
        sim = self.dedup.compute_text_similarity("CƠM TẤM BA GHIỀN", "COM TAM BA GHIEN")
        self.assertGreater(sim, 0.70)
        sim_exact = self.dedup.compute_text_similarity("Cơm Tấm", "Cơm Tấm")
        self.assertEqual(sim_exact, 1.0)

    def test_deduplicate_adjacent_frames(self):
        # 5 keyframes liên tiếp chứa cùng 1 biển hiệu quán cơm
        frame_records = [
            {
                "video_id": "L01_V001",
                "frame_idx": 100,
                "timestamp_ms": 4000,
                "timestamp_str": "00:04.000",
                "detections": [{
                    "text": "CƠM TẤM BA GHIỀN",
                    "text_unsigned": "com tam ba ghien",
                    "confidence": 0.88,
                    "bbox_norm": [0.1, 0.2, 0.3, 0.8],
                    "polygon": [],
                    "entities": []
                }]
            },
            {
                "video_id": "L01_V001",
                "frame_idx": 105,
                "timestamp_ms": 4200,
                "timestamp_str": "00:04.200",
                "detections": [{
                    "text": "CƠM TẤM BA GHIỀN",
                    "text_unsigned": "com tam ba ghien",
                    "confidence": 0.96, # Best confidence
                    "bbox_norm": [0.11, 0.21, 0.31, 0.81],
                    "polygon": [],
                    "entities": []
                }]
            },
            {
                "video_id": "L01_V001",
                "frame_idx": 110,
                "timestamp_ms": 4400,
                "timestamp_str": "00:04.400",
                "detections": [{
                    "text": "COM TAM BA GHIEN",
                    "text_unsigned": "com tam ba ghien",
                    "confidence": 0.85,
                    "bbox_norm": [0.12, 0.22, 0.32, 0.82],
                    "polygon": [],
                    "entities": []
                }]
            }
        ]

        merged = self.dedup.deduplicate_video_detections(frame_records)
        self.assertEqual(len(merged), 1, "Should merge 3 consecutive detections into 1 temporal record")
        rec = merged[0]
        self.assertEqual(rec["video_id"], "L01_V001")
        self.assertEqual(rec["frame_start"], 100)
        self.assertEqual(rec["frame_end"], 110)
        self.assertEqual(rec["frame_idx"], 105, "Representative frame should be frame 105 with highest confidence")
        self.assertEqual(rec["confidence"], 0.96)
        self.assertEqual(rec["occurrences"], 3)


if __name__ == "__main__":
    unittest.main()
