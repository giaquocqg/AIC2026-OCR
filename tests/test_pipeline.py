"""
Unit Tests: End-to-End OCR Pipeline Execution
"""

import os
import shutil
import unittest
import numpy as np
import cv2
from src.ocr.pipeline import OCRPipeline


class TestOCRPipeline(unittest.TestCase):

    def setUp(self):
        self.test_root = "tests/temp_test_dataset"
        self.video_dir = os.path.join(self.test_root, "keyframes", "L01_V001")
        self.out_dir = os.path.join(self.test_root, "output")
        os.makedirs(self.video_dir, exist_ok=True)
        os.makedirs(self.out_dir, exist_ok=True)

        # Tạo 3 ảnh keyframe giả lập có vẽ chữ
        for i, name in enumerate(["0001.jpg", "0002.jpg", "0003.jpg"], 1):
            img = np.zeros((480, 640, 3), dtype=np.uint8)
            img[:] = (240, 240, 240)
            cv2.putText(img, "COM TAM 45K", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3)
            cv2.imwrite(os.path.join(self.video_dir, name), img)

        self.config = {
            "pipeline": {"name": "TestPipeline", "device": "cpu"},
            "preprocessor": {
                "max_image_dim": 800,
                "min_image_dim": 320,
                "blur_laplacian_threshold": 20.0,
                "enable_adaptive_clahe": False
            },
            "detector": {
                "engine": "paddleocr",
                "use_angle_cls": False,
                "min_box_area": 50.0
            },
            "recognizer": {
                "engine": "hybrid",
                "batch_size": 16,
                "min_confidence": 0.20
            },
            "deduplicator": {
                "enable_temporal_merge": True,
                "similarity_threshold": 0.80,
                "max_frame_gap": 5
            },
            "storage": {
                "output_dir": self.out_dir,
                "parquet_filename": "test_ocr.parquet",
                "sqlite_db_filename": "test_ocr.db"
            }
        }

    def tearDown(self):
        if os.path.exists(self.test_root):
            shutil.rmtree(self.test_root)

    def test_pipeline_single_frame(self):
        pipeline = OCRPipeline(config_input=self.config)
        img_path = os.path.join(self.video_dir, "0001.jpg")
        res = pipeline.process_single_frame(img_path, video_id="L01_V001", image_name="0001.jpg")
        self.assertEqual(res["video_id"], "L01_V001")
        self.assertEqual(res["frame_idx"], 1)
        self.assertIn("detections", res)

    def test_pipeline_dataset_processing(self):
        pipeline = OCRPipeline(config_input=self.config)
        summary = pipeline.process_dataset(
            keyframes_root=os.path.join(self.test_root, "keyframes"),
            output_dir=self.out_dir,
            checkpoint=False
        )
        self.assertEqual(summary["total_videos"], 1)
        self.assertEqual(summary["total_frames"], 3)
        self.assertTrue(os.path.exists(summary["parquet_path"]))
        self.assertTrue(os.path.exists(summary["sqlite_db_path"]))


if __name__ == "__main__":
    unittest.main()
