"""
Unit Tests for SubmissionFormatter & SubmissionPackager (AIC 2026 Submission Format Rules).
"""

import os
import zipfile
import tempfile
import unittest
from src.ocr.submission_packager import SubmissionFormatter, SubmissionPackager


class TestSubmissionPackager(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.packager = SubmissionPackager(output_dir=self.temp_dir)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_submission_formatter_clean_rules(self):
        # 1. Strip .mp4 extension
        self.assertEqual(SubmissionFormatter.clean_video_id("L01_V001.mp4"), "L01_V001")
        self.assertEqual(SubmissionFormatter.clean_video_id("L02_V025"), "L02_V025")

        # 2. Integer frame_idx without spaces
        self.assertEqual(SubmissionFormatter.clean_frame_idx("25 300"), 25300)
        self.assertEqual(SubmissionFormatter.clean_frame_idx(1234), 1234)

        # 3. Q&A Answer max 100 chars
        long_ans = "A" * 150
        self.assertEqual(len(SubmissionFormatter.clean_qa_answer(long_ans)), 100)

        # 4. Formats
        kis_row = SubmissionFormatter.format_kis_row("L01_V001.mp4", "1234")
        self.assertEqual(kis_row, ["L01_V001", 1234])

        qa_row = SubmissionFormatter.format_qa_row("L01_V001.mp4", "3450", 'Anh ấy nói "Xin chào"')
        self.assertEqual(qa_row, ["L01_V001", 3450, 'Anh ấy nói "Xin chào"'])

        trake_row = SubmissionFormatter.format_trake_row("L10_V001.mp4", ["1200", 1850, "2100", 2450])
        self.assertEqual(trake_row, ["L10_V001", 1200, 1850, 2100, 2450])

    def test_generate_csv_content_and_quoting(self):
        rows = [
            ["L01_V028", 3450, "Có 3 người, bao gồm nam và nữ"],
            ["L01_V028", 3450, 'Anh ấy nói "Tuyệt vời"'],
            ["L00_V000", 1234]
        ]
        csv_str = self.packager.generate_csv_content(rows)
        lines = csv_str.strip().split("\n")
        self.assertEqual(len(lines), 3)
        self.assertIn('"Có 3 người, bao gồm nam và nữ"', lines[0])
        self.assertIn('"Anh ấy nói ""Tuyệt vời"""', lines[1])
        self.assertEqual(lines[2], "L00_V000,1234")

    def test_package_zip_and_validation(self):
        csv_payloads = {
            "query-1-kis.csv": [["L00_V000", 1234], ["L01_V028", 25300]],
            "query-2-qa.csv": [["L01_V028", 3450, "5"], ["L02_V011", 1200, "Năm người"]],
            "query-3-trake.csv": [["L10_V001", 1200, 1850, 2100, 2450]]
        }

        zip_path = self.packager.package_zip(csv_payloads, zip_filename="team_test_round1.zip")
        self.assertTrue(os.path.exists(zip_path))

        # Check inside zip structure
        with zipfile.ZipFile(zip_path, "r") as zf:
            names = zf.namelist()
            self.assertIn("submission/query-1-kis.csv", names)
            self.assertNotIn("submission/query-2-kis.csv", names)
            self.assertIn("submission/query-2-qa.csv", names)
            self.assertIn("submission/query-3-trake.csv", names)


        # Validate zip
        is_valid, errors = SubmissionPackager.validate_zip(zip_path)
        self.assertTrue(is_valid)
        self.assertEqual(len(errors), 0)


if __name__ == "__main__":
    unittest.main()
