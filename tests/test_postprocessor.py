"""
Unit Tests: Vietnamese Postprocessor & Entity Tagger
"""

import unittest
from src.ocr.postprocessor import VietnamesePostProcessor


class TestVietnamesePostProcessor(unittest.TestCase):

    def test_unicode_normalization(self):
        text_decomp = "hòa bình"
        norm = VietnamesePostProcessor.normalize_unicode_nfc(text_decomp)
        self.assertEqual(norm, "hòa bình")

    def test_to_unsigned(self):
        cases = [
            ("Cơm Tấm Ba Ghiền", "com tam ba ghien"),
            ("Phở Bò Tái Nạm 45k", "pho bo tai nam 45k"),
            ("Đường Nguyễn Trãi Quận 5", "duong nguyen trai quan 5"),
            ("Hội thi AI Challenge TP.HCM 2026", "hoi thi ai challenge tp.hcm 2026"),
        ]
        for src, expected in cases:
            res = VietnamesePostProcessor.to_unsigned(src)
            self.assertEqual(res, expected, f"Failed for {src}: got {res}, expected {expected}")

    def test_entity_extraction_price(self):
        text = "Quán cơm tấm 45k, trà đá 5.000đ, lẩu cá 150 nghìn"
        entities = VietnamesePostProcessor.extract_entities(text)
        prices = [e["value"] for e in entities if e["type"] == "PRICE"]
        self.assertTrue(any("45k" in p for p in prices))
        self.assertTrue(any("5.000đ" in p for p in prices))
        self.assertTrue(any("150 nghìn" in p for p in prices))

    def test_entity_extraction_license_plate(self):
        text = "Xe máy biển số 59-X3 123.45 và ô tô 51F-9999"
        entities = VietnamesePostProcessor.extract_entities(text)
        plates = [e["value"] for e in entities if e["type"] == "LICENSE_PLATE"]
        self.assertTrue(len(plates) >= 1)
        self.assertTrue(any("59-X3" in p or "59X3" in p for p in plates))

    def test_entity_extraction_phone(self):
        text = "Liên hệ hotline: 0901234567 hoặc 02838354409"
        entities = VietnamesePostProcessor.extract_entities(text)
        phones = [e["value"] for e in entities if e["type"] == "PHONE_NUMBER"]
        self.assertIn("0901234567", phones)


if __name__ == "__main__":
    unittest.main()
