"""
Module: postprocessor.py
Description: Vietnamese text normalization, unicode NFC standardizer, unsigned text generator,
             and Entity Extractors for AI Challenge 2026 (KIS & Q&A).
"""

import re
import unicodedata
from typing import Dict, List, Any, Optional

# Vietnamese tone placement map (chuẩn hóa kiểu gõ mới: hòa, thúy thay vì hoà, thuý)
VIETNAMESE_VOWELS = {
    'a': 'a', 'à': 'a', 'á': 'a', 'ả': 'a', 'ã': 'a', 'ạ': 'a',
    'ă': 'ă', 'ằ': 'ă', 'ắ': 'ă', 'ẳ': 'ă', 'ẵ': 'ă', 'ặ': 'ă',
    'â': 'â', 'ầ': 'â', 'ấ': 'â', 'ẩ': 'â', 'ẫ': 'â', 'ậ': 'â',
    'e': 'e', 'è': 'e', 'é': 'e', 'ẻ': 'e', 'ẽ': 'e', 'ẹ': 'e',
    'ê': 'ê', 'ề': 'ê', 'ế': 'ê', 'ể': 'ê', 'ễ': 'ê', 'ệ': 'ê',
    'i': 'i', 'ì': 'i', 'í': 'i', 'ỉ': 'i', 'ĩ': 'i', 'ị': 'i',
    'o': 'o', 'ò': 'o', 'ó': 'o', 'ỏ': 'o', 'õ': 'o', 'ọ': 'o',
    'ô': 'ô', 'ồ': 'ô', 'ố': 'ô', 'ổ': 'ô', 'ỗ': 'ô', 'ộ': 'ô',
    'ơ': 'ơ', 'ờ': 'ơ', 'ớ': 'ơ', 'ở': 'ơ', 'ỡ': 'ơ', 'ợ': 'ơ',
    'u': 'u', 'ù': 'u', 'ú': 'u', 'ủ': 'u', 'ũ': 'u', 'ụ': 'u',
    'ư': 'ư', 'ừ': 'ư', 'ứ': 'ư', 'ử': 'ư', 'ữ': 'ư', 'ự': 'ư',
    'y': 'y', 'ỳ': 'y', 'ý': 'y', 'ỷ': 'y', 'ỹ': 'y', 'ỵ': 'y'
}

# Bảng chuyển đổi ký tự tiếng Việt sang không dấu
CHAR_MAP_UNSIGNED = {
    'à': 'a', 'á': 'a', 'ả': 'a', 'ã': 'a', 'ạ': 'a',
    'ă': 'a', 'ằ': 'a', 'ắ': 'a', 'ẳ': 'a', 'ẵ': 'a', 'ặ': 'a',
    'â': 'a', 'ầ': 'a', 'ấ': 'a', 'ẩ': 'a', 'ẫ': 'a', 'ậ': 'a',
    'è': 'e', 'é': 'e', 'ẻ': 'e', 'ẽ': 'e', 'ẹ': 'e',
    'ê': 'e', 'ề': 'e', 'ế': 'e', 'ể': 'e', 'ễ': 'e', 'ệ': 'e',
    'ì': 'i', 'í': 'i', 'ỉ': 'i', 'ĩ': 'i', 'ị': 'i',
    'ò': 'o', 'ó': 'o', 'ỏ': 'o', 'õ': 'o', 'ọ': 'o',
    'ô': 'o', 'ồ': 'o', 'ố': 'o', 'ổ': 'o', 'ỗ': 'o', 'ộ': 'o',
    'ơ': 'o', 'ờ': 'o', 'ớ': 'o', 'ở': 'o', 'ỡ': 'o', 'ợ': 'o',
    'ù': 'u', 'ú': 'u', 'ủ': 'u', 'ũ': 'u', 'ụ': 'u',
    'ư': 'u', 'ừ': 'u', 'ứ': 'u', 'ử': 'u', 'ữ': 'u', 'ự': 'u',
    'ỳ': 'y', 'ý': 'y', 'ỷ': 'y', 'ỹ': 'y', 'ỵ': 'y',
    'đ': 'd',
    'À': 'A', 'Á': 'A', 'Ả': 'A', 'Ã': 'A', 'Ạ': 'A',
    'Ă': 'A', 'Ằ': 'A', 'Ắ': 'A', 'Ẳ': 'A', 'Ẵ': 'A', 'Ặ': 'A',
    'Â': 'A', 'Ầ': 'A', 'Ấ': 'A', 'Ẩ': 'A', 'Ẫ': 'A', 'Ậ': 'A',
    'È': 'E', 'É': 'E', 'Ẻ': 'E', 'Ẽ': 'E', 'Ẹ': 'E',
    'Ê': 'E', 'Ề': 'E', 'Ế': 'E', 'Ể': 'E', 'Ễ': 'E', 'Ệ': 'E',
    'Ì': 'I', 'Í': 'I', 'Ỉ': 'I', 'Ĩ': 'I', 'Ị': 'I',
    'Ò': 'O', 'Ó': 'O', 'Ỏ': 'O', 'Õ': 'O', 'Ọ': 'O',
    'Ô': 'O', 'Ồ': 'O', 'Ố': 'O', 'Ổ': 'O', 'Ỗ': 'O', 'Ộ': 'O',
    'Ơ': 'O', 'Ờ': 'O', 'Ớ': 'O', 'Ở': 'O', 'Ỡ': 'O', 'Ợ': 'O',
    'Ù': 'U', 'Ú': 'U', 'Ủ': 'U', 'Ũ': 'U', 'Ụ': 'U',
    'Ư': 'U', 'Ừ': 'U', 'Ứ': 'U', 'Ử': 'U', 'Ữ': 'U', 'Ự': 'U',
    'Ỳ': 'Y', 'Ý': 'Y', 'Ỷ': 'Y', 'Ỹ': 'Y', 'Ỵ': 'Y',
    'Đ': 'D'
}

# Regular expressions for AIC entities
REGEX_PRICE = re.compile(r'\b(\d+(?:[.,]\d+)?\s*(?:k|vnđ|vnd|đ|nghìn|ngàn|triệu|tr))\b', re.IGNORECASE)
REGEX_LICENSE_PLATE = re.compile(r'\b([0-9]{2}[-\s]?[A-Z]{1,2}[0-9]?[-\s]?[0-9]{3,5}(?:\.[0-9]{2})?)\b', re.IGNORECASE)
REGEX_PHONE = re.compile(r'\b((?:0|\+84)[235789][0-9]{8})\b')
REGEX_QUANTITY = re.compile(r'\b(\d+(?:\.\d+)?)\b')
REGEX_TIME = re.compile(r'\b([0-2]?[0-9][h:][0-5][0-9](?::[0-5][0-9])?)\b', re.IGNORECASE)


class VietnamesePostProcessor:
    """Xử lý hậu kỳ văn bản tiếng Việt cho bài toán Video Retrieval AIC 2026."""

    @staticmethod
    def normalize_unicode_nfc(text: str) -> str:
        """Đưa văn bản về chuẩn Unicode dựng sẵn (NFC)."""
        if not text:
            return ""
        return unicodedata.normalize('NFC', text.strip())

    @staticmethod
    def to_unsigned(text: str) -> str:
        """
        Chuyển chuỗi tiếng Việt thành chuỗi không dấu (lowercase).
        Phục vụ việc tìm kiếm khi người dùng/Operator gõ không dấu.
        """
        if not text:
            return ""
        text_nfc = unicodedata.normalize('NFC', text)
        res = []
        for char in text_nfc:
            res.append(CHAR_MAP_UNSIGNED.get(char, char))
        return "".join(res).lower().strip()

    @staticmethod
    def clean_text(text: str) -> str:
        """Lọc bỏ các ký tự rác vô nghĩa từ OCR lỗi."""
        if not text:
            return ""
        text = VietnamesePostProcessor.normalize_unicode_nfc(text)
        # Xóa khoảng trắng thừa và ký tự điều khiển
        text = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    @staticmethod
    def extract_entities(text: str) -> List[Dict[str, Any]]:
        """
        Trích xuất các thực thể quan trọng (Giá tiền, Biển số xe, SĐT, Con số, Giờ giấc)
        phục vụ trả lời câu hỏi Visual Q&A.
        """
        entities = []
        if not text:
            return entities

        # 1. Giá tiền (Price)
        for match in REGEX_PRICE.finditer(text):
            entities.append({
                "type": "PRICE",
                "value": match.group(1),
                "span": match.span()
            })

        # 2. Biển số xe (License Plate)
        for match in REGEX_LICENSE_PLATE.finditer(text):
            entities.append({
                "type": "LICENSE_PLATE",
                "value": match.group(1).upper(),
                "span": match.span()
            })

        # 3. Số điện thoại (Phone)
        for match in REGEX_PHONE.finditer(text):
            entities.append({
                "type": "PHONE_NUMBER",
                "value": match.group(1),
                "span": match.span()
            })

        # 4. Thời gian (Time)
        for match in REGEX_TIME.finditer(text):
            entities.append({
                "type": "TIME",
                "value": match.group(1),
                "span": match.span()
            })

        # 5. Con số số lượng (Quantity/Digit)
        for match in REGEX_QUANTITY.finditer(text):
            entities.append({
                "type": "NUMBER",
                "value": match.group(1),
                "span": match.span()
            })

        return entities

    @classmethod
    def process_detection(cls, raw_text: str, confidence: float, bbox: List[float], polygon: Optional[List[List[int]]] = None) -> Dict[str, Any]:
        """Đóng gói một detection thành định dạng chuẩn."""
        cleaned = cls.clean_text(raw_text)
        unsigned = cls.to_unsigned(cleaned)
        entities = cls.extract_entities(cleaned)
        
        return {
            "text": cleaned,
            "text_unsigned": unsigned,
            "confidence": round(float(confidence), 4),
            "bbox_norm": [round(float(c), 4) for c in bbox],  # [ymin, xmin, ymax, xmax] in [0, 1]
            "polygon": polygon if polygon is not None else [],
            "entities": entities
        }
