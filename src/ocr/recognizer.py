"""
Module: recognizer.py
Description: Batch Text Recognition combining VietOCR (Transformer) for Vietnamese diacritics
             and PaddleOCR for numbers, license plates, and English words with GPU Batching.
"""

import cv2
import numpy as np
from PIL import Image
import logging
from typing import List, Dict, Any, Tuple, Optional

logger = logging.getLogger("src.ocr.recognizer")


class BatchTextRecognizer:
    """Module nhận diện văn bản đa mô hình (Hybrid Batch Recognizer)."""

    def __init__(self, 
                 engine: str = "hybrid",
                 vietocr_model: str = "vgg_transformer",
                 paddle_lang: str = "vi",
                 batch_size: int = 64,
                 min_confidence: float = 0.40,
                 device: str = "cuda"):
        self.engine_name = engine
        self.vietocr_model_name = vietocr_model
        self.paddle_lang = paddle_lang
        self.batch_size = batch_size
        self.min_confidence = min_confidence
        self.device = device
        
        self.paddle_rec = None
        self.vietocr_detector = None
        self._init_models()

    def _init_models(self):
        """Khởi tạo VietOCR và PaddleOCR Recognizer."""
        use_gpu = (self.device == "cuda" or self.device == "gpu")

        # 1. Khởi tạo PaddleOCR Recognizer
        try:
            from paddleocr import PaddleOCR
            try:
                self.paddle_rec = PaddleOCR(
                    use_textline_orientation=False,
                    lang=self.paddle_lang
                )
            except Exception as e3:
                logger.debug(f"PaddleOCR 3.x rec fallback to 2.x API: {e3}")
                self.paddle_rec = PaddleOCR(
                    use_angle_cls=False,
                    lang=self.paddle_lang,
                    use_gpu=use_gpu,
                    show_log=False
                )
            logger.info("✓ PaddleOCR Text Recognizer initialized successfully")
        except Exception as e:
            logger.warning(f"⚠️ Không thể khởi tạo PaddleOCR Recognizer ({self.device}): {e}")
            self.paddle_rec = None

        # 2. Khởi tạo VietOCR Predictor
        try:
            from vietocr.tool.config import Cfg
            from vietocr.tool.predictor import Predictor
            config = Cfg.load_config_from_name(self.vietocr_model_name)
            config['device'] = 'cuda:0' if use_gpu else 'cpu'
            config['predictor']['beamsearch'] = False
            self.vietocr_detector = Predictor(config)
            logger.info(f"✓ VietOCR ({self.vietocr_model_name}) initialized successfully on {config['device']}")
        except Exception as e:
            logger.warning(f"⚠️ Không thể khởi tạo VietOCR ({self.vietocr_model_name}) on {self.device}: {e}")
            self.vietocr_detector = None

        if self.paddle_rec is None and self.vietocr_detector is None:
            logger.error("❌ CẢNH BÁO NGHIÊM TRỌNG: Cả PaddleOCR và VietOCR đều không khởi tạo được!")

    @staticmethod
    def count_vietnamese_diacritics(text: str) -> int:
        """Đếm số lượng nguyên âm có dấu tiếng Việt trong chuỗi."""
        vn_chars = "àáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ"
        return sum(1 for c in text.lower() if c in vn_chars)

    def recognize_single(self, crop_bgr: np.ndarray) -> Tuple[str, float]:
        """
        Nhận diện 1 ảnh crop theo chiến lược Hybrid thực sự:
        - VietOCR Transformer: Ưu việt cho tiếng Việt có dấu, chữ viết trên biển hiệu dài.
        - PaddleOCR: Ưu việt cho chữ số, mã hiệu, biển số xe, ký tự tiếng Anh.
        - Chọn kết quả có độ tự tin cao nhất và độ chính xác dấu tiếng Việt tốt nhất.
        """
        if crop_bgr is None or crop_bgr.size == 0:
            return "", 0.0

        paddle_text, paddle_conf = "", 0.0
        vietocr_text, vietocr_conf = "", 0.0

        # 1. Chạy PaddleOCR
        if self.paddle_rec is not None:
            try:
                res = self.paddle_rec.ocr(crop_bgr, det=False, cls=False)
                if res and len(res) > 0 and res[0] and len(res[0]) > 0:
                    paddle_text = str(res[0][0][0]).strip()
                    paddle_conf = float(res[0][0][1])
            except Exception as e:
                logger.debug(f"PaddleOCR recognition failed: {e}")


        # 2. Chạy VietOCR
        if self.vietocr_detector is not None:
            try:
                pil_img = Image.fromarray(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB))
                v_text, v_prob = self.vietocr_detector.predict(pil_img, return_prob=True)
                vietocr_text = str(v_text).strip()
                vietocr_conf = float(v_prob)
            except Exception as e:
                logger.debug(f"VietOCR recognize failed: {e}")

        # 3. Chiến lược Hybrid Voting / Scoring
        if paddle_text and not vietocr_text:
            return paddle_text, paddle_conf
        if vietocr_text and not paddle_text:
            return vietocr_text, vietocr_conf
        if not paddle_text and not vietocr_text:
            return "", 0.0

        # Nếu cả 2 đều có kết quả:
        vn_vietocr = self.count_vietnamese_diacritics(vietocr_text)
        vn_paddle = self.count_vietnamese_diacritics(paddle_text)

        # Nếu VietOCR phát hiện có dấu tiếng Việt chuẩn và confidence khá (>= 0.65)
        if vn_vietocr > 0 and vn_vietocr >= vn_paddle and vietocr_conf >= 0.65:
            return vietocr_text, vietocr_conf

        # Nếu là chuỗi số thuần hoặc mã ký hiệu (biển số xe, số nhà, v.v.)
        if paddle_text.replace(" ", "").isalnum() and vn_vietocr == 0:
            if paddle_conf >= vietocr_conf - 0.10:
                return paddle_text, paddle_conf

        # Trường hợp chung: Chọn theo confidence cao hơn
        if vietocr_conf >= paddle_conf:
            return vietocr_text, vietocr_conf
        else:
            return paddle_text, paddle_conf

    def recognize_batch(self, crop_images: List[np.ndarray]) -> List[Tuple[str, float]]:
        """
        Nhận diện một danh sách lớn các ảnh crop theo từng batch GPU/CPU.
        Hỗ trợ tăng tốc xử lý theo lô ảnh.
        """
        if not crop_images:
            return []

        results = []
        # Xử lý theo từng batch_size
        for i in range(0, len(crop_images), self.batch_size):
            batch = crop_images[i:i + self.batch_size]

            # Kiểm tra nếu VietOCR có hỗ trợ predict_batch
            if self.vietocr_detector is not None and hasattr(self.vietocr_detector, "predict_batch") and self.paddle_rec is None:
                try:
                    pil_batch = [Image.fromarray(cv2.cvtColor(c, cv2.COLOR_BGR2RGB)) for c in batch if c is not None and c.size > 0]
                    v_res = self.vietocr_detector.predict_batch(pil_batch, return_prob=True)
                    for text, prob in v_res:
                        results.append((str(text), float(prob)))
                    continue
                except Exception as e:
                    logger.debug(f"VietOCR predict_batch fallback to single: {e}")

            for crop in batch:
                text, conf = self.recognize_single(crop)
                results.append((text, conf))

        return results

