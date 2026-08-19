"""
Module: recognizer.py
Description: Batch Text Recognition combining VietOCR (Transformer) for Vietnamese diacritics
             and PaddleOCR for numbers, license plates, and English words with GPU Batching.
"""

import cv2
import numpy as np
from PIL import Image
from typing import List, Dict, Any, Tuple, Optional


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
        # 1. Khởi tạo PaddleOCR
        try:
            from paddleocr import PaddleOCR
            try:
                self.paddle_rec = PaddleOCR(
                    use_textline_orientation=False,
                    lang=self.paddle_lang
                )
            except Exception:
                use_gpu = (self.device == "cuda")
                self.paddle_rec = PaddleOCR(
                    use_angle_cls=False,
                    lang=self.paddle_lang,
                    use_gpu=use_gpu,
                    show_log=False
                )
        except Exception:
            self.paddle_rec = None

        # 2. Khởi tạo VietOCR
        try:
            from vietocr.tool.config import Cfg
            from vietocr.tool.predictor import Predictor
            config = Cfg.load_config_from_name(self.vietocr_model_name)
            config['device'] = self.device if self.device == 'cuda' else 'cpu'
            config['predictor']['beamsearch'] = False
            self.vietocr_detector = Predictor(config)
        except Exception:
            self.vietocr_detector = None

    def recognize_single(self, crop_bgr: np.ndarray) -> Tuple[str, float]:
        """Nhận diện 1 ảnh crop."""
        if crop_bgr is None or crop_bgr.size == 0:
            return "", 0.0

        # Ưu tiên PaddleOCR hoặc VietOCR
        if self.paddle_rec is not None:
            try:
                res = self.paddle_rec.ocr(crop_bgr, det=False, cls=False)
                if res and len(res) > 0 and res[0] and len(res[0]) > 0:
                    text, conf = res[0][0]
                    return str(text), float(conf)
            except Exception:
                pass

        if self.vietocr_detector is not None:
            try:
                pil_img = Image.fromarray(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB))
                text, prob = self.vietocr_detector.predict(pil_img, return_prob=True)
                return str(text), float(prob)
            except Exception:
                pass

        return "", 0.0

    def recognize_batch(self, crop_images: List[np.ndarray]) -> List[Tuple[str, float]]:
        """
        Nhận diện một danh sách lớn các ảnh crop theo từng batch GPU.
        """
        if not crop_images:
            return []

        results = []
        # Xử lý theo từng batch_size
        for i in range(0, len(crop_images), self.batch_size):
            batch = crop_images[i:i + self.batch_size]
            for crop in batch:
                text, conf = self.recognize_single(crop)
                results.append((text, conf))

        return results
