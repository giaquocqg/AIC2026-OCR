"""
Module: detector.py
Description: DBNet++ / PaddleOCR Text Detection wrapper with 4-point polygon extraction,
             box filtering, and perspective rectification for curved/tilted scene text.
"""

import cv2
import numpy as np
import logging
from typing import List, Dict, Any, Tuple, Optional

logger = logging.getLogger("src.ocr.detector")


class TextDetector:
    """Module phát hiện vị trí văn bản trong ảnh (Text Detection)."""

    def __init__(self, 
                 engine: str = "paddleocr",
                 use_angle_cls: bool = True,
                 det_db_thresh: float = 0.3,
                 det_db_box_thresh: float = 0.5,
                 det_db_unclip_ratio: float = 1.6,
                 min_box_area: float = 80.0,
                 device: str = "cuda"):
        self.engine_name = engine
        self.min_box_area = min_box_area
        self.device = device
        self.detector = None
        self._init_detector(det_db_thresh, det_db_box_thresh, det_db_unclip_ratio, use_angle_cls)

    def _init_detector(self, db_thresh: float, box_thresh: float, unclip_ratio: float, use_angle_cls: bool):
        """Khởi tạo model PaddleOCR Detection."""
        try:
            from paddleocr import PaddleOCR
            use_gpu = (self.device == "cuda" or self.device == "gpu")
            try:
                # PaddleOCR 3.x
                self.detector = PaddleOCR(
                    use_textline_orientation=use_angle_cls,
                    lang="vi",
                    text_det_thresh=db_thresh,
                    text_det_box_thresh=box_thresh,
                    text_det_unclip_ratio=unclip_ratio
                )
            except Exception as e3:
                logger.debug(f"PaddleOCR 3.x init fallback to 2.x API: {e3}")
                # Legacy PaddleOCR 2.x
                self.detector = PaddleOCR(
                    use_angle_cls=use_angle_cls,
                    lang="vi",
                    use_gpu=use_gpu,
                    det_db_thresh=db_thresh,
                    det_db_box_thresh=box_thresh,
                    det_db_unclip_ratio=unclip_ratio,
                    show_log=False
                )
        except Exception as e:
            logger.error(f"❌ Không thể khởi tạo PaddleOCR Detector ({self.device}): {e}")
            self.detector = None

    def detect(self, image_bgr: np.ndarray) -> List[Dict[str, Any]]:
        """
        Phát hiện tất cả các vùng chứa chữ trong ảnh.
        Trả về danh sách các vùng với: polygon 4 điểm, bbox chuẩn hóa, và ảnh crop đã nắn thẳng.
        """
        if image_bgr is None or image_bgr.size == 0:
            return []

        h, w = image_bgr.shape[:2]
        detections = []

        if self.detector is not None:
            try:
                raw_boxes = []
                # 1. Gọi trực tiếp text_detector (DBNet GPU cực nhanh và chuẩn xác)
                if hasattr(self.detector, "text_detector") and callable(self.detector.text_detector):
                    try:
                        res = self.detector.text_detector(image_bgr)
                        if isinstance(res, tuple) and len(res) > 0 and res[0] is not None:
                            raw_boxes = res[0]
                    except Exception:
                        raw_boxes = []

                # 2. Fallback sang self.detector.ocr(..., rec=False, cls=False)
                if not raw_boxes:
                    try:
                        results = self.detector.ocr(image_bgr, rec=False, cls=False)
                        if results and len(results) > 0 and results[0] is not None:
                            raw_boxes = results[0]
                    except Exception:
                        raw_boxes = []

                for box in raw_boxes:
                    poly = np.array(box, dtype=np.float32)
                    if poly.ndim != 2 or len(poly) < 4:
                        continue

                    area = cv2.contourArea(poly)
                    if area < self.min_box_area:
                        continue

                    # Tính normalized bounding box [ymin, xmin, ymax, xmax]
                    xmin = max(0.0, float(np.min(poly[:, 0])) / w)
                    xmax = min(1.0, float(np.max(poly[:, 0])) / w)
                    ymin = max(0.0, float(np.min(poly[:, 1])) / h)
                    ymax = min(1.0, float(np.max(poly[:, 1])) / h)

                    # Crop và nắn thẳng (Perspective Rectification)
                    crop_img = self.crop_and_rectify(image_bgr, poly)

                    detections.append({
                        "polygon": poly.astype(int).tolist(),
                        "bbox_norm": [ymin, xmin, ymax, xmax],
                        "area": float(area),
                        "crop_img": crop_img
                    })
            except Exception as e:
                logger.warning(f"Lỗi khi thực thi text detection: {e}")



        return detections

    @staticmethod
    def crop_and_rectify(image_bgr: np.ndarray, polygon: np.ndarray) -> np.ndarray:
        """
        Nắn thẳng vùng chữ đa giác 4 điểm bằng phép biến đổi phối cảnh (Perspective Transform).
        Rất quan trọng cho nhãn mác uốn lượn trên chai lọ hoặc biển hiệu góc nghiêng.
        """
        pts = polygon.astype(np.float32)
        # Sắp xếp 4 điểm: top-left, top-right, bottom-right, bottom-left
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]

        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]

        (tl, tr, br, bl) = rect
        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        maxWidth = max(int(widthA), int(widthB), 8)

        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        maxHeight = max(int(heightA), int(heightB), 8)

        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]], dtype="float32")

        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(image_bgr, M, (maxWidth, maxHeight))
        return warped
