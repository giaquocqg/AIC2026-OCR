"""
Module: preprocessor.py
Description: Image quality assessment (Laplacian blur score), Adaptive CLAHE contrast enhancement,
             and Metadata frame_idx mapper for AI Challenge 2026 dataset.
"""

import os
import json
import cv2
import numpy as np
from typing import Dict, Any, Tuple, Optional, Union, List



class ImagePreprocessor:
    """Xử lý tiền kỳ cho ảnh Keyframe từ Kính thông minh (Sousveillance) & CCTV."""

    def __init__(self, 
                 max_dim: int = 1600, 
                 min_dim: int = 640, 
                 blur_threshold: float = 40.0,
                 enable_clahe: bool = True,
                 clahe_clip_limit: float = 2.0,
                 clahe_grid_size: Tuple[int, int] = (8, 8)):
        self.max_dim = max_dim
        self.min_dim = min_dim
        self.blur_threshold = blur_threshold
        self.enable_clahe = enable_clahe
        self.clahe = cv2.createCLAHE(clipLimit=clahe_clip_limit, tileGridSize=clahe_grid_size)

    def assess_blur_score(self, image_bgr: np.ndarray) -> float:
        """
        Tính toán độ sắc nét của ảnh dựa trên biến phân Laplacian.
        Điểm số càng thấp (< 40.0) -> Ảnh càng mờ do rung lắc chuyển động (Motion Blur).
        """
        if image_bgr is None or image_bgr.size == 0:
            return 0.0
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())

    def apply_adaptive_clahe(self, image_bgr: np.ndarray) -> np.ndarray:
        """
        Tăng cường tương phản cục bộ (CLAHE) trong không gian màu LAB.
        Cực kỳ hiệu quả khi người dùng bước từ ngoài nắng vào quầy chợ/siêu thị tối.
        """
        if not self.enable_clahe or image_bgr is None:
            return image_bgr
        
        lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l_enhanced = self.clahe.apply(l)
        lab_enhanced = cv2.merge((l_enhanced, a, b))
        return cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)

    def resize_for_detection(self, image_bgr: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Resize ảnh giữ nguyên tỉ lệ khung hình (Aspect Ratio).
        Đảm bảo cạnh dài nhất không vượt quá max_dim để tối ưu tốc độ và không mất chữ nhỏ.
        """
        h, w = image_bgr.shape[:2]
        max_side = max(h, w)
        if max_side <= self.max_dim and min(h, w) >= self.min_dim:
            return image_bgr, 1.0

        scale = self.max_dim / max_side
        new_w = int(w * scale)
        new_h = int(h * scale)
        resized = cv2.resize(image_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR)
        return resized, scale

    def process(self, image_bgr: np.ndarray) -> Dict[str, Any]:
        """Quy trình tiền xử lý ảnh tổng hợp."""
        blur_score = self.assess_blur_score(image_bgr)
        is_blurry = blur_score < self.blur_threshold

        # Nếu ảnh bị tối hoặc mờ, áp dụng CLAHE
        processed_img = self.apply_adaptive_clahe(image_bgr)
        resized_img, scale = self.resize_for_detection(processed_img)

        return {
            "processed_image": resized_img,
            "orig_shape": image_bgr.shape[:2],
            "scale": scale,
            "blur_score": round(blur_score, 2),
            "is_blurry": is_blurry
        }


import re


class FrameMetadataMapper:
    """
    Quản lý ánh xạ chính xác giữa tên file keyframe (ví dụ: 'keyframe_75.webp', '0042.jpg')
    và frame_id (toàn cục), frame_idx, timestamp thực tế của video theo dữ liệu AIC 2026.
    
    Hỗ trợ:
      1. fps.json (mapping video_id -> fps, ví dụ 873 video)
      2. index.json (mapping frame_id ordinal '0'..'N-1' -> 'video_id/keyframe_idx.webp')
      3. Youtube_URL.json (mapping video_id -> watch_url)
      4. Thư mục metadata JSON riêng lẻ ({video_id}.json)
    """

    # Regex nhận dạng số frame từ tên file như: "keyframe_75.webp", "frame_0150.jpg", "0042.png", "L21_V001_150.jpg"
    _FRAME_RE = re.compile(r"(?:keyframe_|frame_)?(\d+)", re.IGNORECASE)

    def __init__(self, 
                 metadata_dir: Optional[str] = None,
                 fps_file: Optional[Union[str, Dict[str, float]]] = None,
                 index_file: Optional[Union[str, Dict[str, str]]] = None,
                 youtube_url_file: Optional[Union[str, List[Dict[str, str]], Dict[str, str]]] = None,
                 default_fps: float = 25.0):
        self.metadata_dir = metadata_dir
        self.default_fps = default_fps
        self.cache: Dict[str, Dict[str, Any]] = {}
        
        # 1. Bảng tra FPS: video_id -> fps
        self.fps_map: Dict[str, float] = {}
        if fps_file:
            self._load_fps_map(fps_file)

        # 2. Bảng tra YouTube URL: video_id -> watch_url
        self.youtube_map: Dict[str, str] = {}
        if youtube_url_file:
            self._load_youtube_map(youtube_url_file)

        # 3. Bảng tra Index toàn cục: frame_id <-> (video_id, frame_idx)
        self.index_to_path: Dict[str, str] = {}
        self.path_to_index: Dict[str, str] = {}
        self.video_frame_to_index: Dict[Tuple[str, int], str] = {}
        if index_file:
            self._load_index_file(index_file)

    def _load_fps_map(self, fps_input: Union[str, Dict[str, float]]):
        """Nạp dữ liệu FPS từ file JSON hoặc dictionary."""
        if isinstance(fps_input, dict):
            self.fps_map = {k: float(v) for k, v in fps_input.items()}
        elif isinstance(fps_input, str) and os.path.exists(fps_input):
            try:
                with open(fps_input, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, dict):
                        self.fps_map = {k: float(v) for k, v in data.items()}
            except Exception:
                pass

    def _load_youtube_map(self, yt_input: Union[str, List[Dict[str, str]], Dict[str, str]]):
        """Nạp danh sách URL YouTube của các video."""
        if isinstance(yt_input, dict):
            self.youtube_map = dict(yt_input)
        elif isinstance(yt_input, list):
            for item in yt_input:
                if isinstance(item, dict) and "video_id" in item and "watch_url" in item:
                    self.youtube_map[item["video_id"]] = item["watch_url"]
        elif isinstance(yt_input, str) and os.path.exists(yt_input):
            try:
                with open(yt_input, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        for item in data:
                            if isinstance(item, dict) and "video_id" in item and "watch_url" in item:
                                self.youtube_map[item["video_id"]] = item["watch_url"]
                    elif isinstance(data, dict):
                        self.youtube_map = data
            except Exception:
                pass

    def _load_index_file(self, index_input: Union[str, Dict[str, str]]):
        """Nạp file index.json (190k frame) để ánh xạ 2 chiều chính xác với Milvus vector store."""
        raw_map: Dict[str, str] = {}
        if isinstance(index_input, dict):
            raw_map = index_input
        elif isinstance(index_input, str) and os.path.exists(index_input):
            try:
                with open(index_input, "r", encoding="utf-8") as f:
                    raw_map = json.load(f)
            except Exception:
                pass

        for fid_str, rel_path in raw_map.items():
            # rel_path: "L21_V001/keyframe_75.webp"
            self.index_to_path[fid_str] = rel_path
            self.path_to_index[rel_path] = fid_str
            
            parts = rel_path.replace("\\", "/").split("/")
            if len(parts) >= 2:
                vid = parts[-2]
                fname = parts[-1]
                f_idx = self.extract_frame_idx(fname)
                self.video_frame_to_index[(vid, f_idx)] = fid_str

    @staticmethod
    def extract_frame_idx(image_name: str) -> int:
        """Trích xuất frame_idx an toàn từ tên file ảnh."""
        clean_name = os.path.splitext(os.path.basename(image_name))[0]
        # Thử regex trước
        match = FrameMetadataMapper._FRAME_RE.search(clean_name)
        if match:
            try:
                return int(match.group(1))
            except ValueError:
                pass
        # Fallback tìm các cụm số bất kỳ
        digits = re.findall(r"\d+", clean_name)
        if digits:
            try:
                return int(digits[-1])
            except ValueError:
                pass
        return 0

    def get_fps(self, video_id: str) -> float:
        """Lấy FPS của video từ fps_map hoặc metadata cá nhân hoặc default_fps."""
        if video_id in self.fps_map:
            return self.fps_map[video_id]
        
        meta = self.load_video_metadata(video_id)
        if "fps" in meta:
            try:
                return float(meta["fps"])
            except (ValueError, TypeError):
                pass
        return self.default_fps

    def load_video_metadata(self, video_id: str) -> Dict[str, Any]:
        """Đọc metadata JSON của 1 video cụ thể nếu có thư mục metadata rời."""
        if video_id in self.cache:
            return self.cache[video_id]

        if not self.metadata_dir:
            return {}

        json_path = os.path.join(self.metadata_dir, f"{video_id}.json")
        if not os.path.exists(json_path):
            return {}

        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                self.cache[video_id] = data
                return data
        except Exception:
            return {}

    def get_frame_info(self, video_id: str, image_name: str) -> Dict[str, Any]:
        """
        Lấy thông tin frame_id, frame_idx, timestamp_sec, timestamp_ms, timestamp_str, và youtube_url chuẩn.
        """
        clean_name = os.path.splitext(os.path.basename(image_name))[0]
        frame_idx = self.extract_frame_idx(image_name)

        # 1. Tra cứu trong metadata JSON rời nếu có
        meta = self.load_video_metadata(video_id)
        if "frames" in meta and clean_name in meta["frames"]:
            frame_data = meta["frames"][clean_name]
            frame_idx = int(frame_data.get("frame_idx", frame_idx))
            timestamp_ms = int(frame_data.get("timestamp_ms", 0))
            timestamp_sec = round(timestamp_ms / 1000.0, 3)
            timestamp_str = frame_data.get("timestamp_str", f"{timestamp_ms // 60000:02d}:{(timestamp_ms % 60000) // 1000:02d}.{timestamp_ms % 1000:03d}")
        else:
            # 2. Tính timestamp qua FPS
            fps = self.get_fps(video_id)
            timestamp_sec = round(frame_idx / fps, 3) if fps > 0 else 0.0
            timestamp_ms = int(timestamp_sec * 1000)
            
            hours = timestamp_ms // 3600000
            mins = (timestamp_ms % 3600000) // 60000
            secs = (timestamp_ms % 60000) // 1000
            millis = timestamp_ms % 1000
            if hours > 0:
                timestamp_str = f"{hours:02d}:{mins:02d}:{secs:02d}.{millis:03d}"
            else:
                timestamp_str = f"{mins:02d}:{secs:02d}.{millis:03d}"

        # 3. Tra cứu ordinal frame_id (0..N-1) từ index.json
        frame_id = self.video_frame_to_index.get((video_id, frame_idx))
        if frame_id is None:
            # Fallback nếu chưa nạp index.json
            frame_id = f"{video_id}_{frame_idx}"

        # 4. Tra cứu YouTube URL kèm mốc thời gian
        base_yt_url = self.youtube_map.get(video_id, "")
        time_sec_int = int(timestamp_sec)
        if base_yt_url:
            yt_timestamp_url = f"{base_yt_url}&t={time_sec_int}s" if "?" in base_yt_url else f"{base_yt_url}?t={time_sec_int}s"
        else:
            yt_timestamp_url = ""

        return {
            "frame_id": str(frame_id),
            "video_id": video_id,
            "frame_idx": frame_idx,
            "timestamp": timestamp_sec,
            "timestamp_ms": timestamp_ms,
            "timestamp_str": timestamp_str,
            "watch_url": base_yt_url,
            "youtube_url": yt_timestamp_url
        }

