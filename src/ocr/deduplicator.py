"""
Module: deduplicator.py
Description: Temporal text deduplicator across adjacent video keyframes.
             Clustering repeated text detections (billboards, subtitles, price tags)
             to reduce search index redundancy by ~70% and optimize R@k retrieval.
"""

import math
from typing import List, Dict, Any, Optional
import difflib


class TemporalDeduplicator:
    """
    Gộp các phát hiện văn bản trùng lặp xuất hiện liên tục trong chuỗi keyframes liền kề
    của cùng một video (ví dụ biển hiệu quầy hàng, phụ đề, bảng giá).
    """

    def __init__(self, 
                 enable_temporal_merge: bool = True,
                 similarity_threshold: float = 0.82,
                 max_frame_gap: int = 120,
                 iou_threshold: float = 0.25):
        self.enable = enable_temporal_merge
        self.similarity_threshold = similarity_threshold
        self.max_frame_gap = max_frame_gap
        self.iou_threshold = iou_threshold


    @staticmethod
    def compute_text_similarity(text1: str, text2: str) -> float:
        """
        Tính độ tương đồng giữa hai chuỗi văn bản (dùng difflib SequenceMatcher).
        """
        if not text1 or not text2:
            return 0.0
        t1 = text1.strip().lower()
        t2 = text2.strip().lower()
        if t1 == t2:
            return 1.0
        return difflib.SequenceMatcher(None, t1, t2).ratio()

    @staticmethod
    def compute_bbox_iou(box1: List[float], box2: List[float]) -> float:
        """
        Tính IoU giữa 2 bounding box chuẩn hóa [ymin, xmin, ymax, xmax].
        """
        ymin1, xmin1, ymax1, xmax1 = box1
        ymin2, xmin2, ymax2, xmax2 = box2

        inter_ymin = max(ymin1, ymin2)
        inter_xmin = max(xmin1, xmin2)
        inter_ymax = min(ymax1, ymax2)
        inter_xmax = min(xmax1, xmax2)

        if inter_ymax <= inter_ymin or inter_xmax <= inter_xmin:
            return 0.0

        inter_area = (inter_ymax - inter_ymin) * (inter_xmax - inter_xmin)
        area1 = (ymax1 - ymin1) * (xmax1 - xmin1)
        area2 = (ymax2 - ymin2) * (xmax2 - xmin2)

        union_area = area1 + area2 - inter_area
        if union_area <= 0:
            return 0.0
        return inter_area / union_area

    @staticmethod
    def compute_center_distance(box1: List[float], box2: List[float]) -> float:
        """Tính khoảng cách Euclid giữa tâm 2 bbox chuẩn hóa."""
        cy1 = (box1[0] + box1[2]) / 2.0
        cx1 = (box1[1] + box1[3]) / 2.0
        cy2 = (box2[0] + box2[2]) / 2.0
        cx2 = (box2[1] + box2[3]) / 2.0
        return math.sqrt((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2)

    def is_same_track(self, det1: Dict[str, Any], det2: Dict[str, Any]) -> bool:
        """
        Kiểm tra 2 detection ở 2 frame liền kề có cùng thuộc 1 text object không.
        """
        # 1. So sánh độ tương đồng text không dấu
        sim_unsigned = self.compute_text_similarity(
            det1.get("text_unsigned", ""), 
            det2.get("text_unsigned", "")
        )
        if sim_unsigned < self.similarity_threshold:
            # So sánh thêm text có dấu nếu không dấu thấp
            sim_acc = self.compute_text_similarity(
                det1.get("text", ""), 
                det2.get("text", "")
            )
            if sim_acc < self.similarity_threshold:
                return False

        # 2. So sánh vị trí không gian (IoU hoặc tâm gần nhau)
        box1 = det1.get("bbox_norm", [0, 0, 0, 0])
        box2 = det2.get("bbox_norm", [0, 0, 0, 0])
        iou = self.compute_bbox_iou(box1, box2)
        center_dist = self.compute_center_distance(box1, box2)

        # Trùng lặp nếu IoU > threshold hoặc tâm không lệch quá 0.25 đường chéo
        return (iou >= self.iou_threshold) or (center_dist <= 0.25)

    def deduplicate_video_detections(self, frame_records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Gộp các phát hiện văn bản trên toàn bộ danh sách keyframe của 1 video.
        
        Input: Danh sách bản ghi theo frame:
        [
            {
                "video_id": "L01_V001",
                "frame_idx": 42,
                "timestamp_ms": 1680,
                "timestamp_str": "00:01.680",
                "detections": [
                    {
                        "text": "CƠM TẤM BA GHIỀN",
                        "text_unsigned": "com tam ba ghien",
                        "confidence": 0.95,
                        "bbox_norm": [0.1, 0.2, 0.3, 0.8],
                        "polygon": [...],
                        "entities": [...]
                    }
                ]
            }
        ]
        
        Output: Danh sách bản ghi đã gộp theo dòng thời gian (Temporal Tracks)
        """
        if not self.enable:
            # Flatten tất cả detection mà không gộp
            flat_records = []
            for fr in frame_records:
                vid = fr.get("video_id", "")
                f_id = fr.get("frame_id", "")
                f_idx = fr.get("frame_idx", 0)
                t_sec = fr.get("timestamp", 0.0)
                t_ms = fr.get("timestamp_ms", 0)
                t_str = fr.get("timestamp_str", "")
                yt_url = fr.get("youtube_url", "")
                for det in fr.get("detections", []):
                    rec = dict(det)
                    rec["frame_id"] = f_id
                    rec["video_id"] = vid
                    rec["frame_idx"] = f_idx
                    rec["timestamp"] = t_sec
                    rec["timestamp_ms"] = t_ms
                    rec["timestamp_str"] = t_str
                    rec["youtube_url"] = yt_url
                    rec["frame_start"] = f_idx
                    rec["frame_end"] = f_idx
                    rec["occurrences"] = 1
                    flat_records.append(rec)
            return flat_records

        # Sắp xếp theo thứ tự frame_idx tăng dần
        sorted_frames = sorted(frame_records, key=lambda x: x.get("frame_idx", 0))

        active_tracks: List[Dict[str, Any]] = []
        completed_tracks: List[Dict[str, Any]] = []

        for frame_item in sorted_frames:
            vid = frame_item.get("video_id", "")
            f_id = frame_item.get("frame_id", "")
            f_idx = frame_item.get("frame_idx", 0)
            t_sec = frame_item.get("timestamp", 0.0)
            t_ms = frame_item.get("timestamp_ms", 0)
            t_str = frame_item.get("timestamp_str", "")
            yt_url = frame_item.get("youtube_url", "")
            detections = frame_item.get("detections", [])

            # Đóng các track đã quá khoảng cách max_frame_gap
            still_active = []
            for track in active_tracks:
                if (f_idx - track["last_frame_idx"]) > self.max_frame_gap:
                    completed_tracks.append(track)
                else:
                    still_active.append(track)
            active_tracks = still_active

            # Khớp detection hiện tại với các active track
            matched_track_indices = set()
            for det in detections:
                matched = False
                for idx, track in enumerate(active_tracks):
                    if idx in matched_track_indices:
                        continue
                    
                    last_det = track["last_detection"]
                    if self.is_same_track(last_det, det):
                        # Khớp vào track này
                        track["frame_end"] = f_idx
                        track["last_frame_idx"] = f_idx
                        track["frame_indices"].append(f_idx)
                        track["occurrences"] += 1
                        track["last_detection"] = det

                        # Cập nhật thông tin đại diện nếu tự tin cao hơn
                        if det.get("confidence", 0.0) > track["confidence"]:
                            track["best_frame_id"] = f_id
                            track["best_timestamp"] = t_sec
                            track["best_youtube_url"] = yt_url
                            track["text"] = det.get("text", track["text"])
                            track["text_unsigned"] = det.get("text_unsigned", track["text_unsigned"])
                            track["confidence"] = det.get("confidence", track["confidence"])
                            track["bbox_norm"] = det.get("bbox_norm", track["bbox_norm"])
                            track["polygon"] = det.get("polygon", track["polygon"])
                            track["best_frame_idx"] = f_idx
                            track["best_timestamp_ms"] = t_ms
                            track["best_timestamp_str"] = t_str

                        # Hợp nhất thực thể
                        for ent in det.get("entities", []):
                            if ent not in track["entities"]:
                                track["entities"].append(ent)

                        matched_track_indices.add(idx)
                        matched = True
                        break

                if not matched:
                    # Tạo một track mới
                    new_track = {
                        "video_id": vid,
                        "best_frame_id": f_id,
                        "best_timestamp": t_sec,
                        "best_youtube_url": yt_url,
                        "frame_start": f_idx,
                        "frame_end": f_idx,
                        "last_frame_idx": f_idx,
                        "best_frame_idx": f_idx,
                        "best_timestamp_ms": t_ms,
                        "best_timestamp_str": t_str,
                        "frame_indices": [f_idx],
                        "occurrences": 1,
                        "text": det.get("text", ""),
                        "text_unsigned": det.get("text_unsigned", ""),
                        "confidence": det.get("confidence", 0.0),
                        "bbox_norm": det.get("bbox_norm", [0, 0, 0, 0]),
                        "polygon": det.get("polygon", []),
                        "entities": list(det.get("entities", [])),
                        "last_detection": det
                    }
                    active_tracks.append(new_track)

        # Thêm tất cả active track còn lại vào completed
        completed_tracks.extend(active_tracks)

        # Chuyển đổi completed_tracks thành danh sách phẳng chuẩn hóa
        output_records = []
        for track in completed_tracks:
            output_records.append({
                "frame_id": track.get("best_frame_id", ""),
                "video_id": track["video_id"],
                "frame_idx": track["best_frame_idx"],
                "timestamp": track.get("best_timestamp", 0.0),
                "timestamp_ms": track["best_timestamp_ms"],
                "timestamp_str": track["best_timestamp_str"],
                "youtube_url": track.get("best_youtube_url", ""),
                "frame_start": track["frame_start"],
                "frame_end": track["frame_end"],
                "frame_span": track["frame_end"] - track["frame_start"],
                "occurrences": track["occurrences"],
                "text": track["text"],
                "text_unsigned": track["text_unsigned"],
                "confidence": round(float(track["confidence"]), 4),
                "bbox_norm": track["bbox_norm"],
                "polygon": track["polygon"],
                "entities": track["entities"]
            })

        # Sắp xếp output theo frame_start
        output_records.sort(key=lambda x: (x["video_id"], x["frame_start"]))
        return output_records

