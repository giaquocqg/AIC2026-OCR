"""
Module: agent_tool.py
Description: Spatial OCR Tool Card for Agentic AI LLM Planner (STAR Framework).
             Provides standardized Tool Calling interface for GPT-4o, Gemini 2.0 / 3.0,
             and Claude 3.7 to inspect scene text, extract entities, and localize ROIs.
"""

from typing import List, Dict, Any, Optional
from .indexer import OCRIndexSearcher
from .postprocessor import VietnamesePostProcessor


class SpatialOCRTool:
    """
    Spatial Tool Card cho LLM Planner theo khung lập luận STAR (Spatiotemporal Reasoning).
    Cung cấp các hàm công cụ giúp Agent tìm kiếm, đọc chữ, và phóng to vùng quan tâm (ROI).
    """

    def __init__(self, searcher_or_db_path: Any):
        if isinstance(searcher_or_db_path, str):
            self.searcher = OCRIndexSearcher(searcher_or_db_path)
        else:
            self.searcher = searcher_or_db_path

    def tool_search_ocr(self, query_text: str, top_k: int = 20, entity_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        [TOOL 1] Tìm kiếm các khoảnh khắc video có chứa từ khóa hoặc bảng hiệu văn bản.
        
        Args:
            query_text: Từ khóa cần tìm (ví dụ: 'cơm tấm', 'ba ghiền', '59-X3', '45k').
            top_k: Số lượng kết quả tối đa trả về.
            entity_type: (Tùy chọn) Lọc theo loại thực thể: 'PRICE', 'LICENSE_PLATE', 'PHONE_NUMBER', 'NUMBER', 'TIME'.
        
        Returns:
            Danh sách các ứng viên video_id, frame_idx, text phát hiện và gợi ý nộp bài.
        """
        results = self.searcher.search(
            query=query_text,
            top_k=top_k,
            entity_type=entity_type
        )
        # Tối giản hóa thông tin trả về cho Context Window của LLM
        compact_results = []
        for r in results:
            compact_results.append({
                "video_id": r["video_id"],
                "frame_idx": r["frame_idx"],
                "timestamp_str": r["timestamp_str"],
                "detected_text": r["text"],
                "confidence": r["confidence"],
                "bbox_norm": r["bbox_norm"],
                "entities": r["entities"],
                "submission_kis": r["submission_kis"],
                "submission_qa": r["submission_qa"]
            })
        return compact_results

    def tool_read_frame_text(self, video_id: str, frame_idx: int) -> Dict[str, Any]:
        """
        [TOOL 2] Đọc toàn bộ các đoạn văn bản có trong một khung hình cụ thể của video.
        
        Args:
            video_id: Mã video (ví dụ: 'L01_V001').
            frame_idx: Chỉ số khung hình (ví dụ: 150).
            
        Returns:
            Dictionary chứa danh sách tất cả các dòng chữ, vị trí bbox và các thực thể trích xuất được.
        """
        detections = self.searcher.get_frame_detections(video_id, frame_idx)
        all_texts = [d["text"] for d in detections]
        all_entities = []
        for d in detections:
            all_entities.extend(d.get("entities", []))

        return {
            "video_id": video_id,
            "frame_idx": frame_idx,
            "total_detections": len(detections),
            "full_transcription": " | ".join(all_texts),
            "detections": detections,
            "extracted_entities": all_entities
        }

    def tool_inspect_roi(self, video_id: str, frame_idx: int, roi: List[float]) -> Dict[str, Any]:
        """
        [TOOL 3] Phóng to và đọc văn bản trong một Vùng Quan Tâm (Region of Interest - ROI).
        
        Args:
            video_id: Mã video.
            frame_idx: Chỉ số khung hình.
            roi: Tọa độ ROI chuẩn hóa [ymin, xmin, ymax, xmax] trong khoảng [0.0, 1.0].
            
        Returns:
            Văn bản nằm trong vùng ROI yêu cầu.
        """
        detections = self.searcher.get_frame_detections(video_id, frame_idx)
        ymin_r, xmin_r, ymax_r, xmax_r = roi

        roi_texts = []
        for d in detections:
            ymin_b, xmin_b, ymax_b, xmax_b = d["bbox_norm"]
            # Kiểm tra tâm của bbox có nằm trong ROI không
            cy = (ymin_b + ymax_b) / 2.0
            cx = (xmin_b + xmax_b) / 2.0
            if ymin_r <= cy <= ymax_r and xmin_r <= cx <= xmax_r:
                roi_texts.append(d)

        combined_text = " ".join([d["text"] for d in roi_texts])
        return {
            "video_id": video_id,
            "frame_idx": frame_idx,
            "roi": roi,
            "matched_text": combined_text,
            "detections": roi_texts
        }

    def tool_extract_entities_in_video(self, video_id: str, entity_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        [TOOL 4] Thu thập tất cả các thực thể (Giá tiền, Biển số xe, SĐT...) xuất hiện trong toàn bộ video.
        Phục vụ trả lời câu hỏi Visual Q&A (ví dụ: 'Giá của tô phở là bao nhiêu?').
        """
        conn = self.searcher.get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT video_id, frame_idx, timestamp_str, text, entities, bbox_ymin, bbox_xmin, bbox_ymax, bbox_xmax
            FROM ocr_records
            WHERE video_id = ?
        """, (video_id,))
        rows = cursor.fetchall()
        conn.close()

        import json
        results = []
        for r in rows:
            ents = json.loads(r["entities"]) if r["entities"] else []
            for ent in ents:
                if entity_type is None or ent.get("type", "").upper() == entity_type.upper():
                    results.append({
                        "video_id": r["video_id"],
                        "frame_idx": r["frame_idx"],
                        "timestamp_str": r["timestamp_str"],
                        "raw_text": r["text"],
                        "entity_type": ent.get("type"),
                        "entity_value": ent.get("value"),
                        "bbox_norm": [r["bbox_ymin"], r["bbox_xmin"], r["bbox_ymax"], r["bbox_xmax"]]
                    })
        return results

    @staticmethod
    def get_tool_schemas() -> List[Dict[str, Any]]:
        """
        Trả về JSON Schema chuẩn Function Calling cho OpenAI / Gemini API.
        """
        return [
            {
                "name": "search_ocr_text",
                "description": "Tìm kiếm các khoảnh khắc video có chứa từ khóa văn bản, biển hiệu hoặc thông tin số.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query_text": {
                            "type": "string",
                            "description": "Từ khóa tìm kiếm (tiếng Việt có dấu hoặc không dấu)."
                        },
                        "top_k": {
                            "type": "integer",
                            "description": "Số lượng kết quả mong muốn (mặc định 20)."
                        },
                        "entity_type": {
                            "type": "string",
                            "enum": ["PRICE", "LICENSE_PLATE", "PHONE_NUMBER", "NUMBER", "TIME"],
                            "description": "Lọc theo loại thực thể nếu cần."
                        }
                    },
                    "required": ["query_text"]
                }
            },
            {
                "name": "read_frame_text",
                "description": "Đọc tất cả văn bản xuất hiện trong một khung hình cụ thể của video.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "video_id": {"type": "string", "description": "Mã video (ví dụ: 'L01_V001')"},
                        "frame_idx": {"type": "integer", "description": "Chỉ số frame trong video"}
                    },
                    "required": ["video_id", "frame_idx"]
                }
            },
            {
                "name": "inspect_roi_text",
                "description": "Đọc văn bản trong một vùng không gian cụ thể (ROI Bounding Box) của khung hình.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "video_id": {"type": "string", "description": "Mã video"},
                        "frame_idx": {"type": "integer", "description": "Chỉ số frame"},
                        "roi": {
                            "type": "array",
                            "items": {"type": "number"},
                            "description": "Tọa độ chuẩn hóa [ymin, xmin, ymax, xmax] từ 0.0 đến 1.0"
                        }
                    },
                    "required": ["video_id", "frame_idx", "roi"]
                }
            }
        ]
