"""
Module: submission_packager.py
Description: Official AIC 2026 Submission Packager & Format Validator.
             Automatically formats predictions into RFC 4180 compliant CSV files
             (Textual KIS, Visual Q&A, and TRAKE) and packages them into the
             mandatory `submission/` ZIP archive for the competition portal.
"""

import os
import io
import csv
import zipfile
import re
from typing import List, Dict, Any, Optional, Tuple, Union


class SubmissionFormatter:
    """Định dạng kết quả dự đoán thành các dòng CSV chuẩn quy chế AIC 2026."""

    @staticmethod
    def clean_video_id(video_id: str) -> str:
        """Loại bỏ phần mở rộng .mp4 nếu có (theo yêu cầu BTC)."""
        if not video_id:
            return ""
        vid = str(video_id).strip()
        if vid.lower().endswith(".mp4"):
            vid = vid[:-4]
        return vid

    @staticmethod
    def clean_frame_idx(frame_idx: Any) -> int:
        """Đảm bảo Frame ID là số nguyên chuẩn, không chứa khoảng trắng hay ký tự lạ."""
        if isinstance(frame_idx, int):
            return frame_idx
        try:
            # Lọc bỏ khoảng trắng thừa (ví dụ '25 300' -> 25300)
            cleaned = re.sub(r"\s+", "", str(frame_idx))
            return int(cleaned)
        except (ValueError, TypeError):
            return 0

    @staticmethod
    def clean_qa_answer(answer: str, max_chars: int = 100) -> str:
        """
        Chuẩn hóa câu trả lời Visual Q&A:
        - Giới hạn tối đa 100 ký tự (theo quy định BTC)
        - Giữ nguyên tiếng Việt/Anh
        """
        if not answer:
            return ""
        ans = str(answer).strip()
        if len(ans) > max_chars:
            ans = ans[:max_chars].rstrip()
        return ans

    @staticmethod
    def format_kis_row(video_id: str, frame_idx: Any) -> List[Any]:
        """Format dòng KIS: <video_name>, <frame_id>"""
        return [SubmissionFormatter.clean_video_id(video_id), SubmissionFormatter.clean_frame_idx(frame_idx)]

    @staticmethod
    def format_qa_row(video_id: str, frame_idx: Any, answer: str) -> List[Any]:
        """Format dòng Q&A: <video_name>, <frame_id>, <answer> (tối đa 100 ký tự)"""
        return [
            SubmissionFormatter.clean_video_id(video_id),
            SubmissionFormatter.clean_frame_idx(frame_idx),
            SubmissionFormatter.clean_qa_answer(answer, max_chars=100)
        ]

    @staticmethod
    def format_trake_row(video_id: str, frame_indices: List[Any]) -> List[Any]:
        """Format dòng TRAKE: <video_name>, <frame_1>, <frame_2>, ..., <frame_N>"""
        row = [SubmissionFormatter.clean_video_id(video_id)]
        for f in frame_indices:
            row.append(SubmissionFormatter.clean_frame_idx(f))
        return row


class SubmissionPackager:
    """
    Đóng gói toàn bộ file CSV vào thư mục `submission/` bên trong file `.zip`.
    Đảm bảo 100% tuân thủ yêu cầu của hệ thống chấm tự động sotuyenaic.oj.io.vn.
    """

    def __init__(self, output_dir: str = "data/submissions"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    @staticmethod
    def generate_csv_content(rows: List[List[Any]], max_rows: int = 100) -> str:
        """
        Sinh nội dung file CSV chuẩn:
        - Tối đa 100 dòng
        - Không có header row
        - Phân cách bằng dấu phẩy
        - Tự động bao quanh bằng dấu ngoặc kép và escape quote đúng chuẩn RFC 4180
        - Encoding UTF-8
        """
        limited_rows = rows[:max_rows]
        output = io.StringIO()
        writer = csv.writer(output, delimiter=",", quoting=csv.QUOTE_MINIMAL, lineterminator="\n")
        for r in limited_rows:
            writer.writerow(r)
        return output.getvalue()

    def save_query_csv(self, query_filename: str, rows: List[List[Any]], max_rows: int = 100) -> str:
        """Lưu một file CSV cho một truy vấn cụ thể."""
        if not query_filename.endswith(".csv"):
            query_filename += ".csv"

        content = self.generate_csv_content(rows, max_rows=max_rows)
        file_path = os.path.join(self.output_dir, query_filename)
        with open(file_path, "w", encoding="utf-8", newline="") as f:
            f.write(content)
        return file_path

    def package_zip(self, 
                    csv_files: Dict[str, Union[str, List[List[Any]]]], 
                    zip_filename: str = "submission.zip",
                    max_rows: int = 100) -> str:
        """
        Tạo file .zip hoàn chỉnh theo đúng cấu trúc yêu cầu của BTC:
        submission.zip
        └── submission/
            ├── query-1-kis.csv
            ├── query-2-kis.csv
            ├── query-3-qa.csv
            └── query-4-trake.csv
        
        Args:
            csv_files: Dictionary mapping { "query-1-kis.csv": rows_list hoặc file_path }
            zip_filename: Tên file zip xuất ra
        """
        if not zip_filename.endswith(".zip"):
            zip_filename += ".zip"

        zip_path = os.path.join(self.output_dir, zip_filename)

        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for fname, data in csv_files.items():
                clean_fname = os.path.basename(fname)
                if not clean_fname.endswith(".csv"):
                    clean_fname += ".csv"

                # Đường dẫn bắt buộc phải có tiền tố "submission/"
                arcname = f"submission/{clean_fname}"

                if isinstance(data, str) and os.path.exists(data):
                    # Đọc từ file có sẵn
                    with open(data, "r", encoding="utf-8") as f:
                        csv_text = f.read()
                    zf.writestr(arcname, csv_text.encode("utf-8"))
                elif isinstance(data, list):
                    # Sinh từ danh sách rows
                    csv_text = self.generate_csv_content(data, max_rows=max_rows)
                    zf.writestr(arcname, csv_text.encode("utf-8"))

        return zip_path

    @staticmethod
    def validate_zip(zip_path: str) -> Tuple[bool, List[str]]:
        """
        Kiểm tra tính hợp lệ của file ZIP trước khi nộp lên BTC:
        - Phải có thư mục `submission/`
        - Không có file rác hoặc header thừa
        - Các file CSV không quá 100 dòng
        """
        if not os.path.exists(zip_path):
            return False, [f"File ZIP không tồn tại: {zip_path}"]

        errors = []
        try:
            with zipfile.ZipFile(zip_path, "r") as zf:
                namelist = zf.namelist()
                if not any(name.startswith("submission/") and name.endswith(".csv") for name in namelist):
                    errors.append("Lỗi nghiêm trọng: File ZIP thiếu thư mục 'submission/' hoặc không chứa file .csv nào!")

                for name in namelist:
                    if not name.startswith("submission/"):
                        errors.append(f"Cảnh báo: File '{name}' nằm ngoài thư mục 'submission/'!")
                    
                    if name.endswith(".csv"):
                        content = zf.read(name).decode("utf-8", errors="replace")
                        lines = [line for line in content.strip().split("\n") if line.strip()]
                        if len(lines) > 100:
                            errors.append(f"Cảnh báo: File '{name}' có {len(lines)} dòng (vượt quá giới hạn 100 dòng của BTC)!")
        except Exception as e:
            return False, [f"Không thể đọc file ZIP: {e}"]

        is_valid = (len(errors) == 0)
        return is_valid, errors
