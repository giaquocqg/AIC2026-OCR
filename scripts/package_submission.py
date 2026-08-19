"""
Script: package_submission.py
Description: CLI Tool to automatically package search results into an official AIC 2026 Submission ZIP.
Usage:
    # 1. Tự động truy vấn từ danh sách file query của BTC và đóng gói ZIP:
    .venv\\Scripts\\python scripts/package_submission.py --queries_dir data/round1_queries --db data/ocr_results/ocr_fts.db --output_zip submission_round1.zip

    # 2. Đóng gói từ thư mục chứa sẵn các file CSV:
    .venv\\Scripts\\python scripts/package_submission.py --csv_dir data/csv_results --output_zip submission_round1.zip
"""

import os
import sys
import glob
import argparse

# Reconfigure stdout/stderr for Windows console UTF-8 support
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.ocr.indexer import OCRIndexSearcher
from src.ocr.submission_packager import SubmissionFormatter, SubmissionPackager


def process_query_files(queries_dir: str, db_path: str, output_dir: str, zip_name: str) -> str:
    """Đọc các file query .txt do BTC cung cấp và tạo file ZIP nộp bài."""
    if not os.path.exists(db_path):
        print(f"❌ Lỗi: Database FTS5 không tồn tại tại '{db_path}'")
        sys.exit(1)

    searcher = OCRIndexSearcher(db_path)
    packager = SubmissionPackager(output_dir=output_dir)

    query_files = sorted(glob.glob(os.path.join(queries_dir, "*.txt")))
    if not query_files:
        print(f"⚠️ Không tìm thấy file query .txt nào trong thư mục '{queries_dir}'")
        return ""

    print(f"📂 Tìm thấy {len(query_files)} câu truy vấn trong '{queries_dir}'")
    csv_payloads = {}

    for qfile in query_files:
        base_name = os.path.splitext(os.path.basename(qfile))[0]
        csv_name = f"{base_name}.csv"

        with open(qfile, "r", encoding="utf-8") as f:
            query_text = f.read().strip()

        # Nhận diện loại truy vấn qua hậu tố tên file
        is_qa = "qa" in base_name.lower()
        is_trake = "trake" in base_name.lower()

        hits = searcher.search(query=query_text, top_k=100)

        rows = []
        for h in hits:
            vid = h["video_id"]
            fidx = h["frame_idx"]

            if is_qa:
                # Trích xuất thực thể hoặc văn bản ngắn nhất làm câu trả lời dự đoán
                ans = ""
                if h.get("entities"):
                    ans = h["entities"][0].get("value", "")
                if not ans:
                    ans = h["text"][:30]
                rows.append(SubmissionFormatter.format_qa_row(vid, fidx, ans))
            elif is_trake:
                # Mặc định lấy frame chính lặp lại hoặc chuỗi events
                rows.append(SubmissionFormatter.format_trake_row(vid, [fidx]))
            else:
                # Textual KIS
                rows.append(SubmissionFormatter.format_kis_row(vid, fidx))

        csv_payloads[csv_name] = rows
        print(f"   ✓ {base_name}: '{query_text[:30]}...' -> {len(rows)} predictions")

    zip_path = packager.package_zip(csv_payloads, zip_filename=zip_name, max_rows=100)
    return zip_path


def package_from_csv_dir(csv_dir: str, output_dir: str, zip_name: str) -> str:
    """Đóng gói trực tiếp từ các file CSV đã chuẩn bị sẵn."""
    csv_files = sorted(glob.glob(os.path.join(csv_dir, "*.csv")))
    if not csv_files:
        print(f"❌ Không tìm thấy file .csv nào trong '{csv_dir}'")
        sys.exit(1)

    packager = SubmissionPackager(output_dir=output_dir)
    payloads = {os.path.basename(f): f for f in csv_files}
    zip_path = packager.package_zip(payloads, zip_filename=zip_name, max_rows=100)
    return zip_path


def main():
    parser = argparse.ArgumentParser(description="AI Challenge 2026 - Official Submission Packager")
    parser.add_argument("--queries_dir", type=str, default=None, help="Thư mục chứa các file query .txt do BTC phát hành")
    parser.add_argument("--csv_dir", type=str, default=None, help="Thư mục chứa các file .csv kết quả đã tạo sẵn")
    parser.add_argument("--db", type=str, default="data/ocr_results/ocr_fts.db", help="Đường dẫn đến CSDL SQLite FTS5")
    parser.add_argument("--output_dir", type=str, default="data/submissions", help="Thư mục xuất file zip nộp bài")
    parser.add_argument("--output_zip", type=str, default="submission.zip", help="Tên file zip nộp bài (ví dụ: team_submission_r1.zip)")

    args = parser.parse_args()

    print("=" * 75)
    print("      AI CHALLENGE TP.HCM 2026 - OFFICIAL SUBMISSION PACKAGER")
    print("=" * 75)

    if args.queries_dir:
        zip_path = process_query_files(args.queries_dir, args.db, args.output_dir, args.output_zip)
    elif args.csv_dir:
        zip_path = package_from_csv_dir(args.csv_dir, args.output_dir, args.output_zip)
    else:
        print("❌ Vui lòng chỉ định một trong 2 tham số: --queries_dir hoặc --csv_dir")
        sys.exit(1)

    if zip_path and os.path.exists(zip_path):
        is_valid, errors = SubmissionPackager.validate_zip(zip_path)
        print("-" * 75)
        if is_valid:
            print("✅ XÁC THỰC THÀNH CÔNG! File ZIP hoàn toàn tuân thủ quy chế BTC:")
            print(f"📦 Đường dẫn file: {os.path.abspath(zip_path)}")
            print("👉 Bạn có thể nộp trực tiếp file này lên hệ thống sotuyenaic.oj.io.vn")
        else:
            print("⚠️ CẢNH BÁO KIỂM TRA ĐỊNH DẠNG:")
            for err in errors:
                print(f"   - {err}")
    print("=" * 75)


if __name__ == "__main__":
    main()
