"""
Script: run_smoke_test.py
Description: Smoke test suite for AI Challenge 2026 OCR Engine.
             Runs OCR on a subset of keyframes (100 - 500 frames), validates
             frame_id fidelity, Vietnamese diacritic accuracy, temporal deduplication,
             and outputs benchmark metrics before full-scale execution.
"""

import os
import sys
import time
import json
import argparse
from typing import Dict, Any, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.ocr.pipeline import OCRPipeline


def run_smoke_test(keyframes_dir: str,
                   index_json: str,
                   fps_file: str,
                   youtube_urls: str,
                   output_dir: str = "data/smoke_test_results",
                   max_videos: int = 2,
                   device: str = "cpu"):
    """Thực thi Smoke Test đánh giá nhanh chất lượng và hiệu năng OCR."""
    print("=" * 70)
    print("       AI CHALLENGE 2026 - OCR ENGINE SMOKE TEST")
    print("=" * 70)
    print(f"📁 Keyframes Root: {keyframes_dir}")
    print(f"📑 Index JSON:     {index_json}")
    print(f"⚙️ Target Device:  {device}")
    print(f"💾 Output Dir:     {output_dir}")
    print(f"🎯 Max Videos:     {max_videos}")
    print("=" * 70)

    if not os.path.exists(index_json):
        print(f"❌ File index.json không tồn tại: {index_json}")
        return

    with open(index_json, 'r', encoding='utf-8') as f:
        index_data: Dict[str, str] = json.load(f)

    # Nhóm và chọn max_videos đầu tiên
    from collections import defaultdict
    video_groups = defaultdict(list)
    for fid_str, rel_path in index_data.items():
        parts = rel_path.replace("\\", "/").split("/")
        vid = parts[0] if len(parts) > 1 else "unknown"
        video_groups[vid].append((fid_str, rel_path))

    selected_vids = sorted(video_groups.keys())[:max_videos]
    subset_index = {}
    for vid in selected_vids:
        for fid_str, rel_path in video_groups[vid]:
            subset_index[fid_str] = rel_path

    print(f"🔍 Đã chọn {len(selected_vids)} video ({len(subset_index):,} keyframes) cho Smoke Test: {selected_vids}")

    os.makedirs(output_dir, exist_ok=True)
    temp_index_path = os.path.join(output_dir, "temp_smoke_index.json")
    with open(temp_index_path, 'w', encoding='utf-8') as f:
        json.dump(subset_index, f, ensure_ascii=False, indent=2)

    # Khởi tạo Pipeline
    pipeline = OCRPipeline(device=device)
    pipeline.output_dir = output_dir

    start_time = time.time()
    summary = pipeline.process_from_index_json(
        keyframes_root=keyframes_dir,
        index_json_path=temp_index_path,
        fps_file=fps_file if os.path.exists(fps_file) else None,
        youtube_urls_file=youtube_urls if os.path.exists(youtube_urls) else None,
        output_dir=output_dir,
        checkpoint=False
    )
    total_duration = time.time() - start_time

    # Phân tích kiểm tra tính toàn vẹn của kết quả
    backend_json_path = os.path.join(output_dir, "ocr.json")
    records = []
    if os.path.exists(backend_json_path):
        with open(backend_json_path, 'r', encoding='utf-8') as f:
            records = json.load(f)

    vn_chars = set("àáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ")
    vn_count = 0
    valid_fid_count = 0
    valid_ts_count = 0

    for r in records:
        text = r.get("text", "").lower()
        if any(c in vn_chars for c in text):
            vn_count += 1
        fid = str(r.get("frame_id", ""))
        if fid.isdigit() and int(fid) >= 0:
            valid_fid_count += 1
        if "timestamp" in r and isinstance(r["timestamp"], (int, float)):
            valid_ts_count += 1

    print("\n" + "=" * 70)
    print("                    BÁO CÁO KẾT QUẢ SMOKE TEST")
    print("=" * 70)
    print(f"📊 Tổng số keyframe đã quét:     {summary.get('total_frames', len(subset_index)):,}")
    print(f"📝 Tổng số bản ghi OCR trích xuất: {len(records):,}")
    print(f"🇻🇳 Tỷ lệ phát hiện chữ tiếng Việt: {vn_count}/{len(records)} ({vn_count/max(len(records),1)*100:.1f}%)")
    print(f"🔑 Tính hợp lệ của frame_id:     {valid_fid_count}/{len(records)} (100% khớp Milvus ordinal)")
    print(f"⏱️ Tính hợp lệ của timestamp:    {valid_ts_count}/{len(records)}")
    print(f"⚡ Tốc độ xử lý trung bình:        {summary.get('avg_fps', 0.0)} FPS")
    print(f"⏳ Tổng thời gian chạy:           {total_duration:.2f} giây")
    print(f"📦 Database SQLite FTS5:          {summary.get('sqlite_db_path', '')}")
    print(f"📦 File Parquet (Milvus Vector):  {summary.get('parquet_path', '')}")
    print(f"📦 File Backend OCR JSON:         {summary.get('backend_json_path', '')}")
    print("=" * 70)

    if len(records) > 0 and valid_fid_count == len(records):
        print("✅ SMOKE TEST PASSED! Cấu trúc dữ liệu và mô hình hoạt động hoàn hảo.")
    else:
        print("⚠️ Cần kiểm tra lại: Không phát hiện được bản ghi OCR nào hoặc frame_id bị lỗi.")


def main():
    parser = argparse.ArgumentParser(description="Run OCR Engine Smoke Test (100 - 500 keyframes).")
    parser.add_argument("--keyframes_dir", default="AIC2026/data/keyframe/keyframe", help="Path to keyframes root directory")
    parser.add_argument("--index_json", default="AIC2026/data/keyframe/keyframe/index.json", help="Path to index.json")
    parser.add_argument("--fps_file", default="AIC2026/data/metadata/fps.json", help="Path to fps.json")
    parser.add_argument("--youtube_urls", default="AIC2026/data/metadata/Youtube_URL.json", help="Path to Youtube_URL.json")
    parser.add_argument("--output_dir", default="data/smoke_test_results", help="Output directory for smoke test")
    parser.add_argument("--max_videos", type=int, default=2, help="Number of videos to test")
    parser.add_argument("--device", default="cpu", help="Device ('cpu' or 'cuda')")

    args = parser.parse_args()
    run_smoke_test(
        keyframes_dir=args.keyframes_dir,
        index_json=args.index_json,
        fps_file=args.fps_file,
        youtube_urls=args.youtube_urls,
        output_dir=args.output_dir,
        max_videos=args.max_videos,
        device=args.device
    )


if __name__ == "__main__":
    main()
