"""
Script: run_indexing.py
Description: CLI Tool to execute OCR processing & build search indexes for AI Challenge 2026 Keyframes.
Usage:
    python scripts/run_indexing.py --keyframes_dir D:/AICBaseline/KeyFrames_L01/keyframes --metadata_dir D:/AICBaseline/metadata
"""

import os
import sys
import argparse

# Reconfigure stdout/stderr for Windows console UTF-8 support
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.ocr.pipeline import OCRPipeline


def main():
    parser = argparse.ArgumentParser(description="AI Challenge 2026 - OCR Indexing Pipeline")
    parser.add_argument("--keyframes_dir", type=str, required=True, help="Đường dẫn đến thư mục chứa keyframes (ví dụ: AIC2026/data/keyframe/keyframe)")
    parser.add_argument("--index_json", type=str, default=None, help="Đường dẫn đến file index.json (190k keyframes mapping 0..N-1)")
    parser.add_argument("--fps_file", type=str, default=None, help="Đường dẫn đến file fps.json tập trung")
    parser.add_argument("--youtube_urls", type=str, default=None, help="Đường dẫn đến file Youtube_URL.json")
    parser.add_argument("--metadata_dir", type=str, default=None, help="Đường dẫn đến thư mục chứa các file metadata JSON (Lxx_Vxxx.json)")
    parser.add_argument("--config", type=str, default="configs/ocr_config.yaml", help="Đường dẫn đến file cấu hình YAML")
    parser.add_argument("--output_dir", type=str, default="data/ocr_results", help="Thư mục xuất kết quả (.parquet, .db, ocr.json)")
    parser.add_argument("--device", type=str, default=None, help="Thiết bị tính toán: 'cuda' hoặc 'cpu'")
    parser.add_argument("--no_checkpoint", action="store_true", help="Vô hiệu hóa tính năng checkpoint")

    args = parser.parse_args()

    if not os.path.exists(args.keyframes_dir):
        print(f"❌ Lỗi: Thư mục keyframes không tồn tại: {args.keyframes_dir}")
        sys.exit(1)

    print("=" * 65)
    print("      AI CHALLENGE TP.HCM 2026 - OCR RETRIEVAL ENGINE")
    print("=" * 65)
    print(f"📁 Keyframes Root: {args.keyframes_dir}")
    if args.index_json:
        print(f"📑 Index JSON:     {args.index_json}")
    if args.fps_file:
        print(f"⏱️ FPS File:       {args.fps_file}")
    if args.youtube_urls:
        print(f"🎥 YouTube URLs:   {args.youtube_urls}")
    if args.metadata_dir:
        print(f"📄 Metadata Root:  {args.metadata_dir}")
    print(f"⚙️ Config File:    {args.config}")
    print(f"💾 Output Dir:     {args.output_dir}")
    print("=" * 65)

    pipeline = OCRPipeline(config_input=args.config, device=args.device)


    if args.index_json and os.path.exists(args.index_json):
        summary = pipeline.process_from_index_json(
            index_json_path=args.index_json,
            keyframes_root=args.keyframes_dir,
            fps_file=args.fps_file,
            youtube_urls_file=args.youtube_urls,
            output_dir=args.output_dir,
            checkpoint=not args.no_checkpoint
        )
    else:
        summary = pipeline.process_dataset(
            keyframes_root=args.keyframes_dir,
            metadata_root=args.metadata_dir,
            fps_file=args.fps_file,
            youtube_urls_file=args.youtube_urls,
            output_dir=args.output_dir,
            checkpoint=not args.no_checkpoint
        )

    print("\n✅ Quá trình Indexing hoàn tất thành công!")


if __name__ == "__main__":
    main()

