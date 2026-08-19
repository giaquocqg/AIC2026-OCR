"""
Script: search_cli.py
Description: Interactive CLI Search tool for testing OCR queries and copying AIC 2026 submissions.
Usage:
    python scripts/search_cli.py --db data/ocr_results/ocr_fts.db
"""

import os
import sys
import time
import argparse

# Reconfigure stdout/stderr for Windows console UTF-8 support
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.ocr.indexer import OCRIndexSearcher


def print_banner():
    print("=" * 75)
    print("      AI CHALLENGE 2026 - OCR HIGH-SPEED SEARCH CLI")
    print("      Support: Textual KIS | Visual Q&A | TRAKE Alignments")
    print("=" * 75)
    print("💡 Mẹo: Gõ từ khóa có dấu hoặc không dấu. Gõ 'exit' hoặc 'quit' để thoát.")
    print("💡 Cú pháp nâng cao: 'query --entity PRICE' hoặc 'query --video L01_V001'")
    print("-" * 75)


def format_table(results):
    if not results:
        print("⚠️ Không tìm thấy kết quả nào phù hợp!")
        return

    print(f"\n🔍 Tìm thấy {len(results)} kết quả hàng đầu:\n")
    header = f"{'Top':<4} | {'FrameID':<8} | {'Video ID':<10} | {'Frame':<7} | {'Time':<10} | {'Conf':<6} | {'Detected Text':<26} | {'KIS Payload':<15}"
    print(header)
    print("-" * len(header))

    for idx, r in enumerate(results[:20], 1):
        text_disp = (r['text'][:23] + '...') if len(r['text']) > 23 else r['text']
        kis_payload = f"{r['video_id']}, {r['frame_idx']}"
        fid = str(r.get('frame_id', ''))
        print(f"{idx:<4} | {fid:<8} | {r['video_id']:<10} | {r['frame_idx']:<7} | {r['timestamp_str']:<10} | {r['confidence']:<6.2f} | {text_disp:<26} | {kis_payload:<15}")

    print("\n📋 GỢI Ý SUBMISSION (TOP 5 CHO BTC AIC 2026):")
    print("▶ [Textual KIS Format: <video_id>, <frame_idx>]")
    for idx, r in enumerate(results[:5], 1):
        yt_hint = f" ({r['youtube_url']})" if r.get('youtube_url') else ""
        print(f"   Top {idx}: {r['submission_kis']}{yt_hint}")

    print("\n▶ [Visual Q&A Format: <video_id>, <frame_idx>, <answer>]")
    for idx, r in enumerate(results[:5], 1):
        print(f"   Top {idx}: {r['submission_qa']}")
    print("-" * 75)



def main():
    parser = argparse.ArgumentParser(description="AI Challenge 2026 - OCR Search CLI")
    parser.add_argument("--db", type=str, default="data/ocr_results/ocr_fts.db", help="Đường dẫn đến file SQLite FTS5 db")
    args = parser.parse_args()

    if not os.path.exists(args.db):
        print(f"❌ Không tìm thấy database FTS: {args.db}")
        print("Vui lòng chạy 'python scripts/run_indexing.py' trước để tạo chỉ mục.")
        sys.exit(1)

    searcher = OCRIndexSearcher(args.db)
    print_banner()

    while True:
        try:
            raw_input = input("\n🔎 Nhập từ khóa tìm kiếm: ").strip()
            if not raw_input:
                continue
            if raw_input.lower() in ["exit", "quit", "q"]:
                print("👋 Tạm biệt!")
                break

            # Phân tích cú pháp bổ sung (ví dụ: "cơm tấm --video L01_V001")
            parts = raw_input.split("--")
            query = parts[0].strip()
            video_filter = None
            entity_filter = None

            for opt in parts[1:]:
                opt_str = opt.strip()
                if opt_str.startswith("video "):
                    video_filter = opt_str.replace("video ", "").strip()
                elif opt_str.startswith("entity "):
                    entity_filter = opt_str.replace("entity ", "").strip().upper()

            t0 = time.time()
            results = searcher.search(
                query=query, 
                top_k=50, 
                video_filter=video_filter, 
                entity_type=entity_filter
            )
            elapsed_ms = (time.time() - t0) * 1000

            print(f"⚡ Thời gian truy vấn: {elapsed_ms:.2f} ms")
            format_table(results)

        except KeyboardInterrupt:
            print("\n👋 Thoát chương trình!")
            break
        except Exception as e:
            print(f"❌ Lỗi: {e}")


if __name__ == "__main__":
    main()
