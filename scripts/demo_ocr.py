"""
Script: demo_ocr.py
Description: Quick interactive demo tool to test OCR recognition on any single image.
Usage:
    python scripts/demo_ocr.py --image path/to/frame.jpg
"""

import os
import sys
import time
import argparse
import cv2

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.ocr.pipeline import OCRPipeline


def main():
    parser = argparse.ArgumentParser(description="Quick Single Image OCR Demo")
    parser.add_argument("--image", type=str, required=True, help="Đường dẫn đến file ảnh cần test")
    parser.add_argument("--device", type=str, default="cpu", help="Thiết bị ('cpu' hoặc 'cuda')")
    parser.add_argument("--save_vis", action="store_true", help="Lưu ảnh vẽ bounding box kết quả")
    args = parser.parse_args()

    if not os.path.exists(args.image):
        print(f"❌ Không tìm thấy file ảnh: {args.image}")
        sys.exit(1)

    print("=" * 65)
    print("       AI CHALLENGE 2026 - SINGLE IMAGE OCR DEMO")
    print("=" * 65)
    print(f"🖼️ Input Image: {args.image}")
    print(f"⚙️ Target Device: {args.device}")
    print("=" * 65)

    pipeline = OCRPipeline(device=args.device)
    
    t0 = time.time()
    result = pipeline.process_single_frame(args.image, video_id="DEMO_VIDEO", image_name=os.path.basename(args.image))
    elapsed = time.time() - t0

    detections = result.get("detections", [])
    print(f"\n⚡ Thời gian nhận diện: {elapsed:.3f} giây")
    print(f"📝 Phát hiện được {len(detections)} vùng văn bản:\n")

    if not detections:
        print("⚠️ Không phát hiện văn bản nào trong ảnh này.")
        return

    header = f"{'STT':<4} | {'Độ tin cậy':<10} | {'Văn bản có dấu':<30} | {'Văn bản không dấu':<25}"
    print(header)
    print("-" * len(header))
    for i, d in enumerate(detections, 1):
        print(f"{i:<4} | {d['confidence']:<10.2f} | {d['text']:<30} | {d['text_unsigned']:<25}")

    if args.save_vis:
        img = cv2.imread(args.image)
        if img is not None:
            vis_img = pipeline.preprocessor.draw_detections(img, detections)
            out_vis = "demo_result.jpg"
            cv2.imwrite(out_vis, vis_img)
            print(f"\n🎨 Đã lưu ảnh kết quả trực quan tại: {out_vis}")

    print("=" * 65)


if __name__ == "__main__":
    main()
