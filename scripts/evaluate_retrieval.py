"""
Script: evaluate_retrieval.py
Description: Official AIC 2026 Evaluation & Benchmark Engine.
             Computes R@k (k in {1, 5, 20, 50, 100}), Final Score, and Query Latency
             for Textual KIS and Visual Q&A queries according to BTC regulations.
Usage:
    .venv\\Scripts\\python scripts/evaluate_retrieval.py --db data/ocr_results/ocr_fts.db
    .venv\\Scripts\\python scripts/evaluate_retrieval.py --db data/ocr_results/ocr_fts.db --benchmark data/sample_benchmark.json
"""

import os
import sys
import json
import time
import argparse
from typing import List, Dict, Any, Tuple

# Reconfigure stdout/stderr for Windows console UTF-8 support
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.ocr.indexer import OCRIndexSearcher
from src.ocr.postprocessor import VietnamesePostProcessor


# Default sample benchmark queries for quick validation
DEFAULT_BENCHMARK = [
    {
        "query_id": "Q01_KIS",
        "type": "KIS",
        "query_text": "Cơm tấm Ba Ghiền",
        "ground_truth": {
            "video_id": "L01_V001",
            "frame_range": [140, 160]
        }
    },
    {
        "query_id": "Q02_QA_PRICE",
        "type": "QA",
        "query_text": "quán phở thìn giá bao nhiêu",
        "ground_truth": {
            "video_id": "L01_V001",
            "frame_range": [490, 510],
            "answer": "60k"
        }
    },
    {
        "query_id": "Q03_KIS_PLATE",
        "type": "KIS",
        "query_text": "xe máy 59-X3",
        "ground_truth": {
            "video_id": "L02_V005",
            "frame_range": [870, 895]
        }
    },
    {
        "query_id": "Q04_QA_STORE",
        "type": "QA",
        "query_text": "cửa hàng tiện lợi circle k",
        "ground_truth": {
            "video_id": "L03_V010",
            "frame_range": [2440, 2460],
            "answer": "Circle K"
        }
    }
]


def calculate_r_score_kis(submission: Dict[str, Any], ground_truth: Dict[str, Any]) -> float:
    """
    Tính R-Score cho truy vấn Textual KIS:
    R-Score(r_i) = 1 nếu (v_i == GT_v AND frame_idx in [start, end]), ngược lại = 0.
    """
    target_video = ground_truth.get("video_id", "")
    start_f, end_f = ground_truth.get("frame_range", [0, 0])

    sub_video = submission.get("video_id", "")
    sub_frame = submission.get("frame_idx", -1)

    if sub_video == target_video and (start_f <= sub_frame <= end_f):
        return 1.0
    return 0.0


def calculate_r_score_qa(submission: Dict[str, Any], ground_truth: Dict[str, Any]) -> float:
    """
    Tính R-Score cho truy vấn Visual Q&A:
    R-Score(r_i) = 1 nếu đúng video, đúng frame_range, và answer khớp ngữ nghĩa.
    """
    kis_score = calculate_r_score_kis(submission, ground_truth)
    if kis_score < 1.0:
        return 0.0

    gt_answer = VietnamesePostProcessor.to_unsigned(ground_truth.get("answer", ""))
    
    # Kiểm tra answer trong text phát hiện được hoặc trong entities
    sub_text = VietnamesePostProcessor.to_unsigned(submission.get("text", ""))
    sub_entities = [VietnamesePostProcessor.to_unsigned(e.get("value", "")) for e in submission.get("entities", [])]

    if gt_answer in sub_text or any(gt_answer in e for e in sub_entities):
        return 1.0
    return 0.0


def evaluate_query(searcher: OCRIndexSearcher, item: Dict[str, Any]) -> Dict[str, Any]:
    """
    Chạy 1 truy vấn và tính toàn bộ các chỉ số R@k và Final Score theo chuẩn AIC 2026.
    """
    q_id = item.get("query_id", "Q")
    q_type = item.get("type", "KIS").upper()
    q_text = item.get("query_text", "")
    gt = item.get("ground_truth", {})

    t0 = time.time()
    results = searcher.search(query=q_text, top_k=100)
    latency_ms = (time.time() - t0) * 1000.0

    # Tính R-Score cho từng câu trả lời trong danh sách trả về
    r_scores = []
    for r in results:
        if q_type == "QA":
            score = calculate_r_score_qa(r, gt)
        else:
            score = calculate_r_score_kis(r, gt)
        r_scores.append(score)

    # Đảm bảo đủ 100 câu trả lời (điền 0.0 nếu ít hơn)
    while len(r_scores) < 100:
        r_scores.append(0.0)

    # Tính R@k = max(r_scores[:k]) với k in {1, 5, 20, 50, 100}
    r_at_k = {}
    for k in [1, 5, 20, 50, 100]:
        r_at_k[f"R@{k}"] = max(r_scores[:k]) if r_scores else 0.0

    # Final Score = 1/5 * sum(R@k)
    final_score = sum(r_at_k.values()) / 5.0

    # Tìm thứ hạng của câu trả lời đúng đầu tiên (Best Hit Rank)
    best_rank = None
    for idx, sc in enumerate(r_scores, 1):
        if sc > 0:
            best_rank = idx
            break

    return {
        "query_id": q_id,
        "type": q_type,
        "query_text": q_text,
        "target_video": gt.get("video_id", ""),
        "target_range": f"[{gt.get('frame_range', [0,0])[0]}-{gt.get('frame_range', [0,0])[1]}]",
        "best_rank": best_rank if best_rank is not None else "Not Found",
        "r_at_k": r_at_k,
        "final_score": round(final_score, 4),
        "latency_ms": round(latency_ms, 2),
        "total_candidates": len(results)
    }


def run_benchmark(db_path: str, benchmark_path: str = None):
    if not os.path.exists(db_path):
        print(f"[ERROR] Database FTS không tồn tại: {db_path}")
        return

    searcher = OCRIndexSearcher(db_path)

    if benchmark_path and os.path.exists(benchmark_path):
        with open(benchmark_path, "r", encoding="utf-8") as f:
            queries = json.load(f)
    else:
        queries = DEFAULT_BENCHMARK

    print("=" * 85)
    print("        AI CHALLENGE 2026 - OCR RETRIEVAL BENCHMARK & EVALUATION")
    print("=" * 85)
    print(f"Database Path:    {db_path}")
    print(f"Benchmark File:   {benchmark_path if benchmark_path else '[Built-in Test Set]'}")
    print(f"Total Queries:    {len(queries)}")
    print("-" * 85)

    eval_results = []
    for item in queries:
        res = evaluate_query(searcher, item)
        eval_results.append(res)

    # Hiển thị bảng chi tiết
    header = f"{'Q_ID':<12} | {'Type':<5} | {'Query Text':<24} | {'Target':<10} | {'Rank':<8} | {'Final Score':<11} | {'Latency':<8}"
    print(header)
    print("-" * len(header))

    total_final_score = 0.0
    total_latency = 0.0
    r_at_k_totals = {1: 0.0, 5: 0.0, 20: 0.0, 50: 0.0, 100: 0.0}

    for r in eval_results:
        q_text_disp = (r['query_text'][:21] + '...') if len(r['query_text']) > 21 else r['query_text']
        rank_disp = f"#{r['best_rank']}" if isinstance(r['best_rank'], int) else r['best_rank']
        print(f"{r['query_id']:<12} | {r['type']:<5} | {q_text_disp:<24} | {r['target_video']:<10} | {rank_disp:<8} | {r['final_score']:<11.4f} | {r['latency_ms']:<6.2f}ms")

        total_final_score += r["final_score"]
        total_latency += r["latency_ms"]
        for k in [1, 5, 20, 50, 100]:
            r_at_k_totals[k] += r["r_at_k"][f"R@{k}"]

    num_q = len(eval_results) if eval_results else 1
    mean_final_score = total_final_score / num_q
    mean_latency = total_latency / num_q

    print("=" * 85)
    print("                    TONG KET DIEM SO CHUAN BTC AIC 2026")
    print("=" * 85)
    print(f"▶ MEAN FINAL SCORE:  {mean_final_score:.4f} / 1.0000")
    print(f"▶ Average R@1:       {r_at_k_totals[1] / num_q:.4f}")
    print(f"▶ Average R@5:       {r_at_k_totals[5] / num_q:.4f}")
    print(f"▶ Average R@20:      {r_at_k_totals[20] / num_q:.4f}")
    print(f"▶ Average R@50:      {r_at_k_totals[50] / num_q:.4f}")
    print(f"▶ Average R@100:     {r_at_k_totals[100] / num_q:.4f}")
    print(f"▶ Avg Search Latency:{mean_latency:.2f} ms per query (< 5ms target)")
    print("=" * 85)


def main():
    parser = argparse.ArgumentParser(description="AI Challenge 2026 - OCR Evaluation Engine")
    parser.add_argument("--db", type=str, default="data/ocr_results/ocr_fts.db", help="Path to SQLite FTS5 database")
    parser.add_argument("--benchmark", type=str, default=None, help="Path to benchmark ground-truth JSON file")
    args = parser.parse_args()

    run_benchmark(db_path=args.db, benchmark_path=args.benchmark)


if __name__ == "__main__":
    main()
