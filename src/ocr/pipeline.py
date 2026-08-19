"""
Module: pipeline.py
Description: End-to-End OCR Pipeline Orchestrator for AI Challenge 2026.
             Coordinates Preprocessing, DBNet++ Detection, Perspective Crop,
             Hybrid VietOCR/PaddleOCR Recognition, Vietnamese Entity Postprocessing,
             Temporal Deduplication, and SQLite FTS5 / Parquet Indexing.
"""

import os
import glob
import time
import json
import yaml
import cv2
from typing import Dict, List, Any, Optional, Union
from tqdm import tqdm

from .preprocessor import ImagePreprocessor, FrameMetadataMapper
from .detector import TextDetector
from .recognizer import BatchTextRecognizer
from .postprocessor import VietnamesePostProcessor
from .deduplicator import TemporalDeduplicator
from .indexer import OCRIndexer, OCRIndexSearcher


import logging

logger = logging.getLogger("src.ocr.pipeline")


class OCRPipeline:
    """Điều phối toàn bộ quy trình OCR & Text Indexing cho toàn bộ Dataset AIC 2026."""

    def __init__(self, config_input: Union[str, Dict[str, Any]] = "configs/ocr_config.yaml", device: Optional[str] = None):
        if isinstance(config_input, str):
            with open(config_input, 'r', encoding='utf-8') as f:
                self.config = yaml.safe_load(f)
        else:
            self.config = config_input

        # 1. Device configuration
        if device:
            self.device = device
        else:
            self.device = self.config.get("pipeline", {}).get("device", "cuda")

        import torch
        if self.device in ["cuda", "gpu"]:
            if torch.cuda.is_available():
                self.device = "cuda"
                try:
                    gpu_name = torch.cuda.get_device_name(0)
                    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                    print(f"🚀 [OCR Engine] Su dung GPU: {gpu_name} (VRAM: {vram_gb:.1f} GB)")
                except Exception:
                    print("🚀 [OCR Engine] Su dung GPU (CUDA)")
            else:
                logger.warning("CUDA không khả dụng trên môi trường hiện tại. Tự động chuyển sang CPU.")
                print("⚠️ [OCR Engine] CUDA không khả dụng trên môi trường hiện tại. Tự động chuyển sang CPU.")
                self.device = "cpu"
        else:
            self.device = "cpu"
            print("ℹ️ [OCR Engine] Thiết bị tính toán: CPU")

        # 2. Initialize Preprocessor & Metadata Mapper
        prep_cfg = self.config.get("preprocessor", {})
        self.preprocessor = ImagePreprocessor(
            max_dim=prep_cfg.get("max_image_dim", 1600),
            min_dim=prep_cfg.get("min_image_dim", 640),
            blur_threshold=prep_cfg.get("blur_laplacian_threshold", 40.0),
            enable_clahe=prep_cfg.get("enable_adaptive_clahe", True),
            clahe_clip_limit=prep_cfg.get("clahe_clip_limit", 2.0)
        )
        
        meta_cfg = self.config.get("metadata", {})
        self.frame_mapper = FrameMetadataMapper(
            metadata_dir=meta_cfg.get("metadata_dir", None),
            fps_file=meta_cfg.get("fps_file", None),
            index_file=meta_cfg.get("index_file", None),
            youtube_url_file=meta_cfg.get("youtube_url_file", None),
            default_fps=float(meta_cfg.get("default_fps", 25.0))
        )

        # 3. Initialize Detector
        det_cfg = self.config.get("detector", {})
        self.detector = TextDetector(
            engine=det_cfg.get("engine", "paddleocr"),
            use_angle_cls=det_cfg.get("use_angle_cls", True),
            det_db_thresh=det_cfg.get("det_db_thresh", 0.3),
            det_db_box_thresh=det_cfg.get("det_db_box_thresh", 0.5),
            det_db_unclip_ratio=det_cfg.get("det_db_unclip_ratio", 1.6),
            min_box_area=det_cfg.get("min_box_area", 80.0),
            device=self.device
        )

        # 4. Initialize Recognizer
        rec_cfg = self.config.get("recognizer", {})
        self.recognizer = BatchTextRecognizer(
            engine=rec_cfg.get("engine", "hybrid"),
            vietocr_model=rec_cfg.get("vietocr_model", "vgg_transformer"),
            paddle_lang=rec_cfg.get("paddle_rec_lang", "vi"),
            batch_size=rec_cfg.get("batch_size", 64),
            min_confidence=rec_cfg.get("min_confidence", 0.40),
            device=self.device
        )

        # 5. Initialize Postprocessor & Deduplicator
        dedup_cfg = self.config.get("deduplicator", {})
        self.deduplicator = TemporalDeduplicator(
            enable_temporal_merge=dedup_cfg.get("enable_temporal_merge", True),
            similarity_threshold=dedup_cfg.get("similarity_threshold", 0.82),
            max_frame_gap=dedup_cfg.get("max_frame_gap", 15)
        )

        # 6. Initialize Indexer
        storage_cfg = self.config.get("storage", {})
        self.output_dir = storage_cfg.get("output_dir", "data/ocr_results")
        self.indexer = OCRIndexer(output_dir=self.output_dir)


    def process_single_frame(self, 
                             image_input: Union[str, cv2.Mat], 
                             video_id: str = "sample_video", 
                             image_name: str = "0001.jpg") -> Dict[str, Any]:
        """Xử lý OCR cho 1 bức ảnh keyframe đơn lẻ."""
        if isinstance(image_input, str):
            if not os.path.exists(image_input):
                return {"error": f"File không tồn tại: {image_input}"}
            img_bgr = cv2.imread(image_input)
            image_name = os.path.basename(image_input)
        else:
            img_bgr = image_input

        if img_bgr is None or img_bgr.size == 0:
            return {"error": "Ảnh rỗng hoặc không thể đọc được"}

        # 1. Preprocessing
        prep_res = self.preprocessor.process(img_bgr)
        processed_img = prep_res["processed_image"]
        scale = prep_res["scale"]
        blur_score = prep_res["blur_score"]
        is_blurry = prep_res["is_blurry"]

        # 2. Frame mapping (trích xuất frame_id, frame_idx, timestamp_sec, timestamp_ms, timestamp_str, youtube_url)
        frame_info = self.frame_mapper.get_frame_info(video_id, image_name)
        frame_id = frame_info["frame_id"]
        frame_idx = frame_info["frame_idx"]
        timestamp_sec = frame_info["timestamp"]
        timestamp_ms = frame_info["timestamp_ms"]
        timestamp_str = frame_info["timestamp_str"]
        youtube_url = frame_info.get("youtube_url", "")

        # 3. Detection
        detections = self.detector.detect(processed_img)

        # 4. Recognition & Postprocessing
        final_detections = []
        if detections:
            crop_images = [d["crop_img"] for d in detections]
            rec_results = self.recognizer.recognize_batch(crop_images)

            for det, (raw_text, conf) in zip(detections, rec_results):
                if conf < self.recognizer.min_confidence or not raw_text.strip():
                    continue

                # Chuẩn hóa lại tọa độ polygon về kích thước ảnh gốc nếu có scale
                orig_poly = det["polygon"]
                if scale != 1.0:
                    orig_poly = [[int(pt[0] / scale), int(pt[1] / scale)] for pt in orig_poly]

                det_processed = VietnamesePostProcessor.process_detection(
                    raw_text=raw_text,
                    confidence=conf,
                    bbox=det["bbox_norm"],
                    polygon=orig_poly
                )
                det_processed["frame_id"] = frame_id
                det_processed["timestamp"] = timestamp_sec
                det_processed["youtube_url"] = youtube_url
                final_detections.append(det_processed)

        return {
            "frame_id": frame_id,
            "video_id": video_id,
            "image_name": image_name,
            "frame_idx": frame_idx,
            "timestamp": timestamp_sec,
            "timestamp_ms": timestamp_ms,
            "timestamp_str": timestamp_str,
            "youtube_url": youtube_url,
            "blur_score": blur_score,
            "is_blurry": is_blurry,
            "detections": final_detections
        }

    def process_video_keyframes(self, 
                                video_dir: str, 
                                video_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Xử lý toàn bộ keyframes của 1 video cụ thể.
        Hỗ trợ định dạng: .webp, .jpg, .jpeg, .png.
        Tự động gộp các text trùng lặp theo thời gian (Temporal Deduplication).
        """
        if not video_id:
            video_id = os.path.basename(video_dir.rstrip("/\\"))

        # Tìm tất cả file ảnh trong thư mục (bao gồm .webp)
        patterns = ["*.webp", "*.WEBP", "*.jpg", "*.JPG", "*.png", "*.PNG", "*.jpeg", "*.JPEG"]
        img_paths = []
        for p in patterns:
            img_paths.extend(glob.glob(os.path.join(video_dir, p)))
        
        # Sắp xếp theo frame_idx
        img_paths = sorted(list(set(img_paths)), key=lambda x: FrameMetadataMapper.extract_frame_idx(x))

        if not img_paths:
            return []

        frame_results = []
        for p in img_paths:
            res = self.process_single_frame(p, video_id=video_id, image_name=os.path.basename(p))
            if "detections" in res:
                frame_results.append(res)

        # Thực hiện gộp văn bản theo chuỗi thời gian
        dedup_records = self.deduplicator.deduplicate_video_detections(frame_results)
        return dedup_records

    def process_from_index_json(self,
                                index_json_path: str,
                                keyframes_root: str,
                                fps_file: Optional[str] = None,
                                youtube_urls_file: Optional[str] = None,
                                output_dir: Optional[str] = None,
                                checkpoint: bool = True) -> Dict[str, Any]:
        """
        Xử lý OCR trực tiếp từ file index.json (190.822 frames của AIC 2026).
        Đảm bảo thứ tự và frame_id 0..N-1 đồng bộ tuyệt đối với Milvus & CLIP vector store.
        """
        if output_dir:
            self.output_dir = output_dir
            self.indexer = OCRIndexer(output_dir=self.output_dir)

        if fps_file:
            self.frame_mapper._load_fps_map(fps_file)
        if youtube_urls_file:
            self.frame_mapper._load_youtube_map(youtube_urls_file)
        self.frame_mapper._load_index_file(index_json_path)

        with open(index_json_path, 'r', encoding='utf-8') as f:
            index_data: Dict[str, str] = json.load(f)

        os.makedirs(self.output_dir, exist_ok=True)
        checkpoint_vids_path = os.path.join(self.output_dir, ".index_checkpoint.json")
        checkpoint_records_path = os.path.join(self.output_dir, ".ocr_records_checkpoint.jsonl")

        completed_videos = set()
        all_dedup_records = []

        if checkpoint and os.path.exists(checkpoint_vids_path):
            try:
                with open(checkpoint_vids_path, 'r', encoding='utf-8') as f:
                    completed_videos = set(json.load(f))
            except Exception as e:
                logger.warning(f"Lỗi khi đọc danh sách completed videos từ checkpoint: {e}")
                completed_videos = set()

            # Nạp lại toàn bộ records đã trích xuất từ các video trước đó
            if os.path.exists(checkpoint_records_path):
                try:
                    with open(checkpoint_records_path, 'r', encoding='utf-8') as f:
                        for line in f:
                            line = line.strip()
                            if line:
                                all_dedup_records.append(json.loads(line))
                    print(f"🔄 [Checkpoint] Đã nạp lại {len(all_dedup_records):,} bản ghi từ {len(completed_videos)} video trước.")
                except Exception as e:
                    logger.error(f"Lỗi khi đọc file checkpoint records: {e}")

        # Nhóm theo video_id
        from collections import defaultdict
        video_groups = defaultdict(list)
        for fid_str, rel_path in index_data.items():
            parts = rel_path.replace("\\", "/").split("/")
            vid = parts[0] if len(parts) > 1 else "unknown"
            video_groups[vid].append((fid_str, rel_path))

        total_frames = 0
        start_time = time.time()

        print(f"[AIC 2026 OCR Engine] Xử lý {len(index_data):,} keyframes từ {len(video_groups)} video (index.json)")

        with tqdm(total=len(video_groups), desc="Processing Videos (index.json)", unit="video") as pbar:
            for vid, frame_list in sorted(video_groups.items()):
                if vid in completed_videos:
                    pbar.set_postfix_str(f"Skipped {vid} (Checkpoint)")
                    pbar.update(1)
                    continue

                frame_results = []
                for fid_str, rel_path in frame_list:
                    full_path = os.path.join(keyframes_root, rel_path)
                    res = self.process_single_frame(full_path, video_id=vid, image_name=os.path.basename(rel_path))
                    if "detections" in res:
                        frame_results.append(res)
                    total_frames += 1

                v_records = self.deduplicator.deduplicate_video_detections(frame_results)
                all_dedup_records.extend(v_records)

                # Lưu gia tăng ngay vào checkpoint JSONL để tránh mất dữ liệu
                if checkpoint:
                    completed_videos.add(vid)
                    try:
                        with open(checkpoint_records_path, 'a', encoding='utf-8') as f:
                            for r in v_records:
                                f.write(json.dumps(r, ensure_ascii=False) + "\n")
                        with open(checkpoint_vids_path, 'w', encoding='utf-8') as f:
                            json.dump(list(completed_videos), f, ensure_ascii=False)
                    except Exception as e:
                        logger.error(f"Lỗi khi lưu checkpoint cho video {vid}: {e}")

                elapsed = time.time() - start_time
                fps = total_frames / elapsed if elapsed > 0 else 0
                pbar.set_postfix_str(f"FPS: {fps:.1f} | Detections: {len(all_dedup_records):,}")
                pbar.update(1)

        # Xuất dữ liệu sang các định dạng: Parquet, SQLite FTS5, Backend JSON, Elasticsearch JSONL
        storage_cfg = self.config.get("storage", {})
        parquet_file = storage_cfg.get("parquet_filename", "ocr_index.parquet")
        sqlite_file = storage_cfg.get("sqlite_db_filename", "ocr_fts.db")

        parquet_path = self.indexer.export_parquet(all_dedup_records, filename=parquet_file)
        sqlite_path = self.indexer.build_sqlite_fts5(all_dedup_records, db_filename=sqlite_file)
        backend_json_path = self.indexer.export_backend_ocr_json(all_dedup_records, filename="ocr.json")
        es_bulk_path = self.indexer.export_elasticsearch_bulk(all_dedup_records, filename="frames_ocr_bulk.jsonl")

        total_time = time.time() - start_time
        summary = {
            "total_videos": len(video_groups),
            "processed_videos": len(completed_videos),
            "total_frames": total_frames,
            "total_ocr_records": len(all_dedup_records),
            "total_time_sec": round(total_time, 2),
            "avg_fps": round(total_frames / total_time, 2) if total_time > 0 else 0.0,
            "parquet_path": parquet_path,
            "sqlite_db_path": sqlite_path,
            "backend_json_path": backend_json_path,
            "es_bulk_path": es_bulk_path
        }

        print("\n" + "=" * 60)
        print("Hoàn thành Indexing OCR AIC 2026 từ index.json!")
        print(f"  - Tổng số bản ghi OCR: {len(all_dedup_records):,}")
        print(f"  - File Parquet (Vector Search / Milvus RAG): {parquet_path}")
        print(f"  - File SQLite FTS5 (High-Speed Search): {sqlite_path}")
        print(f"  - File Backend OCR JSON: {backend_json_path}")
        print(f"  - File Elasticsearch Bulk JSONL: {es_bulk_path}")
        print(f"  - Tốc độ trung bình: {summary['avg_fps']} FPS")
        print("=" * 60)

        return summary

    def process_dataset(self, 
                        keyframes_root: str, 
                        metadata_root: Optional[str] = None, 
                        fps_file: Optional[str] = None,
                        index_file: Optional[str] = None,
                        youtube_urls_file: Optional[str] = None,
                        output_dir: Optional[str] = None, 
                        checkpoint: bool = True) -> Dict[str, Any]:
        """
        Quét toàn bộ thư mục chứa các video keyframes để trích xuất OCR và build Index.
        Hỗ trợ checkpoint lưu tiến độ tự động khi chạy trên tập dữ liệu lớn.
        """
        if output_dir:
            self.output_dir = output_dir
            self.indexer = OCRIndexer(output_dir=self.output_dir)

        if metadata_root:
            self.frame_mapper.metadata_dir = metadata_root
        if fps_file:
            self.frame_mapper._load_fps_map(fps_file)
        if index_file:
            self.frame_mapper._load_index_file(index_file)
        if youtube_urls_file:
            self.frame_mapper._load_youtube_map(youtube_urls_file)

        os.makedirs(self.output_dir, exist_ok=True)
        checkpoint_vids_path = os.path.join(self.output_dir, ".checkpoint.json")
        checkpoint_records_path = os.path.join(self.output_dir, ".ocr_dataset_records_checkpoint.jsonl")

        completed_videos = set()
        all_dedup_records = []

        if checkpoint and os.path.exists(checkpoint_vids_path):
            try:
                with open(checkpoint_vids_path, 'r', encoding='utf-8') as f:
                    completed_videos = set(json.load(f))
            except Exception as e:
                logger.warning(f"Lỗi khi đọc checkpoint videos: {e}")
                completed_videos = set()

            if os.path.exists(checkpoint_records_path):
                try:
                    with open(checkpoint_records_path, 'r', encoding='utf-8') as f:
                        for line in f:
                            line = line.strip()
                            if line:
                                all_dedup_records.append(json.loads(line))
                    print(f"🔄 [Checkpoint] Đã nạp lại {len(all_dedup_records):,} bản ghi từ {len(completed_videos)} video trước.")
                except Exception as e:
                    logger.error(f"Lỗi khi nạp checkpoint dataset records: {e}")

        # Tìm tất cả thư mục con (mỗi thư mục tương ứng 1 video)
        video_dirs = [d for d in glob.glob(os.path.join(keyframes_root, "*")) if os.path.isdir(d)]
        video_dirs.sort()

        total_frames = 0
        start_time = time.time()

        print(f"[AIC 2026 OCR Engine] Bắt đầu xử lý {len(video_dirs)} video từ: {keyframes_root}")

        with tqdm(total=len(video_dirs), desc="Processing Videos", unit="video") as pbar:
            for vdir in video_dirs:
                vid = os.path.basename(vdir)
                if vid in completed_videos:
                    pbar.set_postfix_str(f"Skipped {vid} (Checkpoint)")
                    pbar.update(1)
                    continue

                v_records = self.process_video_keyframes(vdir, video_id=vid)
                all_dedup_records.extend(v_records)

                # Đếm số frames (.webp, .jpg, .png)
                img_set = set()
                for p in ["*.webp", "*.jpg", "*.png", "*.jpeg"]:
                    img_set.update(glob.glob(os.path.join(vdir, p)))
                total_frames += len(img_set)

                if checkpoint:
                    completed_videos.add(vid)
                    try:
                        with open(checkpoint_records_path, 'a', encoding='utf-8') as f:
                            for r in v_records:
                                f.write(json.dumps(r, ensure_ascii=False) + "\n")
                        with open(checkpoint_vids_path, 'w', encoding='utf-8') as f:
                            json.dump(list(completed_videos), f, ensure_ascii=False)
                    except Exception as e:
                        logger.error(f"Lỗi khi lưu checkpoint cho video {vid}: {e}")

                elapsed = time.time() - start_time
                fps = total_frames / elapsed if elapsed > 0 else 0
                pbar.set_postfix_str(f"FPS: {fps:.1f} | Detections: {len(all_dedup_records):,}")
                pbar.update(1)

        # Xuất dữ liệu sang Parquet, SQLite FTS5, Backend JSON, Elasticsearch JSONL
        storage_cfg = self.config.get("storage", {})
        parquet_file = storage_cfg.get("parquet_filename", "ocr_index.parquet")
        sqlite_file = storage_cfg.get("sqlite_db_filename", "ocr_fts.db")

        parquet_path = self.indexer.export_parquet(all_dedup_records, filename=parquet_file)
        sqlite_path = self.indexer.build_sqlite_fts5(all_dedup_records, db_filename=sqlite_file)
        backend_json_path = self.indexer.export_backend_ocr_json(all_dedup_records, filename="ocr.json")
        es_bulk_path = self.indexer.export_elasticsearch_bulk(all_dedup_records, filename="frames_ocr_bulk.jsonl")

        total_time = time.time() - start_time
        summary = {
            "total_videos": len(video_dirs),
            "processed_videos": len(completed_videos),
            "total_frames": total_frames,
            "total_ocr_records": len(all_dedup_records),
            "total_time_sec": round(total_time, 2),
            "avg_fps": round(total_frames / total_time, 2) if total_time > 0 else 0.0,
            "parquet_path": parquet_path,
            "sqlite_db_path": sqlite_path,
            "backend_json_path": backend_json_path,
            "es_bulk_path": es_bulk_path
        }

        print("\n" + "=" * 60)
        print("Hoàn thành Indexing OCR AI Challenge 2026!")
        print(f"  - Tổng số bản ghi OCR: {len(all_dedup_records):,}")
        print(f"  - File Parquet (Vector Search / RAG): {parquet_path}")
        print(f"  - File SQLite FTS5 (High-Speed Search): {sqlite_path}")
        print(f"  - File Backend OCR JSON: {backend_json_path}")
        print(f"  - File Elasticsearch Bulk JSONL: {es_bulk_path}")
        print(f"  - Tốc độ trung bình: {summary['avg_fps']} FPS")
        print("=" * 60)

        return summary


