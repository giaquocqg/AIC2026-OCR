# Review module AIC2026-OCR

Ngày review: 20/08/2026  
Repo được kiểm tra trên server: `/home/quoc/AIC2026-OCR`  
Commit: `504d791` — `feat: initial commit for AIC 2026 OCR & Spatial Text Retrieval Engine`

## 1. Kết luận

Module có cấu trúc tương đối đầy đủ cho pipeline OCR: tiền xử lý ảnh, phát hiện vùng chữ, nhận dạng, hậu xử lý tiếng Việt, temporal deduplication và xuất nhiều định dạng phục vụ tìm kiếm. Tuy nhiên, phiên bản hiện tại **chưa nên chạy toàn bộ 190.822 keyframe** vì còn một số lỗi có thể làm sai `frame_id`, mất kết quả khi resume, chạy OCR rỗng mà không báo lỗi và không tận dụng GPU như mô tả.

Khuyến nghị: sửa các lỗi mức P0/P1, chạy smoke test 100–500 keyframe trên máy có GPU, đánh giá chất lượng và tốc độ, sau đó mới chạy toàn bộ dataset.

## 2. Phạm vi và dữ liệu đã kiểm tra

- Repo `AIC2026-OCR` ở branch `main`, working tree sạch.
- Repo chính đang chạy: `/home/quoc/AIC2026`.
- Dataset hiện tại:
  - 873 video.
  - 190.822 keyframe WebP.
  - Ảnh keyframe 1280×720.
  - Có `index.json`, `fps.json` và `Youtube_URL.json`.
- Có 17 test case trong thư mục `tests/`.
- Chưa có kết quả OCR được sinh ra trên server tại thời điểm review.
- Server `quoc` không có GPU (`nvidia-smi` không tồn tại).
- Python hệ thống chưa cài các dependency OCR như OpenCV, NumPy, Pandas, PaddleOCR và VietOCR, nên chưa thể chạy test suite trong môi trường hiện tại.

## 3. Những phần đã làm tốt

### 3.1. Cấu trúc module rõ ràng

Repo đã tách các trách nhiệm chính:

- `src/ocr/preprocessor.py`: resize, blur score, CLAHE và ánh xạ metadata.
- `src/ocr/detector.py`: PaddleOCR/DBNet detection và perspective crop.
- `src/ocr/recognizer.py`: PaddleOCR và VietOCR recognition.
- `src/ocr/postprocessor.py`: chuẩn hóa Unicode/tiếng Việt và trích xuất entity.
- `src/ocr/deduplicator.py`: gộp text giữa các keyframe gần nhau.
- `src/ocr/indexer.py`: Parquet, SQLite FTS5, JSON và Elasticsearch bulk.
- `src/ocr/pipeline.py`: điều phối end-to-end.
- `src/ocr/agent_tool.py`: API tìm kiếm OCR cho agent.

### 3.2. Có hỗ trợ dataset hiện tại

Pipeline đã hỗ trợ:

- Ảnh `.webp`.
- Đọc `index.json` để giữ thứ tự keyframe.
- Đọc FPS thật theo từng video.
- Đọc URL YouTube.
- Xuất dữ liệu cho SQLite, Parquet, backend JSON và Elasticsearch.

### 3.3. Hậu xử lý có ích cho AIC

Module có:

- Chuẩn hóa Unicode NFC.
- Sinh text không dấu phục vụ tìm kiếm tiếng Việt.
- Regex nhận dạng giá tiền, biển số xe, số điện thoại, thời gian và con số.
- Lưu bounding box và polygon.

## 4. Các vấn đề cần sửa

### P0 — Làm mất `frame_id` sau temporal deduplication

**Vị trí liên quan:**

- `src/ocr/pipeline.py:150`
- `src/ocr/deduplicator.py:210-250`
- `src/ocr/indexer.py:47`
- `src/ocr/indexer.py:192`

`process_single_frame()` gắn ordinal `frame_id` vào detection, nhưng `TemporalDeduplicator` không mang trường này sang track và output cuối. `OCRIndexer` vì thế dùng fallback:

```python
f"{video_id}_{frame_idx}"
```

Kết quả có thể trở thành `L21_V001_75` thay vì ordinal ID `"1"`. Trong khi đó Milvus và backend đang dùng ID toàn cục dạng chuỗi `"0" ... "190821"`.

**Ảnh hưởng:** OCR không join đúng với embedding/Milvus; fusion, filter và rerank theo OCR có thể trả sai hoặc không tìm thấy frame.

**Khuyến nghị:**

- Giữ `frame_id` của detection đại diện trong temporal track.
- Khi detection tốt nhất thay đổi, cập nhật cả `frame_id`, `timestamp` và `youtube_url`.
- Thêm test end-to-end xác nhận output `ocr.json` dùng đúng ordinal ID từ `index.json`.

### P0 — Checkpoint có thể làm mất kết quả khi resume

**Vị trí:** `src/ocr/pipeline.py:227-287` và `src/ocr/pipeline.py:340-397`.

Pipeline chỉ checkpoint danh sách video đã hoàn thành. OCR records lại chỉ nằm trong biến RAM `all_dedup_records`. Nếu tiến trình dừng rồi chạy lại:

1. Các video cũ bị skip do đã nằm trong checkpoint.
2. Kết quả OCR cũ không được nạp lại.
3. File Parquet/SQLite/JSON cuối bị ghi lại chỉ bằng records của lần chạy mới.

**Ảnh hưởng:** có thể mất phần lớn kết quả sau một lần resume.

**Khuyến nghị:**

- Persist kết quả theo từng video bằng JSONL/Parquet shard ngay khi video hoàn tất.
- Chỉ đánh dấu checkpoint sau khi shard đã được ghi atomic thành công.
- Khi resume, nạp/merge toàn bộ shard đã hoàn thành trước khi export file cuối.
- Lưu fingerprint của `index.json`, config và model trong checkpoint để từ chối resume khi input thay đổi.

### P0 — Có thể chạy hết dataset nhưng output OCR rỗng

**Vị trí:**

- `src/ocr/detector.py:31-55`
- `src/ocr/detector.py:68-96`
- `src/ocr/recognizer.py:34-64`
- `src/ocr/recognizer.py:72-87`

Code bắt `Exception` rộng và không log nguyên nhân. Nếu PaddleOCR/VietOCR lỗi version, CUDA, model download hoặc API thay đổi, detector/recognizer có thể trở thành `None` hoặc trả chuỗi rỗng. Pipeline vẫn tiếp tục chạy và có thể hoàn thành với zero detections.

Trong `src/` và `scripts/` có 16 vị trí `except Exception` tại thời điểm review.

**Khuyến nghị:**

- Fail fast nếu detector và recognizer không khởi tạo được.
- Log exception kèm traceback, version model, device và ảnh/video đang xử lý.
- Đặt ngưỡng health check: nếu N frame liên tiếp không có detection hoặc model unavailable thì dừng job.
- Chỉ fallback ở các exception đã biết, không nuốt mọi lỗi.

### P1 — `--device` không thực sự đổi thiết bị của model

**Vị trí:** `scripts/run_indexing.py:58-62`.

`OCRPipeline` khởi tạo detector và recognizer trước. Sau đó CLI mới sửa:

```python
pipeline.device = args.device
pipeline.detector.device = args.device
pipeline.recognizer.device = args.device
```

Việc đổi attribute sau khởi tạo không reload model. Nếu config là `cuda` nhưng chạy `--device cpu`, model vẫn đã thử khởi tạo bằng CUDA trước đó.

**Khuyến nghị:** override config/device trước khi tạo `OCRPipeline`, hoặc truyền `device` trực tiếp vào constructor.

### P1 — `recognize_batch()` chưa batch thật

**Vị trí:** `src/ocr/recognizer.py:91-105`.

Hàm chia danh sách theo `batch_size`, nhưng bên trong vẫn gọi `recognize_single()` tuần tự cho từng crop. Vì vậy `batch_size=64` không tạo GPU batch thực sự.

**Ảnh hưởng:** tốc độ trên 190k keyframe có thể thấp hơn rất nhiều so với tài liệu mô tả.

**Khuyến nghị:** dùng API batch thực của PaddleOCR/VietOCR, gom crop theo kích thước/aspect ratio và benchmark throughput thực tế.

### P1 — “Hybrid” hiện chỉ là fallback

**Vị trí:** `src/ocr/recognizer.py:71-87`.

PaddleOCR luôn được ưu tiên. VietOCR chỉ chạy khi Paddle không trả kết quả hoặc phát sinh lỗi. Hai model không được chạy/so sánh theo loại nội dung hay confidence.

**Khuyến nghị:** định nghĩa chiến lược hybrid rõ ràng, ví dụ:

- Paddle cho số, mã, biển số và Latin.
- VietOCR cho text tiếng Việt dài.
- So sánh confidence/character plausibility.
- Chỉ chạy model thứ hai cho crop không chắc chắn để tiết kiệm thời gian.

### P1 — Dependency GPU và version chưa ổn định

**Vị trí:** `requirements.txt`.

- Đang dùng `paddlepaddle>=2.5.0`, không phải package/build GPU được khóa theo CUDA.
- Các dependency đều dùng `>=`, dễ tự nâng lên PaddleOCR API mới không tương thích code cũ.
- Detector có nhánh khởi tạo 2.x/3.x nhưng phương thức inference vẫn chủ yếu dùng API `.ocr(...)` kiểu cũ.

**Khuyến nghị:**

- Chốt Python, CUDA, cuDNN, PyTorch, PaddlePaddle và PaddleOCR bằng version cụ thể.
- Tạo Dockerfile GPU tái lập được.
- Thêm smoke test khởi tạo model và OCR một ảnh thật trong CI hoặc GPU runner.

### P1 — Chưa có đường ingest OCR đúng như README

README hướng dẫn:

```bash
python -m ingestion.cli ingest-ocr --ocr-file data/ocr_results/ocr.json --recreate
```

Nhưng repo backend hiện tại chỉ có các subcommand ingestion cho embedding, chưa có `ingest-ocr`.

**Ảnh hưởng:** sinh được `ocr.json` nhưng chưa có workflow chuẩn để nạp vào Elasticsearch `frames_ocr`.

**Khuyến nghị:** bổ sung job/CLI ingest OCR có:

- Validate schema và ordinal frame IDs.
- Tạo/recreate Elasticsearch mapping.
- Bulk insert theo chunk.
- Resume/idempotency.
- Verify document count và sample IDs sau ingest.

### P1 — Temporal gap không phù hợp khoảng cách keyframe

**Vị trí:**

- `configs/ocr_config.yaml:42`
- `src/ocr/deduplicator.py:162-168`

Config đặt `max_frame_gap: 15`, trong khi keyframe hiện tại có thể cách nhau tối đa 75 frame. Code so sánh raw `frame_idx`, nên nhiều detection liên tiếp thực tế sẽ không được merge.

**Khuyến nghị:** dùng một trong các cách:

- Gap theo giây (`timestamp`).
- Gap theo ordinal vị trí keyframe trong video.
- Hoặc đặt raw frame gap phù hợp FPS/keyframe policy, tối thiểu khoảng 75 nếu muốn merge các keyframe liền nhau.

### P2 — Dedup có thể giảm recall của backend

Temporal dedup chỉ giữ frame đại diện có confidence cao nhất. Điều này phù hợp để giảm trùng trong FTS, nhưng backend fusion/filter/rerank có thể cần OCR text ở mọi frame ứng viên.

**Khuyến nghị:** duy trì hai output:

- Raw per-frame OCR cho Elasticsearch/backend fusion.
- Deduplicated tracks cho UI/agent/FTS hoặc phân tích temporal.

### P2 — Đường dẫn mặc định chưa phù hợp server

`configs/ocr_config.yaml` dùng các đường dẫn dạng:

```text
AIC2026/data/...
```

Nếu chạy từ `/home/quoc/AIC2026-OCR`, chúng có thể resolve thành đường dẫn không tồn tại. Đường dẫn đúng trên server hiện là `../AIC2026/data/...` hoặc absolute path.

Nên dùng CLI argument/biến môi trường hoặc resolve path theo project root rõ ràng.

## 5. Tương thích với backend hiện tại

Backend hiện dùng:

- Milvus IDs dạng ordinal string.
- `KeyframeMap` đọc `index.json` và tính:

```text
timestamp = frame_idx / fps
```

- OCR search qua Elasticsearch index `frames_ocr`.
- OCR document cần join đúng `frame_id` với Milvus.

Vì vậy các điều kiện bắt buộc là:

1. OCR output giữ nguyên ordinal `frame_id`.
2. `frame_idx` là số frame trong video gốc.
3. Timestamp dùng FPS thật của đúng video.
4. Elasticsearch document `_id` dùng cùng ordinal `frame_id`.
5. Số frame không OCR được vẫn được xử lý rõ ràng; không được lệch thứ tự.

## 6. Kế hoạch sửa đề xuất

### Giai đoạn 1 — Sửa correctness

- Giữ `frame_id` qua dedup/export.
- Viết lại checkpoint theo shard có persistence.
- Fail fast và log lỗi model.
- Sửa `--device` trước khi model initialization.
- Thêm CLI ingest OCR cho backend.

### Giai đoạn 2 — Chuẩn hóa môi trường

- Chốt Docker image GPU và version dependency.
- Xác nhận CUDA/Paddle/PyTorch bằng health check.
- Cache model vào volume rõ ràng.
- Ghi model/config fingerprint vào output metadata.

### Giai đoạn 3 — Tối ưu tốc độ

- Batch detection/recognition thật.
- Tránh load hai PaddleOCR instance dư thừa nếu có thể dùng chung.
- Chạy theo batch ảnh/crop và prefetch I/O.
- Benchmark VRAM, images/s, crops/s và ETA.

### Giai đoạn 4 — Kiểm thử chất lượng

- Smoke test 100–500 keyframe đại diện.
- Kiểm tra text tiếng Việt, số, biển số, bảng giá, text nghiêng/nhỏ.
- Đo precision/recall OCR retrieval, không chỉ unit test code.
- So sánh raw per-frame và temporal dedup.

### Giai đoạn 5 — Chạy full dataset

- Chạy trên GPU server, không nên chạy full trên server `quoc` hiện tại vì không có GPU.
- Theo dõi log, GPU utilization, tốc độ và checkpoint.
- Sau khi hoàn tất, verify:
  - Số video đã xử lý.
  - Số frame đã đọc.
  - Tỷ lệ frame có text.
  - Số lỗi decode/model.
  - Ordinal ID sample đầu/giữa/cuối.
  - Elasticsearch document count.

## 7. Checklist trước khi cho phép chạy full

- [ ] `frame_id` ordinal không bị mất sau dedup.
- [ ] Resume không làm mất output cũ.
- [ ] Model init failure làm job dừng với lỗi rõ ràng.
- [ ] `--device` có hiệu lực trước model initialization.
- [ ] PaddleOCR/VietOCR chạy đúng GPU.
- [ ] Batch inference là batch thật.
- [ ] Version dependency được pin.
- [ ] Config path đúng trên Linux server.
- [ ] Có output raw per-frame cho backend.
- [ ] Có CLI ingest Elasticsearch hoạt động.
- [ ] Test contract với backend/Milvus ordinal IDs.
- [ ] Smoke test dữ liệu thật đạt chất lượng chấp nhận được.
- [ ] Ước tính thời gian và dung lượng output trước khi chạy full.

## 8. Lệnh chạy dự kiến sau khi đã sửa

Ví dụ đường dẫn trên server GPU:

```bash
python scripts/run_indexing.py \
  --keyframes_dir /home/quoc/AIC2026/data/keyframe/keyframe \
  --index_json /home/quoc/AIC2026/data/keyframe/keyframe/index.json \
  --fps_file /home/quoc/AIC2026/data/metadata/fps.json \
  --youtube_urls /home/quoc/AIC2026/data/metadata/Youtube_URL.json \
  --output_dir /home/quoc/AIC2026/data/ocr_results \
  --device cuda
```

Không nên dùng lệnh này cho full dataset cho đến khi các mục P0/P1 ở trên đã được xử lý và smoke test thành công.

