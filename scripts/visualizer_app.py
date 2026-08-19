"""
Script: visualizer_app.py
Description: Interactive VBS (Video Browser Showdown) Visualizer & Verification Web App
             for AI Challenge 2026. Built with FastAPI + Modern Vanilla CSS/JS.
Usage:
    python scripts/visualizer_app.py --db data/ocr_results/ocr_fts.db --keyframes_dir D:/AICBaseline/KeyFrames_L01/keyframes
"""

import os
import sys
import json
import argparse
from typing import Optional, Dict, List, Any, Tuple


# Reconfigure stdout/stderr for Windows console UTF-8 support
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from fastapi import FastAPI, Query, HTTPException
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.ocr.indexer import OCRIndexSearcher

app = FastAPI(title="AIC 2026 OCR Visualizer", version="1.0.0")

# Global instances
SEARCHER: Optional[OCRIndexSearcher] = None
KEYFRAMES_DIR: Optional[str] = None
DB_PATH: str = "data/ocr_results/ocr_fts.db"
INDEX_MAP: Dict[str, str] = {}  # frame_id -> relative_path from index.json
VIDEO_FRAME_TO_PATH: Dict[tuple, str] = {}


def get_searcher():
    global SEARCHER
    if SEARCHER:
        return SEARCHER
    
    # 1. Thử load file DB
    if os.path.exists(DB_PATH) and os.path.getsize(DB_PATH) > 1024:
        try:
            SEARCHER = OCRIndexSearcher(DB_PATH)
            return SEARCHER
        except Exception:
            pass

    # 2. Nếu DB chưa build xong, nạp trực tiếp từ .ocr_records_checkpoint.jsonl
    checkpoint_jsonl = os.path.join(os.path.dirname(DB_PATH), ".ocr_records_checkpoint.jsonl")
    if os.path.exists(checkpoint_jsonl):
        try:
            records = []
            with open(checkpoint_jsonl, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        records.append(json.loads(line))
            if records:
                from src.ocr.indexer import OCRIndexer
                indexer = OCRIndexer(output_dir=os.path.dirname(DB_PATH))
                db_path = indexer.build_sqlite_fts5(records, db_filename=os.path.basename(DB_PATH))
                SEARCHER = OCRIndexSearcher(db_path)
                return SEARCHER
        except Exception as e:
            print(f"Error loading checkpoint records: {e}")

    return None


@app.get("/api/search")
async def api_search(
    q: str = Query(..., description="Query string"),
    top_k: int = Query(50, description="Top K results"),
    video: Optional[str] = Query(None, description="Filter by Video ID"),
    entity: Optional[str] = Query(None, description="Filter by entity type"),
    min_conf: float = Query(0.30, description="Minimum confidence threshold")
):
    searcher = get_searcher()
    if not searcher:
        raise HTTPException(status_code=500, detail="Searcher not initialized. No OCR records found.")
    
    results = searcher.search(
        query=q,
        top_k=top_k,
        video_filter=video if video else None,
        entity_type=entity if entity and entity != "ALL" else None,
        min_confidence=min_conf
    )
    return JSONResponse(content={"query": q, "total": len(results), "results": results})



@app.get("/api/frame_details")
async def api_frame_details(video_id: str, frame_idx: int):
    if not SEARCHER:
        raise HTTPException(status_code=500, detail="Searcher not initialized.")
    
    detections = SEARCHER.get_frame_detections(video_id, frame_idx)
    return JSONResponse(content={"video_id": video_id, "frame_idx": frame_idx, "detections": detections})


@app.get("/api/image")
async def api_image(video_id: str, frame_idx: int, frame_id: Optional[str] = None):
    """Phục vụ ảnh keyframe (.webp, .jpg, .png) trực tiếp từ thư mục keyframes hoặc index.json."""
    if not KEYFRAMES_DIR or not os.path.exists(KEYFRAMES_DIR):
        raise HTTPException(status_code=404, detail="Keyframes directory not configured")

    # 1. Tra cứu O(1) từ index_json nếu đã nạp
    if frame_id and frame_id in INDEX_MAP:
        img_path = os.path.join(KEYFRAMES_DIR, INDEX_MAP[frame_id])
        if os.path.exists(img_path):
            return FileResponse(img_path)

    if (video_id, frame_idx) in VIDEO_FRAME_TO_PATH:
        img_path = os.path.join(KEYFRAMES_DIR, VIDEO_FRAME_TO_PATH[(video_id, frame_idx)])
        if os.path.exists(img_path):
            return FileResponse(img_path)

    # 2. Thử tìm file theo các quy ước tên phổ biến (bao gồm .webp)
    possible_names = [
        f"keyframe_{frame_idx}.webp",
        f"keyframe_{frame_idx}.jpg",
        f"keyframe_{frame_idx}.png",
        f"{frame_idx:04d}.webp",
        f"{frame_idx:04d}.jpg",
        f"{frame_idx:05d}.jpg",
        f"{frame_idx:06d}.jpg",
        f"{frame_idx}.webp",
        f"{frame_idx}.jpg",
        f"{frame_idx:04d}.png",
        f"{frame_idx}.png"
    ]

    video_dir = os.path.join(KEYFRAMES_DIR, video_id)
    if os.path.exists(video_dir):
        for name in possible_names:
            img_path = os.path.join(video_dir, name)
            if os.path.exists(img_path):
                return FileResponse(img_path)

    # 3. Tìm kiếm đệ quy nông
    import glob
    pattern = os.path.join(KEYFRAMES_DIR, f"**/{video_id}/*{frame_idx}*.*")
    matches = glob.glob(pattern, recursive=True)
    if matches:
        return FileResponse(matches[0])

    raise HTTPException(status_code=404, detail=f"Image not found for {video_id} frame {frame_idx}")



@app.get("/", response_class=HTMLResponse)
async def index_page():
    """Giao diện Web VBS Hiện đại, Sang trọng và Trực quan."""
    html_content = """
    <!DOCTYPE html>
    <html lang="vi">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>AIC 2026 - OCR Video Retrieval & VBS Studio</title>
        <link rel="preconnect" href="https://fonts.googleapis.com">
        <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
        <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
        <style>
            :root {
                --bg-main: #0B0F19;
                --bg-card: #131B2E;
                --bg-glass: rgba(19, 27, 46, 0.85);
                --border-color: rgba(255, 255, 255, 0.08);
                --primary: #6366F1;
                --primary-glow: rgba(99, 102, 241, 0.35);
                --accent: #06B6D4;
                --success: #10B981;
                --warning: #F59E0B;
                --text-main: #F3F4F6;
                --text-muted: #9CA3AF;
                --font-sans: 'Outfit', sans-serif;
                --font-mono: 'JetBrains Mono', monospace;
            }

            * {
                box-sizing: border-box;
                margin: 0;
                padding: 0;
            }

            body {
                background-color: var(--bg-main);
                color: var(--text-main);
                font-family: var(--font-sans);
                min-height: 100vh;
                display: flex;
                flex-direction: column;
                background-image: 
                    radial-gradient(circle at 15% 15%, rgba(99, 102, 241, 0.12) 0%, transparent 40%),
                    radial-gradient(circle at 85% 85%, rgba(6, 182, 212, 0.10) 0%, transparent 40%);
            }

            /* Header */
            header {
                backdrop-filter: blur(12px);
                background: var(--bg-glass);
                border-bottom: 1px solid var(--border-color);
                position: sticky;
                top: 0;
                z-index: 100;
                padding: 1rem 2rem;
                display: flex;
                justify-content: space-between;
                align-items: center;
            }

            .logo-group {
                display: flex;
                align-items: center;
                gap: 0.75rem;
            }

            .logo-badge {
                background: linear-gradient(135deg, var(--primary), var(--accent));
                color: white;
                font-weight: 700;
                font-size: 0.9rem;
                padding: 0.35rem 0.75rem;
                border-radius: 8px;
                box-shadow: 0 0 15px var(--primary-glow);
            }

            .logo-title {
                font-size: 1.25rem;
                font-weight: 700;
                letter-spacing: -0.5px;
            }

            .badge-status {
                background: rgba(16, 185, 129, 0.15);
                color: var(--success);
                border: 1px solid rgba(16, 185, 129, 0.3);
                padding: 0.25rem 0.6rem;
                border-radius: 20px;
                font-size: 0.75rem;
                font-weight: 600;
            }

            /* Container */
            .main-content {
                flex: 1;
                max-width: 1600px;
                width: 100%;
                margin: 0 auto;
                padding: 2rem;
                display: flex;
                flex-direction: column;
                gap: 1.5rem;
            }

            /* Search Panel */
            .search-box-card {
                background: var(--bg-card);
                border: 1px solid var(--border-color);
                border-radius: 16px;
                padding: 1.5rem;
                box-shadow: 0 8px 30px rgba(0, 0, 0, 0.3);
            }

            .search-row {
                display: flex;
                gap: 1rem;
                align-items: center;
            }

            .input-wrapper {
                flex: 1;
                position: relative;
            }

            .search-input {
                width: 100%;
                background: #0D1322;
                border: 1px solid var(--border-color);
                color: var(--text-main);
                font-family: var(--font-sans);
                font-size: 1.1rem;
                padding: 0.9rem 1.2rem;
                border-radius: 12px;
                outline: none;
                transition: all 0.2s ease;
            }

            .search-input:focus {
                border-color: var(--primary);
                box-shadow: 0 0 0 3px var(--primary-glow);
            }

            .btn-search {
                background: linear-gradient(135deg, var(--primary), #4F46E5);
                color: white;
                border: none;
                font-family: var(--font-sans);
                font-weight: 600;
                font-size: 1rem;
                padding: 0.9rem 2rem;
                border-radius: 12px;
                cursor: pointer;
                transition: all 0.2s ease;
                display: flex;
                align-items: center;
                gap: 0.5rem;
            }

            .btn-search:hover {
                transform: translateY(-2px);
                box-shadow: 0 5px 20px var(--primary-glow);
            }

            /* Filter chips */
            .filters-row {
                display: flex;
                gap: 0.75rem;
                margin-top: 1rem;
                align-items: center;
                flex-wrap: wrap;
            }

            .filter-label {
                font-size: 0.85rem;
                color: var(--text-muted);
                font-weight: 500;
            }

            .chip {
                background: rgba(255, 255, 255, 0.05);
                border: 1px solid var(--border-color);
                color: var(--text-muted);
                padding: 0.35rem 0.8rem;
                border-radius: 20px;
                font-size: 0.8rem;
                font-weight: 500;
                cursor: pointer;
                transition: all 0.2s ease;
            }

            .chip:hover, .chip.active {
                background: var(--primary);
                color: white;
                border-color: var(--primary);
                box-shadow: 0 0 10px var(--primary-glow);
            }

            /* Metrics & Info Bar */
            .metrics-bar {
                display: flex;
                justify-content: space-between;
                align-items: center;
                font-size: 0.9rem;
                color: var(--text-muted);
            }

            .btn-copy-all {
                background: rgba(255, 255, 255, 0.05);
                border: 1px solid var(--border-color);
                color: var(--text-main);
                padding: 0.4rem 0.8rem;
                border-radius: 8px;
                font-size: 0.8rem;
                cursor: pointer;
                transition: 0.2s;
            }
            .btn-copy-all:hover {
                background: rgba(255, 255, 255, 0.15);
            }

            /* Results Grid */
            .results-grid {
                display: grid;
                grid-template-columns: repeat(auto-fill, minmax(320px, 1fr));
                gap: 1.5rem;
            }

            .card {
                background: var(--bg-card);
                border: 1px solid var(--border-color);
                border-radius: 14px;
                overflow: hidden;
                display: flex;
                flex-direction: column;
                transition: transform 0.2s ease, border-color 0.2s ease, box-shadow 0.2s ease;
            }

            .card:hover {
                transform: translateY(-4px);
                border-color: rgba(99, 102, 241, 0.4);
                box-shadow: 0 10px 25px rgba(0, 0, 0, 0.4);
            }

            .image-container {
                position: relative;
                width: 100%;
                height: 190px;
                background: #090D16;
                display: flex;
                align-items: center;
                justify-content: center;
            }

            .keyframe-img {
                width: 100%;
                height: 100%;
                object-fit: cover;
            }

            .img-overlay-badge {
                position: absolute;
                top: 8px;
                left: 8px;
                background: rgba(0, 0, 0, 0.7);
                backdrop-filter: blur(4px);
                color: white;
                font-family: var(--font-mono);
                font-size: 0.75rem;
                padding: 0.25rem 0.5rem;
                border-radius: 6px;
                border: 1px solid rgba(255, 255, 255, 0.1);
            }

            .card-body {
                padding: 1rem;
                display: flex;
                flex-direction: column;
                gap: 0.6rem;
                flex: 1;
            }

            .ocr-text-box {
                background: rgba(0, 0, 0, 0.3);
                border-left: 3px solid var(--primary);
                padding: 0.5rem 0.75rem;
                border-radius: 0 8px 8px 0;
            }

            .ocr-text {
                font-size: 0.95rem;
                font-weight: 600;
                color: #FFFFFF;
                word-break: break-word;
            }

            .entity-badge {
                display: inline-block;
                background: rgba(6, 182, 212, 0.15);
                color: var(--accent);
                border: 1px solid rgba(6, 182, 212, 0.3);
                padding: 0.15rem 0.5rem;
                border-radius: 4px;
                font-size: 0.7rem;
                font-weight: 600;
                margin-top: 0.25rem;
            }

            .card-meta {
                display: flex;
                justify-content: space-between;
                align-items: center;
                font-size: 0.75rem;
                color: var(--text-muted);
                font-family: var(--font-mono);
            }

            .submission-actions {
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 0.5rem;
                margin-top: auto;
                padding-top: 0.5rem;
                border-top: 1px solid var(--border-color);
            }

            .btn-sub {
                background: rgba(255, 255, 255, 0.05);
                border: 1px solid var(--border-color);
                color: var(--text-main);
                font-family: var(--font-sans);
                font-size: 0.75rem;
                font-weight: 600;
                padding: 0.4rem;
                border-radius: 6px;
                cursor: pointer;
                transition: all 0.15s ease;
                text-align: center;
            }

            .btn-sub:hover {
                background: var(--primary);
                color: white;
                border-color: var(--primary);
            }

            .btn-sub.copied {
                background: var(--success);
                color: white;
                border-color: var(--success);
            }

            /* Toast Notification */
            .toast {
                position: fixed;
                bottom: 2rem;
                right: 2rem;
                background: #1E293B;
                color: white;
                border: 1px solid var(--primary);
                padding: 0.75rem 1.25rem;
                border-radius: 10px;
                box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5);
                font-size: 0.85rem;
                font-weight: 500;
                transform: translateY(100px);
                opacity: 0;
                transition: all 0.3s cubic-bezier(0.16, 1, 0.3, 1);
                z-index: 999;
            }

            .toast.show {
                transform: translateY(0);
                opacity: 1;
            }
        </style>
    </head>
    <body>
        <header>
            <div class="logo-group">
                <span class="logo-badge">AIC 2026</span>
                <span class="logo-title">OCR Video Retrieval & VBS Studio</span>
            </div>
            <div style="display: flex; gap: 1rem; align-items: center;">
                <span class="badge-status">● SQLite FTS5 Engine Active</span>
            </div>
        </header>

        <main class="main-content">
            <!-- Search Panel -->
            <div class="search-box-card">
                <div class="search-row">
                    <div class="input-wrapper">
                        <input type="text" id="queryInput" class="search-input" placeholder="🔍 Nhập từ khóa OCR (ví dụ: 'cơm tấm', 'ba ghiền', '59-X3', '45k', 'highlands')..." autofocus>
                    </div>
                    <button class="btn-search" id="searchBtn" onclick="doSearch()">
                        <span>Truy Vấn</span>
                    </button>
                </div>

                <div class="filters-row">
                    <span class="filter-label">Thực thể (Entities):</span>
                    <span class="chip active" onclick="setEntityFilter('ALL', this)">Tất cả</span>
                    <span class="chip" onclick="setEntityFilter('PRICE', this)">💰 Giá tiền</span>
                    <span class="chip" onclick="setEntityFilter('LICENSE_PLATE', this)">🚗 Biển số xe</span>
                    <span class="chip" onclick="setEntityFilter('PHONE_NUMBER', this)">📞 Số điện thoại</span>
                    <span class="chip" onclick="setEntityFilter('NUMBER', this)">🔢 Con số</span>
                    <span class="chip" onclick="setEntityFilter('TIME', this)">⏰ Thời gian</span>
                </div>
            </div>

            <!-- Metrics bar -->
            <div class="metrics-bar">
                <span id="resultsCount">Nhập từ khóa để bắt đầu tìm kiếm trong CSDL...</span>
                <div style="display: flex; gap: 0.5rem;">
                    <button class="btn-copy-all" onclick="copyTopKISPayload()">📋 Copy KIS</button>
                    <button class="btn-copy-all" style="background: rgba(34, 197, 94, 0.15); color: #86EFAC; border-color: rgba(34, 197, 94, 0.3);" onclick="downloadCSV('kis')">📥 CSV KIS</button>
                    <button class="btn-copy-all" style="background: rgba(168, 85, 247, 0.15); color: #D8B4FE; border-color: rgba(168, 85, 247, 0.3);" onclick="downloadCSV('qa')">📥 CSV Q&A</button>
                </div>
            </div>

            <!-- Results Grid -->
            <div class="results-grid" id="resultsGrid"></div>
        </main>

        <div id="toast" class="toast">Đã copy payload nộp bài vào Clipboard!</div>

        <script>
            let currentResults = [];
            let currentEntity = "ALL";

            document.getElementById('queryInput').addEventListener('keypress', function (e) {
                if (e.key === 'Enter') {
                    doSearch();
                }
            });

            function setEntityFilter(entityType, elem) {
                currentEntity = entityType;
                document.querySelectorAll('.chip').forEach(c => c.classList.remove('active'));
                elem.classList.add('active');
                doSearch();
            }

            async function doSearch() {
                const q = document.getElementById('queryInput').value.trim();
                if (!q) return;

                const grid = document.getElementById('resultsGrid');
                const countElem = document.getElementById('resultsCount');
                countElem.innerText = "Đang truy vấn SQLite FTS5...";
                grid.innerHTML = "";

                try {
                    const res = await fetch(`/api/search?q=${encodeURIComponent(q)}&entity=${currentEntity}&top_k=100`);
                    const data = await res.json();
                    currentResults = data.results || [];

                    countElem.innerText = `Tìm thấy ${currentResults.length} kết quả cho "${q}" (< 5ms)`;

                    if (currentResults.length === 0) {
                        grid.innerHTML = `<div style="grid-column: 1/-1; text-align: center; padding: 3rem; color: var(--text-muted);">Không tìm thấy frame nào khớp với từ khóa.</div>`;
                        return;
                    }

                    renderCards(currentResults);
                } catch (err) {
                    countElem.innerText = "Lỗi truy vấn: " + err.message;
                }
            }

            function renderCards(results) {
                const grid = document.getElementById('resultsGrid');
                grid.innerHTML = "";

                results.forEach((r, idx) => {
                    const card = document.createElement('div');
                    card.className = "card";

                    const entitiesHtml = (r.entities && r.entities.length > 0)
                        ? r.entities.map(e => `<span class="entity-badge">${e.type}: ${e.value}</span>`).join(' ')
                        : '';

                    const ytButton = r.youtube_url 
                        ? `<a href="${r.youtube_url}" target="_blank" class="btn-sub" style="text-decoration:none;display:inline-flex;align-items:center;justify-content:center;background:rgba(239,68,68,0.2);color:#FCA5A5;border:1px solid rgba(239,68,68,0.4);">▶ YouTube</a>`
                        : '';

                    card.innerHTML = `
                        <div class="image-container">
                            <img src="/api/image?video_id=${r.video_id}&frame_idx=${r.frame_idx}&frame_id=${encodeURIComponent(r.frame_id || '')}" 
                                 class="keyframe-img" 
                                 alt="${r.video_id} ${r.frame_idx}"
                                 onerror="this.onerror=null; this.style.display='none'; this.parentNode.innerHTML='<div style=\\'color:#64748B;font-size:0.8rem;text-align:center;padding:1rem;\\'>[Keyframe ${r.video_id} - #${r.frame_idx}]</div>';">
                            <div class="img-overlay-badge">#${idx + 1} | ID:${r.frame_id || r.video_id}</div>
                        </div>
                        <div class="card-body">
                            <div class="ocr-text-box">
                                <div class="ocr-text">${escapeHtml(r.text)}</div>
                                ${entitiesHtml}
                            </div>
                            <div class="card-meta">
                                <span>Video: ${r.video_id}</span>
                                <span>Frame: ${r.frame_idx}</span>
                                <span>Time: ${r.timestamp_str}</span>
                                <span>Conf: ${(r.confidence * 100).toFixed(0)}%</span>
                            </div>
                            <div class="submission-actions">
                                <button class="btn-sub" onclick="copyText('${r.submission_kis}', this)">Copy KIS</button>
                                <button class="btn-sub" onclick="copyText('${r.submission_qa}', this)">Copy Q&A</button>
                                ${ytButton}
                            </div>
                        </div>
                    `;
                    grid.appendChild(card);
                });
            }

            function copyText(text, btn) {
                navigator.clipboard.writeText(text).then(() => {
                    const orig = btn.innerText;
                    btn.innerText = "✓ Copied";
                    btn.classList.add('copied');
                    showToast(`Đã copy: ${text}`);
                    setTimeout(() => {
                        btn.innerText = orig;
                        btn.classList.remove('copied');
                    }, 1500);
                });
            }

            function copyTopKISPayload() {
                if (!currentResults || currentResults.length === 0) {
                    showToast("Chưa có kết quả để copy!");
                    return;
                }
                const lines = currentResults.slice(0, 100).map(r => r.submission_kis).join('\n');
                navigator.clipboard.writeText(lines).then(() => {
                    showToast(`Đã copy toàn bộ ${Math.min(currentResults.length, 100)} dòng KIS payload!`);
                });
            }

            function downloadCSV(type) {
                if (!currentResults || currentResults.length === 0) {
                    showToast("Chưa có kết quả để tải CSV!");
                    return;
                }
                const q = document.getElementById('queryInput').value.trim() || 'query';
                const rows = currentResults.slice(0, 100);
                let csvContent = "";

                if (type === 'qa') {
                    csvContent = rows.map(r => {
                        let ans = "";
                        if (r.entities && r.entities.length > 0) {
                            ans = r.entities[0].value || "";
                        }
                        if (!ans) ans = r.text.substring(0, 30);
                        ans = ans.substring(0, 100).replace(/"/g, '""');
                        return `${r.video_id},${r.frame_idx},"${ans}"`;
                    }).join('\n');
                } else {
                    // Textual KIS
                    csvContent = rows.map(r => `${r.video_id},${r.frame_idx}`).join('\n');
                }

                const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
                const url = URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = `query-${type}.csv`;
                document.body.appendChild(a);
                a.click();
                document.body.removeChild(a);
                URL.revokeObjectURL(url);
                showToast(`Đã tải file query-${type}.csv chuẩn BTC!`);
            }


            function showToast(msg) {
                const t = document.getElementById('toast');
                t.innerText = msg;
                t.classList.add('show');
                setTimeout(() => t.classList.remove('show'), 2500);
            }

            function escapeHtml(text) {
                const map = {'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#039;'};
                return text.replace(/[&<>"']/g, function(m) { return map[m]; });
            }
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)


def main():
    global SEARCHER, KEYFRAMES_DIR, DB_PATH, INDEX_MAP, VIDEO_FRAME_TO_PATH
    parser = argparse.ArgumentParser(description="AI Challenge 2026 - OCR VBS Visualizer")
    parser.add_argument("--db", type=str, default="data/ocr_results/ocr_fts.db", help="Path to SQLite FTS5 database")
    parser.add_argument("--keyframes_dir", type=str, default=None, help="Path to Keyframes root directory")
    parser.add_argument("--index_json", type=str, default=None, help="Path to index.json (190k keyframes mapping)")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host address")
    parser.add_argument("--port", type=int, default=8000, help="Port number")
    args = parser.parse_args()

    DB_PATH = args.db
    KEYFRAMES_DIR = args.keyframes_dir

    if args.index_json and os.path.exists(args.index_json):
        try:
            with open(args.index_json, "r", encoding="utf-8") as f:
                INDEX_MAP = json.load(f)
                import re
                for fid, rel_path in INDEX_MAP.items():
                    parts = rel_path.replace("\\", "/").split("/")
                    if len(parts) >= 2:
                        vid = parts[-2]
                        fname = parts[-1]
                        m = re.search(r"\d+", fname)
                        if m:
                            VIDEO_FRAME_TO_PATH[(vid, int(m.group(0)))] = rel_path
            print(f"📑 Đã nạp index.json ({len(INDEX_MAP):,} keyframes)")
        except Exception as e:
            print(f"⚠️ Không thể nạp index.json: {e}")

    if os.path.exists(DB_PATH):
        SEARCHER = OCRIndexSearcher(DB_PATH)
        print(f"✅ Đã kết nối cơ sở dữ liệu FTS5: {DB_PATH}")
    else:
        print(f"⚠️ Cảnh báo: File database chưa tồn tại ({DB_PATH}). Hãy build index trước khi search.")

    print(f"🚀 Khởi chạy VBS Visualizer Server tại: http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()

