"""
Check 'HỎA HỒNG NHẬT TẢO OANH THIÊN ĐỊA' across OCR and ASR on the server.
"""
import os
import json
import glob
import sqlite3
import requests

def main():
    query = "Hỏa hồng Nhật Tảo oanh thiên địa"
    print(f"🔎 SEARCHING FOR: '{query}'\n")

    # 1. Check in OCR checkpoint
    cp_path = "data/ocr_results/.ocr_records_checkpoint.jsonl"
    if os.path.exists(cp_path):
        with open(cp_path, "r", encoding="utf-8") as f:
            ocr_recs = [json.loads(l) for l in f if l.strip()]
        
        matches = []
        for r in ocr_recs:
            t = (r.get("text", "") + " " + r.get("text_unsigned", "")).lower()
            if "nhật tảo" in t or "nhat tao" in t or "hỏa hồng" in t or "hoa hong" in t or "oanh thiên địa" in t or "oanh thien dia" in t:
                matches.append(r)
        
        print(f"1. OCR Checkpoint ({len(ocr_recs):,} records): Found {len(matches)} matches")
        for m in matches[:10]:
            print(f"   [OCR] Frame {m.get('frame_id')} | {m.get('video_id')} #{m.get('frame_idx')}: {m.get('text')}")
    else:
        print("OCR checkpoint not found")

    # 2. Check in SQLite FTS
    db_path = "data/ocr_results/ocr_fts.db"
    if os.path.exists(db_path):
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.execute("SELECT frame_id, video_id, text FROM ocr_fts WHERE text MATCH ? LIMIT 10", ("\"Nhật Tảo\" OR \"Hỏa hồng\" OR \"oanh thiên địa\"",))
        fts_rows = cur.fetchall()
        print(f"\n2. SQLite FTS: Found {len(fts_rows)} matches")
        for r in fts_rows:
            print(f"   [FTS] Frame {r[0]} | {r[1]}: {r[2]}")

    # 3. Check in ASR Transcripts (Files in /home/quoc/AIC2026/data/transcripts)
    asr_dir = "/home/quoc/AIC2026/data/transcripts"
    if os.path.exists(asr_dir):
        asr_files = glob.glob(os.path.join(asr_dir, "*.json")) + glob.glob(os.path.join(asr_dir, "*.jsonl"))
        print(f"\n3. ASR Transcripts ({len(asr_files)} files in {asr_dir}):")
        asr_matches = []
        for fpath in asr_files:
            try:
                content = open(fpath, "r", encoding="utf-8").read()
                if "nhật tảo" in content.lower() or "nhat tao" in content.lower() or "hỏa hồng" in content.lower() or "oanh thiên địa" in content.lower() or "nguyễn trung trực" in content.lower():
                    asr_matches.append(os.path.basename(fpath))
            except Exception:
                pass
        print(f"   Found {len(asr_matches)} matching ASR files: {asr_matches[:10]}")

    # 4. Check in Elasticsearch 'transcript_segments'
    try:
        es_res = requests.post("http://localhost:9200/transcript_segments/_search", json={
            "query": {
                "match": {
                    "clean_text": "Hỏa hồng Nhật Tảo oanh thiên địa"
                }
            },
            "size": 5
        }).json()
        hits = es_res.get("hits", {}).get("hits", [])
        print(f"\n4. Elasticsearch ASR 'transcript_segments': Found {len(hits)} hits (total {es_res.get('hits', {}).get('total', {})})")
        for h in hits:
            src = h["_source"]
            print(f"   [ASR ES] Score {h['_score']:.2f} | Video {src.get('video_id')} [{src.get('start_sec')}s - {src.get('end_sec')}s]: {src.get('clean_text')}")
    except Exception as e:
        print(f"Error querying ES: {e}")

if __name__ == "__main__":
    main()
