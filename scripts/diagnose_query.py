"""
Script to diagnose search matches for 'cơm tấm 3 ghiền'.
"""
import json
import sqlite3
import requests

def main():
    print("=== 1. CHECKING RAW CHECKPOINT DATA ===")
    checkpoint_file = "data/ocr_results/.ocr_records_checkpoint.jsonl"
    with open(checkpoint_file, "r", encoding="utf-8") as f:
        recs = [json.loads(l) for l in f if l.strip()]

    print(f"Total records: {len(recs):,}")

    ghien_matches = [r for r in recs if "ghiền" in r["text"].lower() or "ghien" in r.get("text_unsigned", "").lower()]
    print(f"Records containing 'ghiền/ghien': {len(ghien_matches)}")
    for r in ghien_matches[:5]:
        print(f"   -> {r['video_id']} #{r['frame_idx']}: {r['text']}")

    comtam_matches = [r for r in recs if "cơm tấm" in r["text"].lower() or "com tam" in r.get("text_unsigned", "").lower()]
    print(f"Records containing 'cơm tấm': {len(comtam_matches)}")
    for r in comtam_matches[:5]:
        print(f"   -> {r['video_id']} #{r['frame_idx']}: {r['text']}")

    print("\n=== 2. CHECKING SQLITE FTS5 (STRICT MATCH) ===")
    conn = sqlite3.connect("data/ocr_results/ocr_fts.db")
    cur = conn.cursor()
    cur.execute("SELECT frame_id, video_id, text FROM ocr_fts WHERE text MATCH ? LIMIT 10", ("\"cơm tấm\"",))
    fts_comtam = cur.fetchall()
    print(f"SQLite FTS exact '\"cơm tấm\"' matches: {len(fts_comtam)}")

    # Check some real keywords that exist in the indexed videos
    for test_kw in ["thời sự", "bản tin", "sức khỏe", "nghiên cứu", "covid", "giá"]:
        cur.execute("SELECT count(*) FROM ocr_fts WHERE text MATCH ?", (f'"{test_kw}"',))
        cnt = cur.fetchone()[0]
        print(f"   -> Keyword '{test_kw}': {cnt} matches in SQLite FTS")

    print("\n=== 3. CHECKING ELASTICSEARCH RAW MATCHES ===")
    # Query exact phrase
    es_res = requests.post("http://localhost:9200/frames_ocr/_search", json={
        "query": {
            "match_phrase": {
                "ocr_text": "cơm tấm"
            }
        },
        "size": 10
    }).json()
    hits = es_res.get("hits", {}).get("hits", [])
    print(f"Elasticsearch match_phrase 'cơm tấm': {len(hits)} hits (total: {es_res.get('hits', {}).get('total', {})})")
    for h in hits[:5]:
        print(f"   -> ID {h['_id']} | Score {h['_score']:.2f} | {h['_source']['video_id']} #{h['_source']['frame_idx']}: {h['_source']['ocr_text']}")

    print("\n=== 4. CHECKING ELASTICSEARCH DEFAULT OR MATCH (What backend uses) ===")
    es_or_res = requests.post("http://localhost:9200/frames_ocr/_search", json={
        "query": {
            "match": {
                "ocr_text": {
                    "query": "cơm tấm 3 ghiền",
                    "operator": "or",
                    "fuzziness": "AUTO"
                }
            }
        },
        "size": 10
    }).json()
    or_hits = es_or_res.get("hits", {}).get("hits", [])
    print(f"Elasticsearch OR match 'cơm tấm 3 ghiền': total {es_or_res.get('hits', {}).get('total', {})}")
    for h in or_hits[:5]:
        print(f"   -> ID {h['_id']} | Score {h['_score']:.2f} | {h['_source']['video_id']} #{h['_source']['frame_idx']}: {h['_source']['ocr_text']}")

if __name__ == "__main__":
    main()
