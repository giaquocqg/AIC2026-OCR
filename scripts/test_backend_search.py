"""
Test searching directly against the main backend API.
"""
import requests

payload = {
    "mode": "single",
    "translate_query": False,
    "events": [
        {
            "query": None,
            "ocr_query": "cơm tấm",
            "asr_query": None,
            "searchers": {
                "beit3": False,
                "clip_h14": False,
                "siglip2": False,
                "ocr": True,
                "asr": False
            }
        }
    ]
}

r = requests.post("http://localhost:8080/api/search", json=payload)
print("Status Code:", r.status_code)
if r.status_code == 200:
    res = r.json()
    candidates = res.get("events", [{}])[0].get("candidates", [])
    print(f"Found {len(candidates)} candidates for 'cơm tấm'!")
    for idx, c in enumerate(candidates[:5], 1):
        print(f"Top {idx}: Frame {c['frame_id']} | Video: {c['video_id']} #{c['frame_idx']} | OCR: {c['ocr_text'][:60]}")
else:
    print("Error:", r.text)
