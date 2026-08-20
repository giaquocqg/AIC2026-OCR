"""
Test searching 'Hỏa hồng Nhật Tảo oanh thiên địa' via Backend API using ASR channel.
"""
import requests

payload = {
    "mode": "single",
    "translate_query": False,
    "events": [
        {
            "query": None,
            "ocr_query": None,
            "asr_query": "Hỏa hồng Nhật Tảo oanh thiên địa",
            "searchers": {
                "beit3": False,
                "clip_h14": False,
                "siglip2": False,
                "ocr": False,
                "asr": True
            }
        }
    ]
}

r = requests.post("http://localhost:8080/api/search", json=payload)
print("Status Code:", r.status_code)
if r.status_code == 200:
    res = r.json()
    candidates = res.get("events", [{}])[0].get("candidates", [])
    print(f"🎉 Found {len(candidates)} candidates for 'Hỏa hồng Nhật Tảo oanh thiên địa' via ASR!")
    for idx, c in enumerate(candidates[:5], 1):
        print(f"Top {idx}: Video {c['video_id']} | Frame {c['frame_idx']} | Time {c.get('timestamp')}s | ASR: {c.get('asr_text')}")
else:
    print("Error:", r.text)
