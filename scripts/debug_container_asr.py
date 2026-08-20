"""
Diagnose AsrSearcher inside backend.
"""
from app.searchers.asr import AsrSearcher
from app.stores.keyframe_map import get_keyframe_map
from app.services.search_service import SearchService, SearchRequest, EventQuery

def main():
    print("1. Loading KeyframeMap...")
    km = get_keyframe_map()
    print(f"KeyframeMap size: {len(km):,} keys")
    v_ids = km.frame_ids_for_video("L27_V010")
    print(f"L27_V010 frame_ids in map: {len(v_ids)}")
    for fid in v_ids[:5]:
        meta = km.get(fid)
        print(f"   fid={fid} meta={meta}")

    print("\n2. Running AsrSearcher directly...")
    asr = AsrSearcher()
    hits = asr.search({0: "Hỏa hồng Nhật Tảo oanh thiên địa"}, top_k=10)
    print(f"Hits for event 0: {len(hits.get(0, []))}")
    for h in hits.get(0, []):
        meta = km.get(h.frame_id)
        print(f"   Hit fid={h.frame_id} score={h.score:.3f} meta={meta} evidence={h.evidence.get('text') if h.evidence else None}")

    print("\n3. Running Full SearchService...")
    svc = SearchService()
    import asyncio
    req = SearchRequest(
        mode="single",
        translate_query=False,
        events=[
            EventQuery(
                asr_query="Hỏa hồng Nhật Tảo oanh thiên địa",
                searchers={"asr": True, "beit3": False, "clip_h14": False, "siglip2": False, "ocr": False}
            )
        ]
    )
    res = asyncio.run(svc.search(req))
    candidates = res.events[0].candidates
    print(f"Full search result: {len(candidates)} candidates")
    for c in candidates[:5]:
        print(f"   Candidate: Video {c.video_id} Frame {c.frame_idx} Time {c.timestamp}s ASR: {c.asr_text[:50]}")

if __name__ == "__main__":
    main()
