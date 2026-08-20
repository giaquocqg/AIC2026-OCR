"""
Diagnose AsrSearcher inside backend.
"""
from app.searchers.asr import AsrSearcher
from app.stores.keyframe_map import get_keyframe_map


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
        ev_text = h.evidence.get('text') if h.evidence else None
        print(f"   Hit fid={h.frame_id} score={h.score:.3f} meta={meta} evidence={ev_text}")

if __name__ == "__main__":
    main()

