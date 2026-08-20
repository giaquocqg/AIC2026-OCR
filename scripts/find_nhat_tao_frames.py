"""
Find keyframes for L27_V010 around 219.3s - 228.7s.
"""
import os
import json

def main():
    fps_file = "/home/quoc/AIC2026/data/metadata/fps.json"
    fps = 25.0
    if os.path.exists(fps_file):
        fps_map = json.load(open(fps_file, "r", encoding="utf-8"))
        fps = float(fps_map.get("L27_V010", 25.0))

    start_sec = 219.319
    end_sec = 228.72
    start_frame = int(start_sec * fps)
    end_frame = int(end_sec * fps)

    print(f"Video: L27_V010 | FPS: {fps}")
    print(f"Time range: {start_sec}s - {end_sec}s -> Frames: {start_frame} - {end_frame}")

    # Check keyframe files in /home/quoc/AIC2026/data/keyframe/keyframe/L27_V010
    vdir = "/home/quoc/AIC2026/data/keyframe/keyframe/L27_V010"
    if os.path.exists(vdir):
        files = sorted(os.listdir(vdir))
        print(f"Found {len(files)} keyframes in {vdir}")
        for f in files:
            # check frame index
            try:
                # e.g. keyframe_5500.webp or 5500.webp
                num_str = "".join([c for c in f.split(".")[0] if c.isdigit()])
                if num_str:
                    num = int(num_str)
                    if start_frame - 200 <= num <= end_frame + 200:
                        sec = round(num / fps, 2)
                        print(f"   🎯 Keyframe: {f} (Frame #{num}, ~{sec}s)")
            except Exception:
                pass

if __name__ == "__main__":
    main()
