"""Stage 00 — shot segmentation.

Cuts a video into shots (camera takes) with PySceneDetect. Downstream stages
scope `track_id` to `shot_id` and never assume identity persists across a cut.
Times are stored as seconds on the source PTS time base (approximate at v1:
fps-derived; refine to true packet PTS later — see sync_uncertainty).

    python 00_ingest/shots.py <video> [--threshold 27] [--downscale 2]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from scenedetect import AdaptiveDetector, ContentDetector, SceneManager, open_video

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fightcv import contracts as C  # noqa: E402


def detect_shots(video_path: str, threshold: float = 27.0, downscale: int | None = None,
                 detector: str = "adaptive", min_scene_len: int = 15) -> pd.DataFrame:
    """Shot boundaries. `adaptive` (rolling-average) is far more robust to MMA's
    fast camera motion than `content` (fewer false cuts); min_scene_len suppresses
    sub-second flicker shots. NOTE: scenedetect still misses cuts between similar
    cage angles — TransNetV2 (learned) is the SOTA upgrade if this isn't enough."""
    video = open_video(video_path)
    sm = SceneManager()
    if downscale:
        sm.auto_downscale = False
        sm.downscale = downscale
    if detector == "adaptive":
        sm.add_detector(AdaptiveDetector(adaptive_threshold=threshold, min_scene_len=min_scene_len))
    else:
        sm.add_detector(ContentDetector(threshold=threshold, min_scene_len=min_scene_len))
    sm.detect_scenes(video, show_progress=False)
    scenes = sm.get_scene_list()
    rows = []
    for i, (start, end) in enumerate(scenes):
        rows.append({
            "shot_id": i,
            "start_frame": start.get_frames(),
            "end_frame": end.get_frames(),
            "start_s": round(start.get_seconds(), 4),
            "end_s": round(end.get_seconds(), 4),
            "n_frames": end.get_frames() - start.get_frames(),
            "dur_s": round(end.get_seconds() - start.get_seconds(), 4),
        })
    # single-shot fallback (scenedetect returns [] when it finds no cuts)
    if not rows:
        dur = video.duration.get_seconds() if video.duration else 0.0
        rows = [{"shot_id": 0, "start_frame": 0, "end_frame": video.duration.get_frames() if video.duration else 0,
                 "start_s": 0.0, "end_s": round(dur, 4), "n_frames": 0, "dur_s": round(dur, 4)}]
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--detector", choices=["adaptive", "content"], default="adaptive")
    ap.add_argument("--threshold", type=float, default=None,
                    help="adaptive_threshold (~3) for adaptive; content threshold (~27) for content")
    ap.add_argument("--min-scene-len", type=int, default=15)
    ap.add_argument("--downscale", type=int, default=2)
    args = ap.parse_args()

    thr = args.threshold if args.threshold is not None else (3.0 if args.detector == "adaptive" else 27.0)
    prov = C.SourceProvenance.probe(args.video)
    run_id = C.new_run_id()
    df = detect_shots(args.video, thr, args.downscale, args.detector, args.min_scene_len)
    out = C.VIDEO_DATA / prov.video_uid / "shots.parquet"
    C.write_table(df, out, run_id=run_id, source=prov,
                  extra={"detector": args.detector, "threshold": thr,
                         "min_scene_len": args.min_scene_len, "downscale": args.downscale})
    print(f"[shots] {len(df)} shots over {prov.duration_s:.0f}s "
          f"(median {df.dur_s.median():.1f}s) -> {out}")


if __name__ == "__main__":
    main()
