"""Stage 01 (partial) — per-shot fighter detection + tracking.

Reuses the mma-fighter-tracker approach (YOLO person detection + BotSORT) but
honors the review feedback: track_id is scoped to shot_id (the tracker is RESET
at every shot boundary — never a stable id through a cut). Identity assignment
(fighter_uid) is a separate, later, probabilistic step; this stage only emits
per-shot tracks. Frames are stamped with an approximate source-PTS second
(frame/fps at v1; refine to packet PTS later).

    python 01_perceive/track.py <video> [--t0 0 --t1 0] [--stride 3] [--model yolo11m.pt]

Writes data/ufc/video/<uid>/tracks.parquet (append-safe per window).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fightcv import contracts as C  # noqa: E402


def shot_of(frame_idx: int, shots: pd.DataFrame, fps: float) -> int:
    if not len(shots):
        return 0
    t = frame_idx / fps
    m = shots[(shots.start_s <= t) & (t < shots.end_s)]
    return int(m.shot_id.iloc[0]) if len(m) else -1


def run(video: str, t0: float, t1: float, stride: int, model_path: str,
        conf: float, max_persons: int) -> pd.DataFrame:
    from ultralytics import YOLO
    import torch
    prov = C.SourceProvenance.probe(video)
    fps = prov.avg_fps or 30.0
    shots_fp = C.VIDEO_DATA / prov.video_uid / "shots.parquet"
    shots = pd.read_parquet(shots_fp) if shots_fp.exists() else pd.DataFrame()
    dev = 0 if torch.cuda.is_available() else "cpu"
    model = YOLO(model_path)

    cap = cv2.VideoCapture(video)
    f0 = int(t0 * fps)
    f1 = int(t1 * fps) if t1 > 0 else int(prov.duration_s * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, f0)

    rows, cur_shot, first_in_shot = [], None, True
    fidx = f0
    while fidx < f1:
        ok, frame = cap.read()
        if not ok:
            break
        if (fidx - f0) % stride != 0:
            fidx += 1
            continue
        sid = shot_of(fidx, shots, fps)
        if sid != cur_shot:            # shot boundary -> reset tracker
            cur_shot, first_in_shot = sid, True
        res = model.track(frame, persist=not first_in_shot, classes=[0], conf=conf,
                          tracker="botsort.yaml", device=dev, verbose=False)
        first_in_shot = False
        r = res[0]
        if r.boxes is not None and r.boxes.id is not None:
            xyxy = r.boxes.xyxy.cpu().numpy()
            ids = r.boxes.id.cpu().numpy().astype(int)
            cf = r.boxes.conf.cpu().numpy()
            # keep the largest max_persons boxes (the fighters, not background people)
            order = np.argsort(-(xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1]))[:max_persons]
            for i in order:
                x1, y1, x2, y2 = xyxy[i]
                rows.append({
                    "frame": fidx, "pts_s": round(fidx / fps, 3),
                    "shot_id": sid, "shot_track_id": f"{sid}:{ids[i]}",
                    "track_id": int(ids[i]), "x1": float(x1), "y1": float(y1),
                    "x2": float(x2), "y2": float(y2), "conf": round(float(cf[i]), 3),
                })
        fidx += 1
    cap.release()
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--t0", type=float, default=0.0)
    ap.add_argument("--t1", type=float, default=0.0)
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--model", default="yolo11m.pt")
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--max-persons", type=int, default=2)
    args = ap.parse_args()

    prov = C.SourceProvenance.probe(args.video)
    run_id = C.new_run_id()
    df = run(args.video, args.t0, args.t1, args.stride, args.model, args.conf, args.max_persons)
    out = C.VIDEO_DATA / prov.video_uid / "tracks.parquet"
    C.write_table(df, out, run_id=run_id, source=prov,
                  extra={"model": args.model, "stride": args.stride, "conf": args.conf,
                         "window_s": [args.t0, args.t1], "per_shot_reset": True,
                         "note": "shot-scoped track_id; identity assignment is a later stage"})
    n_shots = df.shot_id.nunique() if len(df) else 0
    print(f"[track] {len(df)} boxes over {n_shots} shots -> {out}")


if __name__ == "__main__":
    main()
