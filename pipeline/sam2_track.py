"""SAM 2 video tracking of the two fighters through occlusion (grappling fix).

Seed each fighter with a box at a clean pre-pile frame; SAM 2 propagates with a
bounded memory bank that holds the occluded (bottom) fighter through the pile.
Reports per-fighter coverage (and inside a pile window) + writes masklets + an
overlay mp4.

    python sam2_track.py --clip <clip> --t0 558 --seed-frame 38 \
        --seed-boxes "466,122,720,542" "744,84,844,450" --pile 30 39 --out out/sam2
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def mask_bbox_poly(mask):
    m = mask.astype(np.uint8)
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None, None
    c = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(c) < 30:
        return None, None
    eps = 0.008 * cv2.arcLength(c, True)
    poly = cv2.approxPolyDP(c, eps, True).reshape(-1, 2).astype(int).tolist()
    x, y, w, h = cv2.boundingRect(c)
    return [int(x), int(y), int(x + w), int(y + h)], poly


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", required=True)
    ap.add_argument("--t0", type=float, default=0.0)
    ap.add_argument("--seed-frame", type=int, default=0)
    ap.add_argument("--seed-boxes", nargs="+", required=True, help="'x1,y1,x2,y2' px per fighter")
    ap.add_argument("--pile", type=float, nargs=2, default=None)
    ap.add_argument("--out", default="out/sam2")
    ap.add_argument("--ckpt", default="/workspace/fightcv/sam2_ckpt/sam2.1_hiera_large.pt")
    ap.add_argument("--cfg", default="configs/sam2.1/sam2.1_hiera_l.yaml")
    args = ap.parse_args()
    outdir = Path(args.out); outdir.mkdir(parents=True, exist_ok=True)

    import torch
    import pandas as pd
    from sam2.build_sam import build_sam2_video_predictor

    cap = cv2.VideoCapture(args.clip)
    W, H = int(cap.get(3)), int(cap.get(4)); fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    nfr = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); cap.release()

    predictor = build_sam2_video_predictor(args.cfg, args.ckpt, device="cuda")
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        state = predictor.init_state(video_path=args.clip)
        boxes = [np.array([float(a) for a in b.split(",")], dtype=np.float32) for b in args.seed_boxes]
        for i, box in enumerate(boxes):
            predictor.add_new_points_or_box(state, frame_idx=args.seed_frame, obj_id=i, box=box)
        print(f"[sam2] seeded {len(boxes)} fighters by box at frame {args.seed_frame}")

        rows = []
        present = {i: 0 for i in range(len(boxes))}
        pile_present = {i: 0 for i in range(len(boxes))}; pile_total = 0
        for fidx, obj_ids, mask_logits in predictor.propagate_in_video(state):
            pts = args.t0 + fidx / fps
            in_pile = args.pile and (args.pile[0] <= fidx / fps <= args.pile[1])
            if in_pile:
                pile_total += 1
            for k, oid in enumerate(obj_ids):
                m = (mask_logits[k, 0] > 0).cpu().numpy()
                bbox, poly = mask_bbox_poly(m)
                if bbox is None:
                    continue
                present[int(oid)] += 1
                if in_pile:
                    pile_present[int(oid)] += 1
                rows.append({"frame": fidx, "pts_s": round(pts, 3), "fighter": int(oid),
                             "x1": bbox[0], "y1": bbox[1], "x2": bbox[2], "y2": bbox[3],
                             "polygon": json.dumps(poly)})

    pd.DataFrame(rows).to_parquet(outdir / "masklets_video.parquet", index=False)
    print(f"[sam2] per-fighter coverage over {nfr} frames: " +
          ", ".join(f"F{i}={present[i]}" for i in sorted(present)))
    if args.pile and pile_total:
        print(f"[sam2] PILE {args.pile}s ({pile_total} frames): " +
              ", ".join(f"F{i}={pile_present[i]}/{pile_total}" for i in sorted(present)))
    print(f"[sam2] -> {outdir/'masklets_video.parquet'}")


if __name__ == "__main__":
    main()
