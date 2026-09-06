"""SAM 3 multiplex VIDEO tracking on the pod — detect-through-occlusion.

Seeds the 'fighter' concept at a clean (pre-pile) frame, keeps the two
fighters, and PROPAGATES with memory so the occluded (bottom) fighter is held
through the pile instead of being dropped by per-frame detection. Reports
per-fighter frame coverage (and coverage inside a pile window) and writes
masklets + an overlay mp4.

    python sam3_video_grapple.py --clip <clip.mp4> --t0 558 --seed-frame 0 \
        --pile 30 45 --out out/sam3_grapple
"""
import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np


def mask_bbox_poly(mask, W, H):
    m = (mask > 0).astype(np.uint8)
    if m.shape[:2] != (H, W):
        m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
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
    ap.add_argument("--seed-boxes", nargs="+", required=True,
                    help="one 'x1,y1,x2,y2' per fighter (relative 0-1 coords) at the seed frame")
    ap.add_argument("--pile", type=float, nargs=2, default=None, help="pile window in clip seconds")
    ap.add_argument("--out", default="out/sam3_grapple")
    args = ap.parse_args()
    outdir = Path(args.out); outdir.mkdir(parents=True, exist_ok=True)

    import torch
    import pandas as pd
    from sam3.model_builder import build_sam3_multiplex_video_predictor

    cap = cv2.VideoCapture(args.clip)
    W, H = int(cap.get(3)), int(cap.get(4)); fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    predictor = build_sam3_multiplex_video_predictor(
        bpe_path="/workspace/fightcv/sam3/sam3/assets/bpe_simple_vocab_16e6.txt.gz")
    import inspect
    _m = getattr(predictor, "model", None)
    if _m is not None and "offload_state_to_cpu" not in inspect.signature(_m.init_state).parameters:
        _orig = _m.init_state
        _m.init_state = lambda *a, offload_state_to_cpu=None, **k: _orig(*a, **k)
    if _m is not None:
        for mod in _m.modules():
            if getattr(mod, "use_fa3", False):
                mod.use_fa3 = False
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_math_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)

    t = time.time()
    r = predictor.handle_request(dict(type="start_session", resource_path=args.clip,
                                      offload_video_to_cpu=True))
    sid = r["session_id"]
    # box-based seed: one box per fighter, each its own obj_id (detector geometry prompt)
    seed_boxes = [[float(a) for a in b.split(",")] for b in args.seed_boxes]
    for i, box in enumerate(seed_boxes):
        resp = predictor.handle_request(dict(
            type="add_prompt", session_id=sid, frame_index=args.seed_frame,
            bounding_boxes=torch.tensor([box], dtype=torch.float32),
            bounding_box_labels=torch.tensor([1], dtype=torch.int32),
            obj_id=i, rel_coordinates=True))
        oids = list(resp.get("outputs", {}).get("out_obj_ids", []))
        print(f"[sam3vid] seed box obj {i}: response obj_ids {oids}")
    fighter_of = {i: i for i in range(len(seed_boxes))}
    keep_ids = set(fighter_of)
    print(f"[sam3vid] seeded {len(seed_boxes)} fighters by box at frame {args.seed_frame} "
          f"(model loaded+seeded in {time.time()-t:.0f}s)")

    rows = []
    present = {oid: 0 for oid in keep_ids}
    pile_present = {oid: 0 for oid in keep_ids}
    pile_total = 0
    for resp in predictor.handle_stream_request(dict(type="propagate_in_video", session_id=sid)):
        ci = resp["frame_index"]; o = resp["outputs"]
        pts = args.t0 + ci / fps
        in_pile = args.pile and (args.pile[0] <= ci / fps <= args.pile[1])
        if in_pile:
            pile_total += 1
        seen_this = set()
        for oid, mk in zip(o["out_obj_ids"], o["out_binary_masks"]):
            if int(oid) not in keep_ids:
                continue
            bbox, poly = mask_bbox_poly(mk, W, H)
            if bbox is None:
                continue
            present[int(oid)] += 1
            seen_this.add(int(oid))
            rows.append({"frame": ci, "pts_s": round(pts, 3), "obj_id": int(oid),
                         "fighter": fighter_of[int(oid)], "x1": bbox[0], "y1": bbox[1],
                         "x2": bbox[2], "y2": bbox[3], "polygon": json.dumps(poly)})
        if in_pile:
            for oid in seen_this:
                pile_present[oid] += 1
    predictor.handle_request(dict(type="close_session", session_id=sid))

    df = pd.DataFrame(rows)
    df.to_parquet(outdir / "masklets_video.parquet", index=False)
    nfr = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); cap.release()
    print(f"[sam3vid] per-fighter coverage over {nfr} frames: " +
          ", ".join(f"F{fighter_of[o]}={present[o]}" for o in sorted(keep_ids)))
    if args.pile and pile_total:
        print(f"[sam3vid] PILE window {args.pile}s ({pile_total} frames) coverage: " +
              ", ".join(f"F{fighter_of[o]}={pile_present[o]}/{pile_total}" for o in sorted(keep_ids)))
    print(f"[sam3vid] -> {outdir/'masklets_video.parquet'}")


if __name__ == "__main__":
    main()
