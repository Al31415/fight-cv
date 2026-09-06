"""Direction A prototype — appearance-based per-fighter identity (the 'tattoo' fix).

Build a per-fighter appearance prototype from a clean frame (where tattoos/shorts
are visible) using DINOv2 dense patch features pooled inside each SAM 2 mask, then
assign every patch on the pile frames to whichever fighter it *looks* like. This
separates intertwined bodies by appearance where mask-tracking blends them.
Outputs a comparison sheet: [original | SAM 2 mask | appearance-identity].

Run from /root (sam2 pkg shadow):
    python /root/appearance_identity.py --clip <clip> --masks <sam2 parquet> \
        --seed-frame 38 --test-frames 930 990 1050 1110 --out <dir>
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

PATCH = 14
FCOL = {0: (0, 165, 255), 1: (255, 150, 0)}


def dino_features(model, frame_bgr, Wp, Hp, device):
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    img = cv2.cvtColor(cv2.resize(frame_bgr, (Wp * PATCH, Hp * PATCH)), cv2.COLOR_BGR2RGB)
    t = torch.from_numpy(img).float().permute(2, 0, 1).unsqueeze(0).to(device) / 255.0
    t = (t - mean) / std
    with torch.inference_mode():
        out = model(pixel_values=t).last_hidden_state  # (1, 1+Hp*Wp, D)
    feat = out[0, 1:].reshape(Hp, Wp, -1)
    return torch.nn.functional.normalize(feat, dim=-1)   # (Hp,Wp,D)


def mask_at(masks, frame, W, H):
    out = {}
    for _, r in masks[masks.frame == frame].iterrows():
        m = np.zeros((H, W), np.uint8)
        cv2.fillPoly(m, [np.array(json.loads(r.polygon), np.int32)], 1)
        out[int(r.fighter)] = m
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", required=True)
    ap.add_argument("--masks", required=True)
    ap.add_argument("--seed-frame", type=int, default=38)
    ap.add_argument("--test-frames", type=int, nargs="+", required=True)
    ap.add_argument("--thresh", type=float, default=0.35)
    ap.add_argument("--out", default="/workspace/fightcv/out/appearance")
    args = ap.parse_args()
    outdir = Path(args.out); outdir.mkdir(parents=True, exist_ok=True)

    import pandas as pd
    from transformers import AutoModel
    device = "cuda"
    model = AutoModel.from_pretrained("facebook/dinov2-large").to(device).eval()
    masks = pd.read_parquet(args.masks)

    cap = cv2.VideoCapture(args.clip)
    W, H = int(cap.get(3)), int(cap.get(4))
    Wp, Hp = (W // PATCH // 2) * 2, (H // PATCH // 2) * 2   # ~ half-res patch grid, even

    # build per-fighter prototypes from the clean seed frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.seed_frame); _, seed = cap.read()
    feat = dino_features(model, seed, Wp, Hp, device)     # (Hp,Wp,D)
    seed_masks = mask_at(masks, args.seed_frame, W, H)
    protos = {}
    for f, m in seed_masks.items():
        mp = cv2.resize(m, (Wp, Hp), interpolation=cv2.INTER_NEAREST).astype(bool)
        if mp.sum() < 3:
            continue
        # a few part-prototypes (k=4) so a tattooed leg matches a leg, not the torso mean
        pf = feat[torch.from_numpy(mp).to(device)]        # (n,D)
        k = min(4, pf.shape[0])
        idx = torch.randperm(pf.shape[0], device=device)[:k]
        cent = pf[idx]
        for _ in range(5):
            sim = pf @ cent.T
            assign = sim.argmax(1)
            cent = torch.stack([torch.nn.functional.normalize(pf[assign == j].mean(0), dim=0)
                                if (assign == j).any() else cent[j] for j in range(k)])
        protos[f] = cent                                   # (k,D)
    print(f"[appearance] prototypes for fighters {sorted(protos)} (k-part each)")

    sheet = []
    for fr in args.test_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fr); ok, img = cap.read()
        if not ok:
            continue
        feat = dino_features(model, img, Wp, Hp, device)
        flat = feat.reshape(-1, feat.shape[-1])
        best = torch.full((flat.shape[0],), -1, dtype=torch.long, device=device)
        bestsim = torch.full((flat.shape[0],), args.thresh, device=device)
        for f, cent in protos.items():
            s = (flat @ cent.T).max(1).values
            take = s > bestsim
            best[take] = f; bestsim[take] = s[take]
        lab = best.reshape(Hp, Wp).cpu().numpy()
        labpx = cv2.resize(lab.astype(np.int16), (W, H), interpolation=cv2.INTER_NEAREST)
        # panels: original | SAM2 mask | appearance-identity
        p_orig = img.copy()
        p_sam = img.copy()
        for f, m in mask_at(masks, fr, W, H).items():
            ov = np.zeros_like(img); ov[m.astype(bool)] = FCOL[f]; p_sam = cv2.addWeighted(ov, 0.45, p_sam, 0.55, 0)
        p_app = img.copy()
        for f in protos:
            ov = np.zeros_like(img); ov[labpx == f] = FCOL[f]; p_app = cv2.addWeighted(ov, 0.5, p_app, 0.5, 0)
        for lab_txt, p in [("orig", p_orig), ("SAM2 mask", p_sam), ("appearance-ID", p_app)]:
            cv2.putText(p, f"f{fr} {lab_txt}", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        sheet.append(np.hstack([cv2.resize(x, (426, 240)) for x in (p_orig, p_sam, p_app)]))
        print(f"[appearance] frame {fr} done")
    cap.release()
    cv2.imwrite(str(outdir / "appearance_compare.png"), np.vstack(sheet))
    print(f"[appearance] -> {outdir/'appearance_compare.png'}")


if __name__ == "__main__":
    main()
