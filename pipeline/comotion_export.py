"""Export CoMotion per-detection COCO-17 joints (2D projected + 3D) to parquet,
so identity-composition + the event layer run locally without the pod.
Runs in the CoMotion venv on the pod.
"""
import sys
import numpy as np
import pandas as pd
import torch
from comotion_demo.utils import smpl_kinematics, helper

PT = sys.argv[1] if len(sys.argv) > 1 else "/workspace/fightcv/out/comotion/trial_clip.pt"
START = int(sys.argv[2]) if len(sys.argv) > 2 else 300
OUT = sys.argv[3] if len(sys.argv) > 3 else "/workspace/fightcv/out/comotion/pose_coco.parquet"
CLIP = sys.argv[4] if len(sys.argv) > 4 else "/workspace/fightcv/videos/trial_clip.mp4"
import cv2
cap = cv2.VideoCapture(CLIP); W = int(cap.get(3)); H = int(cap.get(4)); cap.release()
maxres = max(W, H)
K = torch.tensor([[2*maxres, 0, 0.5*W],[0, 2*maxres, 0.5*H]], dtype=torch.float32, device="cuda")

d = torch.load(PT, map_location="cuda", weights_only=False)
pose, trans, betas = d["pose"].cuda().float(), d["trans"].cuda().float(), d["betas"].cuda().float()
ids, fidx = d["id"].cpu().numpy(), d["frame_idx"].cpu().numpy()
smpl = smpl_kinematics.SMPLKinematics().cuda().eval()

def j3d_of(sel):
    try:
        return smpl(pose=pose[sel], betas=betas[sel], trans=trans[sel], output_format="joints_coco")
    except TypeError:
        return smpl(pose=pose[sel], betas=betas[sel], output_format="joints_coco") + trans[sel][:, None, :]

rows = []
rel0 = int(fidx.min())
uniq = sorted(set(fidx.tolist()))
for f in uniq:
    sel = np.where(fidx == f)[0]
    if not len(sel):
        continue
    j3 = j3d_of(torch.as_tensor(sel, device="cuda"))          # (n,17,3)
    j2 = helper.project_to_2d(K, j3).detach().cpu().numpy()   # (n,17,2)
    j3n = j3.detach().cpu().numpy()
    src_frame = START + (f - rel0)
    for k, det in enumerate(sel):
        for kp in range(17):
            rows.append({"src_frame": int(src_frame), "track_id": int(ids[det]), "kpt": kp,
                         "x2d": float(j2[k, kp, 0]), "y2d": float(j2[k, kp, 1]),
                         "x3d": float(j3n[k, kp, 0]), "y3d": float(j3n[k, kp, 1]), "z3d": float(j3n[k, kp, 2])})
df = pd.DataFrame(rows)
df.to_parquet(OUT, index=False)
print("EXPORT_DONE", OUT, "rows", len(df), "frames", df.src_frame.nunique(), "tracks", df.track_id.nunique(), "WH", W, H)
