"""Render CoMotion 3D joints (COCO-17) projected to 2D, overlaid on the clip.
Runs in the CoMotion venv on the pod. Colors per track id; highlights wrists/ankles
(strike-contact points) in red. Produces an mp4 + a contact sheet around the kick.
"""
import sys
import cv2
import numpy as np
import torch
from comotion_demo.utils import smpl_kinematics, helper

PT = "/workspace/fightcv/out/comotion/trial_clip.pt"
CLIP = "/workspace/fightcv/videos/trial_clip.mp4"
START = int(sys.argv[1]) if len(sys.argv) > 1 else 480  # matches demo --start-frame
OUT_MP4 = "/workspace/fightcv/out/comotion/verify_comotion.mp4"
OUT_SHEET = "/workspace/fightcv/out/comotion/verify_comotion_sheet.png"

EDGES = [(5,7),(7,9),(6,8),(8,10),(11,13),(13,15),(12,14),(14,16),
         (5,6),(11,12),(5,11),(6,12),(0,5),(0,6)]
EXTREM = {9,10,15,16}

def color_for(i):
    np.random.seed(int(i)*7919 + 1)
    return tuple(int(x) for x in np.random.randint(60,255,3))

d = torch.load(PT, map_location="cuda", weights_only=False)
pose, trans, betas = d["pose"].cuda().float(), d["trans"].cuda().float(), d["betas"].cuda().float()
ids, fidx = d["id"].cpu().numpy(), d["frame_idx"].cpu().numpy()
print("detections:", len(ids), "frame_idx range:", fidx.min(), fidx.max(), "n tracks:", len(set(ids.tolist())))

smpl = smpl_kinematics.SMPLKinematics().cuda().eval()
cap = cv2.VideoCapture(CLIP)
W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
maxres = max(W, H)
K = torch.tensor([[2*maxres, 0, 0.5*W],[0, 2*maxres, 0.5*H]], dtype=torch.float32, device="cuda")

# does forward accept trans? try, else add manually
def joints2d(sel):
    try:
        j3d = smpl(pose=pose[sel], betas=betas[sel], trans=trans[sel], output_format="joints_coco")
    except TypeError:
        j3d = smpl(pose=pose[sel], betas=betas[sel], output_format="joints_coco") + trans[sel][:, None, :]
    return helper.project_to_2d(K, j3d).detach().cpu().numpy()  # (n,17,2)

frames = sorted(set(fidx.tolist()))
rel0 = min(frames)                      # frame_idx may be 0-based or absolute
vw = cv2.VideoWriter(OUT_MP4, cv2.VideoWriter_fourcc(*"mp4v"), 8.0, (W, H))
kick_rel = 540 - START                  # kick ~ source frame 540
sheet = []
for f in frames:
    src_frame = START + (f - rel0)
    cap.set(cv2.CAP_PROP_POS_FRAMES, src_frame)
    ok, img = cap.read()
    if not ok:
        continue
    sel = np.where(fidx == f)[0]
    if len(sel):
        j2 = joints2d(torch.as_tensor(sel, device="cuda"))
        for k, det in enumerate(sel):
            col = color_for(ids[det]); kp = j2[k]
            for a, b in EDGES:
                pa, pb = kp[a], kp[b]
                if np.all(np.isfinite(pa)) and np.all(np.isfinite(pb)):
                    cv2.line(img, tuple(pa.astype(int)), tuple(pb.astype(int)), col, 2)
            for jj in range(17):
                p = kp[jj]
                if np.all(np.isfinite(p)):
                    r = 6 if jj in EXTREM else 3
                    c = (0,0,255) if jj in EXTREM else col
                    cv2.circle(img, tuple(p.astype(int)), r, c, -1)
    vw.write(img)
    if abs((f - rel0) - kick_rel) <= 24 and (f - rel0) % 8 == 0 and len(sheet) < 6:
        sheet.append(cv2.resize(img, (426, 240)))
vw.release(); cap.release()
if sheet:
    rows = [np.hstack(sheet[i:i+3]) for i in range(0, len(sheet), 3)]
    mw = max(r.shape[1] for r in rows)
    rows = [np.pad(r, ((0,0),(0,mw-r.shape[1]),(0,0))) for r in rows]
    cv2.imwrite(OUT_SHEET, np.vstack(rows))
print("RENDER_DONE", OUT_MP4, OUT_SHEET)
