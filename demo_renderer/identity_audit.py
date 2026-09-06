"""Audit fighter identity in the trial (CFFC) clip via shorts-color.

Samples color patches around the shorts region (hips + upper thighs) for each
labeled fighter every frame, clusters all samples into the two true fighters,
then reports which label maps to which cluster over time -> swap intervals.
Writes identity_map.json used by the renderer.
"""
import json
import sys

sys.path.insert(0, r"d:\CFL Data\apps\fight_cv\demo")
import cv2
import numpy as np
import render_demo as rd

F0, F1 = 320, 779
tp = rd.load_trial_pose()
cap = cv2.VideoCapture(rd.TRIAL)
cap.set(cv2.CAP_PROP_POS_FRAMES, F0)

samples = {}  # frame -> {fid: feature}
for f in range(F0, F1):
    ok, img = cap.read()
    if not ok:
        break
    for fid, k in tp.get(f, {}).items():
        pts = []
        hips = k[[11, 12]]
        knees = k[[13, 14]]
        for p in list(hips) + list((hips + knees) / 2):
            if np.isfinite(p).all():
                x, y = int(p[0]), int(p[1])
                patch = img[max(0, y - 6):y + 6, max(0, x - 6):x + 6]
                if patch.size:
                    pts.append(patch.reshape(-1, 3).mean(axis=0))
        if pts:
            samples.setdefault(f, {})[fid] = np.mean(pts, axis=0)

# k-means (k=2) over all samples
X = np.array([v for d in samples.values() for v in d.values()], np.float32)
crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.1)
_, labels, centers = cv2.kmeans(X, 2, None, crit, 10, cv2.KMEANS_PP_CENTERS)
print("cluster centers (BGR):", centers.round(1).tolist())

def assign(feat):
    d = np.linalg.norm(centers - feat, axis=1)
    return int(np.argmin(d)), float(abs(d[0] - d[1]))

# per-frame raw mapping: which cluster does label-0 belong to?
raw = {}   # frame -> (cluster_of_label0, margin)
for f, d in samples.items():
    if 0 in d and 1 in d:
        c0, m0 = assign(d[0])
        c1, m1 = assign(d[1])
        if c0 != c1:
            raw[f] = (c0, m0 + m1)
    elif 0 in d:
        c0, m0 = assign(d[0])
        raw[f] = (c0, m0)
    elif 1 in d:
        c1, m1 = assign(d[1])
        raw[f] = (1 - c1, m1)

frames = sorted(raw)
vals = np.array([raw[f][0] for f in frames])
# median smoothing over 21 frames
sm = vals.copy()
for i in range(len(vals)):
    lo, hi = max(0, i - 10), min(len(vals), i + 11)
    sm[i] = 1 if vals[lo:hi].mean() > 0.5 else 0

# swap intervals (relative to label0==cluster of its first appearance)
base = sm[0]
print("label0 starts as cluster", base)
intervals = []
cur = sm[0]; start = frames[0]
for i in range(1, len(frames)):
    if sm[i] != cur:
        intervals.append((start, frames[i - 1], int(cur)))
        cur = sm[i]; start = frames[i]
intervals.append((start, frames[-1], int(cur)))
print("identity intervals (start, end, cluster_of_label0):")
for iv in intervals:
    print("  ", iv)

# Full-coverage map: every frame in [F0, F1) gets a flip value from its
# enclosing/nearest interval; frames in a gap BETWEEN intervals of different
# value are marked hidden (ambiguous transition).
id_map = {}
hidden = []
for f in range(F0, F1):
    before = [iv for iv in intervals if iv[0] <= f]
    after = [iv for iv in intervals if iv[0] > f]
    cur = before[-1] if before else intervals[0]
    if f <= cur[1]:
        id_map[f] = bool(cur[2] != base)
    else:  # in a gap after cur
        nxt = after[0] if after else None
        if nxt is None or nxt[2] == cur[2]:
            id_map[f] = bool(cur[2] != base)
        else:
            hidden.append(f)
with open(r"d:\CFL Data\apps\fight_cv\demo\identity_map.json", "w") as fh:
    json.dump({"flip": {str(k): v for k, v in id_map.items()},
               "hidden": hidden}, fh)
print("wrote identity_map.json;", sum(id_map.values()), "flipped /",
      len(id_map), "mapped;", len(hidden), "hidden:",
      hidden[:3], "...", hidden[-3:] if hidden else "")
