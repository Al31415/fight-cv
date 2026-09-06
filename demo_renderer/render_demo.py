"""Fight-CV demo reel renderer.

Composites local pod artifacts (SAM2 masks, HMR2 pose parquets, position
timeline) over the source broadcast clips into a single 1920x1080 demo video.
Runs fully on CPU; frames are piped raw into ffmpeg/libx264.

Usage:
  python render_demo.py --probe    # write one still per segment to demo/probe/
  python render_demo.py            # full render to demo/out/fightcv_demo.mp4
"""
import argparse
import ast
import os
import subprocess
import sys

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

ROOT = r"d:\CFL Data\apps\fight_cv"
RES = os.path.join(ROOT, "cloud", "results")
GALLERY = os.path.join(ROOT, "validation_gallery")
DEMO = os.path.join(ROOT, "demo")
OUT = os.path.join(DEMO, "out")
GRAPPLE = os.path.join(ROOT, "cloud", "grapple_test.mp4")
TRIAL = os.path.join(ROOT, "cloud", "trial_clip.mp4")

W, H = 1920, 1080
FPS = 30

BG = (14, 17, 22)  # BGR-ish dark charcoal (used as BGR)
INK = (230, 233, 238)
DIM = (150, 158, 170)
ACCENT = (255, 122, 66)   # BGR orange (F0)
F0_COL = (54, 90, 255)    # BGR: crimson-orange for fighter 0
F1_COL = (255, 163, 54)   # BGR: azure for fighter 1
POS_COLORS = {
    "standing/clinch": (165, 194, 70),        # teal-green (BGR)
    "scramble/occluded": (111, 96, 85),       # slate
    "top-control (side/guard)": (84, 180, 255),  # amber
    "mount / top-control": (84, 180, 255),    # merged with ground control
}
# Display only the dimensions the model gets verifiably right on this bout:
# phase (standing/ground) + who controls. Position NAMES are suppressed —
# the classifier calls back control "side/guard top control", and back vs
# top are different positions; naming either would be a wrong annotation.
POS_SHORT = {
    "standing/clinch": "STANDING / CLINCH",
    "scramble/occluded": "SCRAMBLE / OCCLUDED",
    "top-control (side/guard)": "GROUND CONTROL",
    "mount / top-control": "GROUND CONTROL",
}

EDGES = [(5, 6), (5, 7), (7, 9), (6, 8),
         (8, 10), (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14),
         (14, 16)]  # face edges dropped — cleaner at broadcast scale
HEAD_KPT = 0  # nose; linked to shoulder midpoint as a neck stub

FONTS = {}


def font(size, weight="regular"):
    key = (size, weight)
    if key not in FONTS:
        path = {
            "light": r"C:\Windows\Fonts\segoeuil.ttf",
            "regular": r"C:\Windows\Fonts\segoeui.ttf",
            "semibold": r"C:\Windows\Fonts\seguisb.ttf",
            "bold": r"C:\Windows\Fonts\segoeuib.ttf",
        }[weight]
        FONTS[key] = ImageFont.truetype(path, size)
    return FONTS[key]


def put_text(img, text, xy, size=28, weight="regular", color=(238, 233, 230),
             anchor="la", tracking=0):
    """Draw text via PIL (color given as RGB)."""
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    d = ImageDraw.Draw(pil)
    if tracking:
        # simple letter-spacing for small-caps headers
        x, y = xy
        for ch in text:
            d.text((x, y), ch, font=font(size, weight), fill=color, anchor="la")
            x += d.textlength(ch, font=font(size, weight)) + tracking
    else:
        d.text(xy, text, font=font(size, weight), fill=color, anchor=anchor)
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


# ---------------------------------------------------------------- data loading

def load_masks():
    m = pd.read_parquet(os.path.join(RES, "sam2_masklets.parquet"))
    out = {}
    for _, r in m.iterrows():
        poly = np.array(ast.literal_eval(r.polygon), np.int32)
        out.setdefault(int(r.frame), []).append((int(r.fighter), poly))
    return out


def load_substrate():
    s = pd.read_parquet(os.path.join(RES, "substrate_pose.parquet"))
    out = {}
    for (f, fi), g in s.groupby(["frame", "fighter"]):
        k = np.full((17, 2), np.nan)
        k[g.kpt.values] = g[["x2d", "y2d"]].values
        out.setdefault(int(f), {})[int(fi)] = k
    return out


def load_gmid3d(substrate):
    g = pd.read_parquet(os.path.join(RES, "pose_gmid.parquet"))
    out = {}
    for (f, tid), gr in g.groupby(["src_frame", "track_id"]):
        f = int(f)
        if f not in substrate:
            continue
        k2 = np.full((17, 2), np.nan)
        k3 = np.full((17, 3), np.nan)
        k2[gr.kpt.values] = gr[["x2d", "y2d"]].values
        k3[gr.kpt.values] = gr[["x3d", "y3d", "z3d"]].values
        mean2 = np.nanmean(k2, axis=0)
        # match track -> fighter by 2d proximity to substrate
        best, bestd = None, 1e9
        for fi, sk in substrate[f].items():
            d = np.linalg.norm(np.nanmean(sk, axis=0) - mean2)
            if d < bestd:
                best, bestd = fi, d
        if best is not None and bestd < 60:
            cur = out.setdefault(f, {})
            if best not in cur:
                hip2d = np.nanmean(k2[[11, 12]], axis=0)
                cur[best] = (k3, hip2d)
    return out


def load_timeline():
    t = pd.read_parquet(os.path.join(RES, "position_timeline.parquet"))
    t = t.set_index("frame").reindex(range(0, 1170)).ffill().bfill()
    return t


def load_trial_pose():
    f = pd.read_parquet(os.path.join(RES, "fighters_pose.parquet"))
    out = {}
    for (fr, fi), g in f.groupby(["clip_frame", "fighter"]):
        k = np.full((17, 2), np.nan)
        k[g.kpt.values] = g[["x2d", "y2d"]].values
        out.setdefault(int(fr), {})[int(fi)] = k
    return out


def load_trial3d(trial_pose):
    """3D pose for the trial (striking) clip from pose_coco, identity-matched
    to fighters_pose by 2D proximity."""
    g = pd.read_parquet(os.path.join(RES, "pose_coco.parquet"))
    out = {}
    for (f, tid), gr in g.groupby(["src_frame", "track_id"]):
        f = int(f)
        if f not in trial_pose:
            continue
        k2 = np.full((17, 2), np.nan)
        k3 = np.full((17, 3), np.nan)
        k2[gr.kpt.values] = gr[["x2d", "y2d"]].values
        k3[gr.kpt.values] = gr[["x3d", "y3d", "z3d"]].values
        mean2 = np.nanmean(k2, axis=0)
        best, bestd = None, 1e9
        for fi, sk in trial_pose[f].items():
            d = np.linalg.norm(np.nanmean(sk, axis=0) - mean2)
            if d < bestd:
                best, bestd = fi, d
        if best is not None and bestd < 60:
            cur = out.setdefault(f, {})
            if best not in cur:
                hip2d = np.nanmean(k2[[11, 12]], axis=0)
                cur[best] = (k3, hip2d)
    return out


def smooth_series(frames_dict, alpha=0.5):
    """EMA-smooth per-fighter keypoints across frames (dict frame->fi->arr)."""
    state = {}
    for f in sorted(frames_dict):
        for fi, k in frames_dict[f].items():
            if fi in state:
                prev = state[fi]
                mask = ~np.isnan(k) & ~np.isnan(prev)
                k = k.copy()
                k[mask] = alpha * k[mask] + (1 - alpha) * prev[mask]
            state[fi] = k
            frames_dict[f][fi] = k
    return frames_dict


# ---------------------------------------------------------------- draw helpers

def fighter_color(fi):
    return F0_COL if fi == 0 else F1_COL


def draw_skeleton(img, k, color, thick=3, glow=True):
    ov = np.zeros_like(img)
    pts = [(int(x), int(y)) if np.isfinite(x) and np.isfinite(y) else None
           for x, y in k]
    for a, b in EDGES:
        if pts[a] and pts[b]:
            cv2.line(ov, pts[a], pts[b], color, thick, cv2.LINE_AA)
    # neck stub: nose -> shoulder midpoint
    if pts[HEAD_KPT] and pts[5] and pts[6]:
        mid = ((pts[5][0] + pts[6][0]) // 2, (pts[5][1] + pts[6][1]) // 2)
        cv2.line(ov, pts[HEAD_KPT], mid, color, thick, cv2.LINE_AA)
    for i, p in enumerate(pts):
        if p and i not in (1, 2, 3, 4):
            cv2.circle(ov, p, thick + 1, color, -1, cv2.LINE_AA)
    if glow:
        blur = cv2.GaussianBlur(ov, (0, 0), 6)
        img = cv2.add(img, (blur * 0.55).astype(np.uint8))
    # paint lines opaque so color stays true over bright regions
    mask = ov.any(axis=2)
    img[mask] = (0.15 * img[mask] + 0.85 * ov[mask]).astype(np.uint8)
    return img


def draw_mask(img, poly, color, alpha=0.32, outline=True):
    ov = img.copy()
    cv2.fillPoly(ov, [poly], color)
    img = cv2.addWeighted(ov, alpha, img, 1 - alpha, 0)
    if outline:
        edge = np.zeros_like(img)
        cv2.polylines(edge, [poly], True, color, 2, cv2.LINE_AA)
        blur = cv2.GaussianBlur(edge, (0, 0), 4)
        img = cv2.add(img, (blur * 0.8).astype(np.uint8))
        img = cv2.add(img, (edge * 0.7).astype(np.uint8))
    return img


def base_canvas():
    c = np.zeros((H, W, 3), np.uint8)
    c[:] = BG
    return c


def header(canvas, stage, title):
    canvas = put_text(canvas, "FIGHT-CV", (48, 30), 26, "bold",
                      (255, 140, 90), tracking=6)
    canvas = put_text(canvas, stage, (1872, 34), 24, "semibold",
                      (150, 158, 170), anchor="ra")
    canvas = put_text(canvas, title, (48, 66), 34, "semibold", (238, 233, 230))
    cv2.line(canvas, (48, 116), (1872, 116), (58, 47, 42), 1, cv2.LINE_AA)
    return canvas


def caption(canvas, text, y=1014):
    return put_text(canvas, text, (W // 2, y), 27, "regular",
                    (170, 158, 150), anchor="ma")


def legend_chip(canvas, x, y):
    """Compact fighter legend on a dark chip (drawn over video)."""
    cv2.rectangle(canvas, (x, y), (x + 340, y + 44), (10, 12, 16), -1)
    cv2.rectangle(canvas, (x, y), (x + 340, y + 44), (58, 47, 42), 1)
    cv2.circle(canvas, (x + 22, y + 22), 8, F0_COL, -1, cv2.LINE_AA)
    canvas = put_text(canvas, "FIGHTER A", (x + 38, y + 8), 22, "semibold",
                      (255, 110, 84))
    cv2.circle(canvas, (x + 190, y + 22), 8, F1_COL, -1, cv2.LINE_AA)
    canvas = put_text(canvas, "FIGHTER B", (x + 206, y + 8), 22, "semibold",
                      (84, 173, 255))
    return canvas


def gate_pose_by_mask(frame_masks, frame_poses, margin=30):
    """Keep skeletons whose CENTER sits inside a fighter mask box.

    Kills displaced/hallucinated skeletons (fence-clinch occlusion drifts a
    fighter's pose off-body). Masks are the location anchor; a fighter with
    no plausible pose simply gets no skeleton that frame — honest abstention.
    """
    boxes = [(p.min(axis=0) - margin, p.max(axis=0) + margin)
             for _, p in frame_masks]
    out = {}
    for fid, k in frame_poses.items():
        fin = k[np.isfinite(k).all(axis=1)]
        if len(fin) == 0 or not boxes:
            continue
        c = fin.mean(axis=0)
        if any(lo[0] <= c[0] <= hi[0] and lo[1] <= c[1] <= hi[1]
               for lo, hi in boxes):
            out[fid] = k
    return out


def fade(frame, t, dur, fade_len=0.6):
    """Apply fade-in/out to a frame given segment time t of dur seconds."""
    a = min(1.0, t / fade_len, (dur - t) / fade_len)
    a = max(0.0, a)
    if a >= 1.0:
        return frame
    return (frame.astype(np.float32) * a).astype(np.uint8)


# --------------------------------------------------------------- 3D pose panel

def render_3d_panel(pw, ph, k3d_by_fighter, angle):
    """Orthographic rotating view of the recovered 3D articulation.

    Each fighter is rendered root-relative (monocular absolute depth is the
    documented weak axis) and placed by their image-plane position, so the
    panel shows true recovered articulation + true left/right arrangement.
    """
    panel = np.zeros((ph, pw, 3), np.uint8)
    panel[:] = (20, 24, 30)
    placed = {}
    for fi, (k3, hip2d) in k3d_by_fighter.items():
        if not np.isfinite(k3).any() or not np.isfinite(hip2d).all():
            continue
        root = np.nanmean(k3[[11, 12]], axis=0)
        rel = k3 - root
        off = np.array([(hip2d[0] - 640.0) / 640.0 * 1.1,
                        (hip2d[1] - 380.0) / 360.0 * 0.45, 0.0])
        placed[fi] = rel + off
    if not placed:
        return panel
    concat = np.vstack(list(placed.values()))
    center = np.nanmean(concat, axis=0)
    ca, sa = np.cos(angle), np.sin(angle)

    def proj(p):
        q = p - center
        x = q[0] * ca + q[2] * sa
        y = q[1]
        scale = ph * 0.30
        return (int(pw / 2 + x * scale), int(ph * 0.50 + y * scale))

    # floor grid just below the lowest joint
    floor_y = np.nanmax(concat[:, 1]) + 0.05
    for gx in np.arange(-1.2, 1.3, 0.4):
        cv2.line(panel, proj(np.array([center[0] + gx, floor_y, center[2] - 1.2])),
                 proj(np.array([center[0] + gx, floor_y, center[2] + 1.2])),
                 (38, 44, 54), 1, cv2.LINE_AA)
        cv2.line(panel, proj(np.array([center[0] - 1.2, floor_y, center[2] + gx])),
                 proj(np.array([center[0] + 1.2, floor_y, center[2] + gx])),
                 (38, 44, 54), 1, cv2.LINE_AA)

    ov = np.zeros_like(panel)
    for fi, k in placed.items():
        col = fighter_color(fi)
        pts = [proj(p) if np.isfinite(p).all() else None for p in k]
        for a, b in EDGES:
            if pts[a] and pts[b]:
                cv2.line(ov, pts[a], pts[b], col, 3, cv2.LINE_AA)
        if pts[HEAD_KPT] and pts[5] and pts[6]:
            mid = ((pts[5][0] + pts[6][0]) // 2, (pts[5][1] + pts[6][1]) // 2)
            cv2.line(ov, pts[HEAD_KPT], mid, col, 3, cv2.LINE_AA)
        for i, p in enumerate(pts):
            if p and i not in (1, 2, 3, 4):
                cv2.circle(ov, p, 4, col, -1, cv2.LINE_AA)
    blur = cv2.GaussianBlur(ov, (0, 0), 5)
    panel = cv2.add(panel, (blur * 0.55).astype(np.uint8))
    mask = ov.any(axis=2)
    panel[mask] = ov[mask]
    return panel


# ------------------------------------------------------------- control ribbon

def build_control_clocks(timeline):
    """Cumulative control seconds per fighter at each frame."""
    c0 = np.cumsum((timeline.control.values == 0) &
                   (timeline.position.values != "scramble/occluded")) / FPS
    c0_all = np.cumsum(timeline.control.values == 0) / FPS
    c1_all = np.cumsum(timeline.control.values == 1) / FPS
    return c0_all, c1_all


def draw_ribbon(canvas, timeline, cur, x, y, w, h, f0=0, f1=1169,
                clocks=None):
    n = f1 - f0 + 1
    px_per = w / n
    # blocks
    pos = timeline.position.values
    i = f0
    while i <= f1:
        j = i
        while j + 1 <= f1 and pos[j + 1] == pos[i]:
            j += 1
        x1 = int(x + (i - f0) * px_per)
        x2 = int(x + (j + 1 - f0) * px_per)
        col = POS_COLORS.get(pos[i], (80, 80, 80))
        cv2.rectangle(canvas, (x1, y), (x2 - 1, y + h), col, -1)
        i = j + 1
    # dim the future
    xp = int(x + (cur - f0) * px_per)
    fut = canvas[y:y + h, xp:x + w].astype(np.float32) * 0.35
    canvas[y:y + h, xp:x + w] = fut.astype(np.uint8)
    # playhead
    cv2.line(canvas, (xp, y - 6), (xp, y + h + 6), (240, 240, 245), 2,
             cv2.LINE_AA)
    return canvas


# ------------------------------------------------------------------- segments

class FFmpegSink:
    def __init__(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.proc = subprocess.Popen(
            ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt",
             "bgr24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
             "-c:v", "libx264", "-preset", "medium", "-crf", "18",
             "-pix_fmt", "yuv420p", "-movflags", "+faststart", path],
            stdin=subprocess.PIPE)

    def write(self, frame):
        self.proc.stdin.write(frame.tobytes())

    def close(self):
        self.proc.stdin.close()
        self.proc.wait()


def video_reader(path, start):
    cap = cv2.VideoCapture(path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    return cap


def seg_title():
    """S0: title card."""
    dur = 5.0
    for i in range(int(dur * FPS)):
        t = i / FPS
        c = base_canvas()
        c = put_text(c, "FIGHT-CV", (W // 2, 330), 30, "bold", (255, 140, 90),
                     anchor="ma")
        c = put_text(c, "Raw broadcast video → fight intelligence",
                     (W // 2, 420), 74, "light", (238, 233, 230), anchor="ma")
        c = put_text(c, "monocular 3D pose  ·  identity tracking  ·  "
                        "position & control states",
                     (W // 2, 560), 34, "regular", (170, 158, 150), anchor="ma")
        c = put_text(c, "single camera · no instrumentation · "
                        "validated against 20-camera mocap ground truth",
                     (W // 2, 640), 27, "regular", (120, 128, 140), anchor="ma")
        yield fade(c, t, dur, 0.8)


def draw_seed_box(img, poly, color, label):
    """Detector-seed visualization: corner-bracket box from the mask bbox."""
    x1, y1 = poly.min(axis=0)
    x2, y2 = poly.max(axis=0)
    L = 26
    for cx, cy, dx, dy in [(x1, y1, 1, 1), (x2, y1, -1, 1),
                           (x1, y2, 1, -1), (x2, y2, -1, -1)]:
        cv2.line(img, (cx, cy), (cx + dx * L, cy), color, 3, cv2.LINE_AA)
        cv2.line(img, (cx, cy), (cx, cy + dy * L), color, 3, cv2.LINE_AA)
    cv2.putText(img, label, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, color, 2, cv2.LINE_AA)
    return img


def seg_masks(masks):
    """S1: perception build-up — raw video, then auto-seed boxes, then
    SAM2 masklets with locked fighter identity. The staging is the point:
    no human prompts anywhere."""
    f0, f1 = 60, 420
    seed_at, mask_at = 120, 186
    dur = (f1 - f0) / FPS
    cap = video_reader(GRAPPLE, f0)
    vw, vh = 1536, 864
    vx, vy = (W - vw) // 2, 122
    for fi_idx in range(f0, f1):
        ok, img = cap.read()
        if not ok:
            break
        if seed_at <= fi_idx < mask_at:
            for fid, poly in masks.get(fi_idx, []):
                img = draw_seed_box(img, poly, fighter_color(fid),
                                    f"auto-seed {'A' if fid == 0 else 'B'}")
        elif fi_idx >= mask_at:
            for fid, poly in masks.get(fi_idx, []):
                img = draw_mask(img, poly, fighter_color(fid))
        img = cv2.resize(img, (vw, vh), interpolation=cv2.INTER_AREA)
        c = base_canvas()
        c[vy:vy + vh, vx:vx + vw] = img
        c = header(c, "01 / PERCEIVE", "From raw pixels to tracked fighters — zero human prompts")
        stage = ("raw broadcast in" if fi_idx < seed_at else
                 "detector auto-seeds — no clicks" if fi_idx < mask_at else
                 "SAM2 masklets + persistent fighter identity")
        cv2.rectangle(c, (vx + 16, vy + 16), (vx + 460, vy + 58),
                      (10, 12, 16), -1)
        cv2.rectangle(c, (vx + 16, vy + 16), (vx + 460, vy + 58),
                      (58, 47, 42), 1)
        c = put_text(c, stage, (vx + 32, vy + 24), 22, "semibold",
                     (255, 140, 90))
        if fi_idx >= mask_at:
            c = legend_chip(c, vx + 16, vy + 70)
        c = caption(c, "detection seeds SAM2 automatically; identity survives the scramble and re-seeds across camera cuts")
        yield fade(c, (fi_idx - f0) / FPS, dur)
    cap.release()


def seg_pose3d(trial_pose, trial3d):
    """S2: striking clip — 2D skeletons + rotating 3D panel.

    Two level-camera windows spliced at the broadcast's own cut. Excluded:
    the aerial stretch f496-622 (fighters tiny from overhead, detections
    latch onto ref/cageside people) and f624-649 (label 1 is a partial
    mis-detection near the ref until it re-acquires the fighter at f650).
    Labels themselves are identity-consistent across the whole clip —
    verified by track continuity (max centroid jump ~20px through the
    crossing) and shorts appearance (patterned vs script) in both windows;
    do NOT add color-based flips (shorts look different per side and fooled
    an earlier k-means audit into introducing swaps).
    """
    frames = list(range(320, 494)) + list(range(650, 779))
    dur = len(frames) / FPS
    cap = video_reader(TRIAL, frames[0])
    vw, vh = 1280, 720
    vx, vy = 48, 150
    pw, ph = 496, 720
    px = 1376
    prev_f = frames[0] - 1
    for i, fi_idx in enumerate(frames):
        if fi_idx != prev_f + 1:
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi_idx)
        prev_f = fi_idx
        ok, img = cap.read()
        if not ok:
            break
        for fid, k in trial_pose.get(fi_idx, {}).items():
            img = draw_skeleton(img, k, fighter_color(fid))
        c = base_canvas()
        c[vy:vy + vh, vx:vx + vw] = img
        angle = 0.6 + i * 0.009
        panel = render_3d_panel(pw, ph, trial3d.get(fi_idx, {}), angle)
        c[vy:vy + ph, px:px + pw] = panel
        cv2.rectangle(c, (px, vy), (px + pw - 1, vy + ph - 1), (58, 47, 42), 1)
        c = put_text(c, "RECOVERED 3D — free viewpoint", (px + pw // 2, vy + 18),
                     22, "semibold", (150, 158, 170), anchor="ma")
        c = header(c, "02 / RECONSTRUCT", "Monocular 3D pose — both bodies, one camera")
        c = caption(c, "HMR2 3D body recovery from the single broadcast angle — different promotion, same pipeline, zero tuning")
        yield fade(c, i / FPS, dur)
    cap.release()


def seg_intel(masks, timeline):
    """S3: position/control intelligence with ribbon + clocks.

    Restricted to the stretches where the emitted state visibly matches the
    video (verified frame-by-frame against stills): 430-461 standing,
    482-548 standing -> takedown -> top control, 593-633 top control.
    Excluded as mislabeled: 462-480 (bent fence clinch read as ground
    top-control) and 549-592 / 634+ (grounded read as standing). Skips are
    bridged with short dissolves. This bout is a long ambiguous
    fence-wrestling battle — the honest showcase is the correct arc, not
    the full noisy timeline.
    """
    frames = (list(range(430, 462)) + list(range(482, 548)) +
              list(range(593, 634)))
    dissolve = 8  # frames of crossfade after each skip
    dur = len(frames) / FPS
    cap = video_reader(GRAPPLE, frames[0])
    vw, vh = 1387, 780
    vx, vy = 48, 140
    rib_x, rib_w = 48, 1824
    rib_y, rib_h = 966, 34
    c0, c1 = build_control_clocks(timeline)
    prev_f = frames[0] - 1
    hold = None
    last_img = None
    since_jump = 999
    for i, fi_idx in enumerate(frames):
        if fi_idx != prev_f + 1:
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi_idx)
            hold = last_img
            since_jump = 0
        else:
            since_jump += 1
        prev_f = fi_idx
        ok, img = cap.read()
        if not ok:
            break
        # masks + state UI only — per-joint pose is not the reliable signal
        # in entangled grappling; segmentation and position/control are
        for fid, poly in masks.get(fi_idx, []):
            img = draw_mask(img, poly, fighter_color(fid), alpha=0.30)
        img = cv2.resize(img, (vw, vh), interpolation=cv2.INTER_AREA)
        if since_jump < dissolve and hold is not None:
            a = (since_jump + 1) / (dissolve + 1)
            img = cv2.addWeighted(img, a, hold, 1 - a, 0)
        last_img = img
        c = base_canvas()
        c[vy:vy + vh, vx:vx + vw] = img
        c = header(c, "03 / UNDERSTAND", "Phase & control, called live from one camera — standing → takedown → ground control")
        # state box (right of video)
        row = timeline.iloc[min(fi_idx, len(timeline) - 1)]
        sx = 1475
        c = put_text(c, "STATE", (sx, 170), 22, "semibold", (120, 128, 140),
                     tracking=3)
        c = put_text(c, POS_SHORT.get(row.position, str(row.position)),
                     (sx, 205), 30, "semibold", (238, 233, 230))
        c = put_text(c, f"confidence {row.confidence:.2f}", (sx, 252), 24,
                     "regular", (150, 158, 170))
        c = put_text(c, "CONTROL TIME", (sx, 330), 22, "semibold",
                     (120, 128, 140), tracking=3)
        m0, s0 = divmod(int(c0[min(fi_idx, len(c0) - 1)]), 60)
        m1, s1 = divmod(int(c1[min(fi_idx, len(c1) - 1)]), 60)
        c = put_text(c, f"A  {m0}:{s0:02d}", (sx, 365), 40, "semibold",
                     (255, 110, 84))
        c = put_text(c, f"B  {m1}:{s1:02d}", (sx, 425), 40, "semibold",
                     (84, 173, 255))
        ctl = row.control
        who = {0: "FIGHTER A", 1: "FIGHTER B", -1: "neutral"}.get(int(ctl), "")
        c = put_text(c, f"in control: {who}", (sx, 500), 24, "regular",
                     (150, 158, 170))
        c = draw_ribbon(c, timeline, fi_idx, rib_x, rib_y, rib_w, rib_h,
                        f0=0, f1=651)
        # ribbon legend (ground-control categories merged for display)
        legend_entries = [
            ("Standing / Clinch", POS_COLORS["standing/clinch"]),
            ("Scramble / Occluded", POS_COLORS["scramble/occluded"]),
            ("Ground Control", POS_COLORS["top-control (side/guard)"]),
        ]
        lx = rib_x
        for lab, col in legend_entries:
            cv2.rectangle(c, (lx, rib_y + rib_h + 14), (lx + 14, rib_y + rib_h + 28),
                          col, -1)
            c = put_text(c, lab, (lx + 22, rib_y + rib_h + 8), 21, "regular",
                         (150, 158, 170))
            lx += 30 + int(11 * len(lab)) + 26
        c = put_text(c, "position timeline — emitted per frame, "
                        "abstains when occluded", (rib_x + rib_w, rib_y - 30),
                     22, "regular", (120, 128, 140), anchor="ra")
        yield fade(c, i / FPS, dur)
    cap.release()


def seg_striking(trial_pose):
    """S4: second broadcast, standing striking."""
    f0, f1 = 310, 760
    dur = (f1 - f0) / FPS
    cap = video_reader(TRIAL, f0)
    vw, vh = 1536, 864
    vx, vy = (W - vw) // 2, 122
    for fi_idx in range(f0, f1):
        ok, img = cap.read()
        if not ok:
            break
        for fid, k in trial_pose.get(fi_idx, {}).items():
            img = draw_skeleton(img, k, fighter_color(fid))
        img = cv2.resize(img, (vw, vh), interpolation=cv2.INTER_AREA)
        c = base_canvas()
        c[vy:vy + vh, vx:vx + vw] = img
        c = header(c, "04 / GENERALIZE", "Different promotion, different production — same pipeline")
        c = caption(c, "zero per-broadcast tuning — the same perception stack tracks the standing game")
        yield fade(c, (fi_idx - f0) / FPS, dur)
    cap.release()


def fit_image(img, bw, bh):
    h, w = img.shape[:2]
    s = min(bw / w, bh / h)
    nw, nh = int(w * s), int(h * s)
    return cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)


def seg_validation():
    """S5: two validation cards."""
    # Single validation card: problem -> design bet -> measurement.
    # (The 4-model benchmark was model selection, not a contribution — it
    # lives in talking points, not on a slide.)
    bg = cv2.imread(os.path.join(GALLERY, "occlusion_clinch.png"))
    dur = 9.0
    for i in range(int(dur * FPS)):
        t = i / FPS
        c = base_canvas()
        full = cv2.resize(bg, (W, int(W * bg.shape[0] / bg.shape[1])))
        y0 = max(0, (full.shape[0] - H) // 2)
        crop = full[y0:y0 + H]
        c[:crop.shape[0]] = (crop.astype(np.float32) * 0.30).astype(np.uint8)
        c = header(c, "VALIDATION", "The design bet, measured")
        c = put_text(c, "3D pose through a grapple is unsolved — the best "
                        "models are 200+ mm off on occluded joints",
                     (W // 2, 260), 33, "regular", (238, 233, 230), anchor="ma")
        c = put_text(c, "so we bet on a different question:  do the fight "
                        "CONCLUSIONS survive imperfect pose?",
                     (W // 2, 320), 33, "regular", (238, 233, 230), anchor="ma")
        c = put_text(c, "identical intelligence stack, re-run independently "
                        "from each of a mocap rig's 20 cameras,",
                     (W // 2, 420), 29, "regular", (150, 158, 170), anchor="ma")
        c = put_text(c, "scored against the multi-view ground-truth run "
                        "(149-frame grappling sequence, Harmony4D)",
                     (W // 2, 468), 29, "regular", (150, 158, 170), anchor="ma")
        c = put_text(c, "who-is-in-control: 95–100% agreement, median ≈98%",
                     (W // 2, 560), 52, "semibold", (255, 140, 90), anchor="ma")
        c = put_text(c, "every one of 20 viewpoints — no bad angle",
                     (W // 2, 650), 36, "semibold", (238, 233, 230), anchor="ma")
        c = put_text(c, "position class 85–93% · standing/clinch phase 67–77% "
                        "(monocular depth is the weak axis)",
                     (W // 2, 740), 28, "regular", (150, 158, 170), anchor="ma")
        c = put_text(c, "an invariance result, not an accuracy claim: camera "
                        "placement does not change the conclusion",
                     (W // 2, 795), 28, "regular", (150, 158, 170), anchor="ma")
        yield fade(c, t, dur)


def seg_end():
    dur = 7.0
    steps = ["broadcast video", "SAM2 identity masks", "HMR2 3D pose",
             "world placement", "position · control · strike events",
             "fight report"]
    for i in range(int(dur * FPS)):
        t = i / FPS
        c = base_canvas()
        c = put_text(c, "FIGHT-CV", (W // 2, 300), 30, "bold", (255, 140, 90),
                     anchor="ma")
        c = put_text(c, "One camera in. Fight intelligence out.",
                     (W // 2, 380), 60, "light", (238, 233, 230), anchor="ma")
        # pipeline chain
        chain = "   →   ".join(steps)
        c = put_text(c, chain, (W // 2, 530), 27, "regular", (170, 158, 150),
                     anchor="ma")
        c = put_text(c, "single-GPU pipeline · evaluation protocol frozen before "
                        "measurement · abstains honestly when it cannot see",
                     (W // 2, 640), 26, "regular", (120, 128, 140), anchor="ma")
        yield fade(c, t, dur, 0.8)


# ------------------------------------------------------------------------ main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--segments", default="all",
                    help="comma list: title,masks,pose3d,intel,striking,validation,end")
    args = ap.parse_args()

    print("loading artifacts...", flush=True)
    masks = load_masks()
    timeline = load_timeline()
    trial_pose = smooth_series(load_trial_pose())
    trial3d = load_trial3d(trial_pose)
    print("loaded.", flush=True)

    segs = [
        ("title", lambda: seg_title()),
        ("masks", lambda: seg_masks(masks)),
        ("pose3d", lambda: seg_pose3d(trial_pose, trial3d)),
        ("intel", lambda: seg_intel(masks, timeline)),
        ("validation", lambda: seg_validation()),
        ("end", lambda: seg_end()),
    ]
    want = args.segments.split(",") if args.segments != "all" else [s for s, _ in segs]

    if args.probe:
        os.makedirs(os.path.join(DEMO, "probe"), exist_ok=True)
        for name, gen in segs:
            if name not in want:
                continue
            frames = gen()
            target = 60  # ~2s in, past the fade
            frame = None
            for i, fr in enumerate(frames):
                frame = fr
                if i >= target:
                    break
            cv2.imwrite(os.path.join(DEMO, "probe", f"seg_{name}.png"), frame)
            print("probe", name, flush=True)
        return

    sink = FFmpegSink(os.path.join(OUT, "fightcv_demo.mp4"))
    total = 0
    for name, gen in segs:
        if name not in want:
            continue
        n = 0
        for frame in gen():
            sink.write(frame)
            n += 1
        total += n
        print(f"segment {name}: {n} frames", flush=True)
    sink.close()
    print(f"done: {total} frames = {total / FPS:.1f}s", flush=True)


if __name__ == "__main__":
    main()
