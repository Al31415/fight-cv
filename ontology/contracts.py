"""Artifact & provenance contracts for the fight-CV pipeline.

Everything the review feedback demanded as first-class: content-hash provenance,
a source-PTS time base, ids that are explicitly distinct from the UFC data model,
run/checkpoint stamping, and a parquet+sidecar-metadata writer. Import these
helpers rather than re-deriving ids or writing bare parquet anywhere in the
pipeline.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from .ontology import VERSION as ONTOLOGY_VERSION

ROOT = Path(__file__).resolve().parents[3]          # d:\CFL Data
VIDEO_DATA = ROOT / "data" / "ufc" / "video"        # artifact root
APP_DIR = Path(__file__).resolve().parents[1]       # apps/fight_cv


# ── ids (kept explicitly distinct from UFC `fight_id`) ────────────────────────

def sha256_file(path: str | Path, _bufsize: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while chunk := fh.read(_bufsize):
            h.update(chunk)
    return h.hexdigest()


def video_uid(video_path: str | Path) -> str:
    """Stable id for a source video file: sha256[:16] of its bytes."""
    return sha256_file(video_path)[:16]


def video_fight_uid(video_uid_: str, segment_index: int) -> str:
    """Id for one bout segmented out of a (possibly full-card) video.

    Deliberately NOT the UFC `fight_id`; joins to the UFC model happen later via
    an explicit identity_map, never by key collision.
    """
    return f"vf_{video_uid_}_{segment_index:03d}"


def new_run_id() -> str:
    """Monotonic-ish, unique run id: <utc-compact>_<short-uuid>."""
    return f"{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}_{uuid.uuid4().hex[:8]}"


def git_rev() -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=10,
        ).stdout.strip() or "nogit"
    except Exception:
        return "nogit"


# ── source provenance (ffprobe + hashes + PTS time base) ──────────────────────

@dataclass
class SourceProvenance:
    video_path: str
    source_video_sha256: str
    video_uid: str
    width: int
    height: int
    duration_s: float
    avg_fps: float
    video_time_base: str          # canonical PTS time base, e.g. "1/15360"
    codec: str
    has_audio: bool

    @classmethod
    def probe(cls, video_path: str | Path) -> "SourceProvenance":
        p = str(video_path)
        info = json.loads(subprocess.run(
            ["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", p],
            capture_output=True, text=True, timeout=120,
        ).stdout or "{}")
        streams = info.get("streams", [])
        v = next((s for s in streams if s.get("codec_type") == "video"), {})
        has_audio = any(s.get("codec_type") == "audio" for s in streams)
        rate = v.get("avg_frame_rate", "0/0")
        fps = 0.0
        if "/" in rate:
            n, d = rate.split("/"); fps = float(n) / float(d) if float(d) else 0.0
        dur = info.get("format", {}).get("duration") or v.get("duration") or 0.0
        return cls(
            video_path=p, source_video_sha256=sha256_file(p), video_uid=video_uid(p),
            width=int(v.get("width", 0)), height=int(v.get("height", 0)),
            duration_s=float(dur), avg_fps=fps, video_time_base=v.get("time_base", ""),
            codec=v.get("codec_name", ""), has_audio=has_audio,
        )


# ── parquet + sidecar metadata writer ─────────────────────────────────────────

def write_table(df: pd.DataFrame, path: str | Path, *, run_id: str,
                source: SourceProvenance | None = None, extra: dict[str, Any] | None = None) -> Path:
    """Write parquet + a `<path>.metadata.json` sidecar with full provenance."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    meta: dict[str, Any] = {
        "rows": int(len(df)),
        "columns": list(df.columns),
        "run_id": run_id,
        "git_rev": git_rev(),
        "ontology_version": ONTOLOGY_VERSION,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    if source is not None:
        meta["source"] = asdict(source)
    if extra:
        meta["extra"] = extra
    Path(str(path) + ".metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return path
