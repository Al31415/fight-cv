"""Stage 00 — ingest orchestrator.

Runs the Stage-00 components for one source video and writes an
`ingest_manifest.json` capturing provenance + per-component artifacts +
gate-relevant coverage (shots, transcript coverage, score-bug availability).
Heavy components are opt-in so slices can be assembled incrementally.

    python 00_ingest/run_ingest.py <video> [--shots] [--transcript] [--threshold 27]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fightcv import contracts as C  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--shots", action="store_true")
    ap.add_argument("--transcript", action="store_true")
    ap.add_argument("--threshold", type=float, default=27.0)
    ap.add_argument("--downscale", type=int, default=2)
    args = ap.parse_args()

    prov = C.SourceProvenance.probe(args.video)
    run_id = C.new_run_id()
    outdir = C.VIDEO_DATA / prov.video_uid
    outdir.mkdir(parents=True, exist_ok=True)
    manifest = {"run_id": run_id, "video_uid": prov.video_uid,
                "source": prov.__dict__, "components": {}, "timing_s": {}}

    if args.shots:
        from shots import detect_shots
        t = time.time()
        df = detect_shots(args.video, args.threshold, args.downscale)
        C.write_table(df, outdir / "shots.parquet", run_id=run_id, source=prov,
                      extra={"threshold": args.threshold, "downscale": args.downscale})
        manifest["components"]["shots"] = {
            "n_shots": int(len(df)), "median_shot_s": float(df.dur_s.median()),
            "artifact": "shots.parquet"}
        manifest["timing_s"]["shots"] = round(time.time() - t, 1)

    if args.transcript and prov.has_audio:
        from transcript import transcribe
        import json as _json
        t = time.time()
        df = transcribe(args.video)
        cov = float(df.end_s.max()) if len(df) else 0.0
        df_out = df.copy(); df_out["words"] = df_out["words"].apply(_json.dumps)
        C.write_table(df_out, outdir / "transcript.parquet", run_id=run_id, source=prov,
                      extra={"asr_model": "large-v3", "evidence_only": True})
        manifest["components"]["transcript"] = {
            "n_segments": int(len(df)), "coverage_s": round(cov, 1),
            "coverage_frac": round(cov / prov.duration_s, 3) if prov.duration_s else 0.0,
            "artifact": "transcript.parquet"}
        manifest["timing_s"]["transcript"] = round(time.time() - t, 1)

    # score-bug availability is recorded as a gate fact, not assumed present
    manifest["components"].setdefault("score_bug", {
        "status": "uncalibrated",
        "note": "no cv_regions_ufc_*.json for this template; many regional "
                "broadcasts (e.g. CFFC) show no persistent round/clock timer — "
                "clock/round must come from round-card/audio/commentary/manual"})

    (outdir / "ingest_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[ingest] {prov.video_uid} -> {outdir/'ingest_manifest.json'}")
    print(json.dumps({k: v for k, v in manifest["components"].items()}, indent=2)[:800])


if __name__ == "__main__":
    main()
