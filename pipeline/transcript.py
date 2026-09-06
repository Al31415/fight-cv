"""Stage 00 — English commentary transcript (faster-whisper).

Commentary is *timestamped evidence, not ground truth* (overlapping speech,
diarization are weak) — it feeds candidate generation and adjudication, never
authoritative counts. Word-level timestamps are on the source PTS clock.

    python 00_ingest/transcript.py <video> [--model large-v3] [--compute float16]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fightcv import contracts as C  # noqa: E402


def _resolve_model(model_name: str) -> str:
    """Download the CT2 model to a plain local dir (copy, no symlinks).

    Windows blocks HF's default cache symlinks without Developer Mode, so we
    snapshot_download with local_dir_use_symlinks=False into apps/fight_cv/_models.
    """
    if Path(model_name).exists():
        return model_name
    from huggingface_hub import snapshot_download
    repo = model_name if "/" in model_name else f"Systran/faster-whisper-{model_name}"
    local = C.APP_DIR / "_models" / repo.replace("/", "__")
    if not (local / "model.bin").exists():
        snapshot_download(repo, local_dir=str(local), local_dir_use_symlinks=False)
    return str(local)


def transcribe(video_path: str, model_name: str = "large-v3", compute: str = "int8_float16") -> pd.DataFrame:
    from faster_whisper import WhisperModel
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        compute = "int8"
    model = WhisperModel(_resolve_model(model_name), device=device, compute_type=compute)
    segments, info = model.transcribe(
        video_path, language="en", word_timestamps=True, vad_filter=True,
        beam_size=5, condition_on_previous_text=False,
    )
    rows = []
    for seg in segments:
        rows.append({
            "seg_id": seg.id, "start_s": round(seg.start, 3), "end_s": round(seg.end, 3),
            "text": seg.text.strip(), "avg_logprob": round(seg.avg_logprob, 3),
            "no_speech_prob": round(seg.no_speech_prob, 3),
            "words": [{"w": w.word, "t0": round(w.start, 3), "t1": round(w.end, 3),
                       "p": round(w.probability, 3)} for w in (seg.words or [])],
        })
    df = pd.DataFrame(rows)
    df.attrs["lang_prob"] = float(info.language_probability)
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--model", default="large-v3")
    ap.add_argument("--compute", default="int8_float16")
    args = ap.parse_args()

    prov = C.SourceProvenance.probe(args.video)
    if not prov.has_audio:
        print("[transcript] no audio stream; skipping")
        return
    run_id = C.new_run_id()
    df = transcribe(args.video, args.model, args.compute)
    # store word lists as JSON strings for parquet friendliness
    import json as _json
    df_out = df.copy()
    df_out["words"] = df_out["words"].apply(_json.dumps)
    out = C.VIDEO_DATA / prov.video_uid / "transcript.parquet"
    C.write_table(df_out, out, run_id=run_id, source=prov,
                  extra={"asr_model": args.model, "compute": args.compute,
                         "evidence_only": True})
    dur = df.end_s.max() if len(df) else 0.0
    print(f"[transcript] {len(df)} segments, {dur:.0f}s covered -> {out}")
    if len(df):
        print("  sample:", df.iloc[len(df)//2].text[:100])


if __name__ == "__main__":
    main()
