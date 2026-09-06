"""Freeze the pre-registration: validate PREREG.yaml parses, then record its
sha256 in PREREG.sha256.json BEFORE any model is evaluated against reference data.

Run once now (policy stage) and again in Phase 0a after `frozen_manifest.resolved`
is populated. Phase 1 must refuse to run unless the CURRENT file hash matches a
recorded `frozen` entry here.

    python apps/fight_cv/freeze_prereg.py            # record current hash
    python apps/fight_cv/freeze_prereg.py --check    # verify file unchanged since freeze
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PREREG = HERE / "PREREG.yaml"
LEDGER = HERE / "PREREG.sha256.json"


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_rev() -> str:
    try:
        return subprocess.run(["git", "-C", str(HERE), "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, timeout=10).stdout.strip() or "nogit"
    except Exception:
        return "nogit"


def load_and_validate() -> dict:
    text = PREREG.read_text(encoding="utf-8")
    try:
        import yaml
        doc = yaml.safe_load(text)
    except ModuleNotFoundError:
        print("[freeze] WARNING: pyyaml not installed; skipped structural validation (hash still recorded)")
        return {"version": "?", "status": "?"}
    assert isinstance(doc, dict) and "frozen_policy" in doc, "PREREG.yaml missing frozen_policy"
    g = doc["frozen_policy"]["pose_benchmark"]["gate"]
    assert g["primary"]["min_relative_reduction"] == 0.25
    assert g["guardrails"]["top_bottom_noninferiority"]["margin_Y_pp"] == 3.0
    assert doc["frozen_policy"]["power_requirement"]["min_independent_debut_fighters"] == 150
    return doc


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="verify file unchanged vs a recorded freeze")
    args = ap.parse_args()

    if not PREREG.exists():
        raise SystemExit(f"[freeze] {PREREG} not found")
    doc = load_and_validate()
    digest = sha256_file(PREREG)
    ledger = json.loads(LEDGER.read_text(encoding="utf-8")) if LEDGER.exists() else {"freezes": []}

    if args.check:
        frozen = {f["sha256"] for f in ledger["freezes"]}
        if digest in frozen:
            print(f"[freeze] OK — current PREREG.yaml matches a recorded freeze ({digest[:16]}...)")
        else:
            raise SystemExit(f"[freeze] MISMATCH — PREREG.yaml has changed since freeze; current={digest[:16]}... "
                             f"not in recorded freezes. Bump version + re-freeze on a NEW holdout.")
        return

    entry = {
        "sha256": digest,
        "prereg_version": doc.get("version", "?"),
        "status": doc.get("status", "?"),
        "recorded_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_rev": git_rev(),
    }
    if any(f["sha256"] == digest for f in ledger["freezes"]):
        print(f"[freeze] already recorded: {digest[:16]}... (no change)")
    else:
        ledger["freezes"].append(entry)
        LEDGER.write_text(json.dumps(ledger, indent=2), encoding="utf-8")
        print(f"[freeze] recorded freeze {digest[:16]}... version={entry['prereg_version']} "
              f"status={entry['status']}")
    print(f"[freeze] PREREG sha256 = {digest}")


if __name__ == "__main__":
    main()
