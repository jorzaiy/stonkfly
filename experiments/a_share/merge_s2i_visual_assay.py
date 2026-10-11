"""Merge a targeted corrected stimulus rerun without rerunning the other frames."""
import argparse, hashlib, json
from pathlib import Path

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("original", type=Path)
    ap.add_argument("corrected", type=Path)
    ap.add_argument("output", type=Path)
    a = ap.parse_args()
    old, new = json.loads(a.original.read_text()), json.loads(a.corrected.read_text())
    replacement = {r["name"]: r for r in new["records"]}
    superseded = [r for r in old["records"] if r["name"] in replacement]
    records = [replacement.get(r["name"], r) for r in old["records"]]
    assert len(records) == 24 and set(replacement) == {r["name"] for r in superseded}
    out = dict(old)
    out["records"] = records
    out["provenance"] = {
        "merge": "targeted correction of four uint8-overflow pattern frames",
        "original_assay_path": str(a.original), "original_assay_sha256": sha(a.original),
        "corrected_subset_path": str(a.corrected), "corrected_subset_sha256": sha(a.corrected),
        "superseded_stimuli": [
            {"name": r["name"], "frame_sha256": r["frame_sha256"], "actual_mean": r["actual_mean"]}
            for r in superseded
        ],
        "replacement_stimuli": [
            {"name": r["name"], "frame_sha256": r["frame_sha256"], "actual_mean": r["actual_mean"]}
            for r in new["records"]
        ],
    }
    out.setdefault("limitations", []).append("Four mean=0.5 pattern frames were superseded after uint8 overflow was detected; remaining 20 records were retained byte-for-byte.")
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(out, indent=2, allow_nan=False))
    print(json.dumps({"frames": len(records), "superseded": len(superseded), "path": str(a.output)}))
if __name__ == "__main__": main()
