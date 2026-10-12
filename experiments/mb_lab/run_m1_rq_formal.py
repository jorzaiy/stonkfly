#!/usr/bin/env python3
"""Freeze-v5 M1-RQ formal runner. Requires a signed RQ-only authorization."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import traceback
import tempfile

from cryptography.hazmat.primitives.ciphers.aead import AESGCM
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stonkfly.mb_lab.authorization import verify_rq_authorization
from stonkfly.mb_lab.budget import BudgetExceeded, RuntimeBudget, append_failure, exclusive_json
from stonkfly.mb_lab.checkpoint import validated_chunk, write_chunk_once
from stonkfly.mb_lab.dynamics import M1Parameters
from stonkfly.mb_lab.formal import (
    domain_seed, i8_summary, science_statistics, simulate_collection_chunk, simulate_zero_chunk,
)
from stonkfly.mb_lab.inventory import load_source, selected_nodes
from stonkfly.mb_lab.protocol import load_frozen_config, topology_control_seeds
from stonkfly.mb_lab.provenance import bundle_sha256, sha256_file, verify_bundle
from stonkfly.mb_lab.stimuli import collection_sha256, generate_m1_collection
from stonkfly.mb_lab.topology import compile_real_topology

PUBLIC_KEY = ROOT / "protocols/mb_lab/m1_freeze_v5/authorization_public.pem"
RQ_MANIFEST = ROOT / "protocols/mb_lab/m1_freeze_v5/domains/m1_rq-manifest.json"
TOPOLOGY_MANIFEST = ROOT / "protocols/mb_lab/m1_freeze_v5/topology_manifest.json"
BUNDLE = ROOT / "protocols/mb_lab/m1_freeze_v5/provenance_bundle.json"
RQ_SEALED = ROOT / "data/mb_lab/custody/freeze-v5/m1_rq/payload.sealed"
RQ_KEY = ROOT / "data/mb_lab/custody/freeze-v5/m1_rq/domain.key"
FORBIDDEN_PATHS = (
    ROOT / "data/mb_lab/custody/freeze-v5/m1_rt",
    ROOT / "data/mb_lab/custody/freeze-v5/m2_lq",
    ROOT / "data/mb_lab/custody/freeze-v5/m2_lt",
)
CURRENT_BUDGET: RuntimeBudget | None = None


def jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def write_npz_once(path: Path, arrays: dict[str, np.ndarray]) -> None:
    if path.exists():
        raise FileExistsError(path)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".npz", dir=path.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        np.savez_compressed(temporary, **arrays)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def prior_usage(output: Path) -> tuple[float, float]:
    cpu = wall = 0.0
    for path in sorted((output / "failures").glob("attempt-*/formal.json")):
        usage = json.loads(path.read_text()).get("resource_usage", {})
        cpu = max(cpu, float(usage.get("cpu_seconds", 0.0)))
        wall = max(wall, float(usage.get("wall_seconds", 0.0)))
    return cpu, wall


def load_rq_collection(manifest: dict[str, object], config: dict[str, object]) -> dict[str, np.ndarray]:
    if any(os.access(path, os.R_OK) for path in FORBIDDEN_PATHS):
        raise PermissionError("RQ process can read a forbidden RT/M2 custody path")
    if sha256_file(RQ_SEALED) != manifest["sealed_payload_sha256"]:
        raise RuntimeError("RQ sealed payload hash mismatch")
    encrypted, key = RQ_SEALED.read_bytes(), RQ_KEY.read_bytes()
    plaintext = AESGCM(key).decrypt(encrypted[:12], encrypted[12:], b"M1_RQ")
    envelope = json.loads(plaintext)
    if envelope.get("domain") != "M1_RQ" or envelope.get("bundle_sha256") != manifest["protocol_bundle_sha256"]:
        raise RuntimeError("RQ envelope provenance mismatch")
    seed_bytes = bytes.fromhex(envelope["seed_hex"])
    if __import__("hashlib").sha256(seed_bytes).hexdigest() != manifest["seed_commitment_sha256"]:
        raise RuntimeError("RQ seed commitment mismatch")
    collection = generate_m1_collection(int.from_bytes(seed_bytes[:8], "little"), config["stimulus"])
    if collection_sha256(collection) != manifest["content_sha256"]:
        raise RuntimeError("RQ collection content mismatch")
    return collection


def parameters(config: dict[str, object]) -> M1Parameters:
    p = config["parameters"]
    return M1Parameters(float(p["tau_k_ms"]), float(p["tau_apl_ms"]), float(p["input_scale"]),
                        float(p["apl_gain"]), float(p["threshold"]),
                        float(p["active_mean_rate_threshold"]) * 500.0)


def npz_dict(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def run(args: argparse.Namespace) -> dict[str, object]:
    global CURRENT_BUDGET
    output = args.out.resolve()
    if (output / "COMPLETE.json").exists():
        raise FileExistsError("completed formal run is immutable")
    bundle = verify_bundle(ROOT, BUNDLE)
    bundle_hash = bundle_sha256(BUNDLE)
    rq_manifest = json.loads(RQ_MANIFEST.read_text())
    topology_manifest = json.loads(TOPOLOGY_MANIFEST.read_text())
    verify_rq_authorization(
        args.authorization, PUBLIC_KEY,
        expected_bundle_sha256=bundle_hash,
        expected_manifest_sha256=sha256_file(RQ_MANIFEST),
        expected_run_id=args.run_id,
        expected_output=str(output),
    )
    output.mkdir(parents=True, exist_ok=True)
    identity = {"run_id": args.run_id, "bundle_sha256": bundle_hash,
                "rq_manifest_sha256": sha256_file(RQ_MANIFEST), "domain": "M1_RQ"}
    run_record = output / "run.json"
    if run_record.exists():
        if json.loads(run_record.read_text()) != identity:
            raise RuntimeError("resume identity mismatch")
    else:
        exclusive_json(run_record, identity)
    prior_cpu, prior_wall = prior_usage(output)
    budget = RuntimeBudget(output, 24 * 3600, 24 * 3600, 4 * 1024**3, 5 * 1024**3,
                           prior_cpu_seconds=prior_cpu, prior_wall_seconds=prior_wall)
    CURRENT_BUDGET = budget
    budget.start()
    budget.install_process_limits()
    config = load_frozen_config()
    collection = load_rq_collection(rq_manifest, config)
    source = load_source()
    pn, kc = selected_nodes("R", olfactory_covered_only=True)
    real = compile_real_topology(source["ptr"], source["post"], pn, kc)
    swaps = int(config["topology_controls"]["accepted_double_edge_swaps_per_edge"]) * real.edge_count
    graphs = [real] + [real.rewire(seed, swaps=swaps) for seed in topology_control_seeds(config)]
    hashes = [graph.sha256 for graph in graphs]
    if hashes != topology_manifest["network_hashes"]:
        raise RuntimeError("formal topology hashes differ from frozen manifest")
    p = parameters(config)
    common = dict(identity)
    common["topology_manifest_sha256"] = sha256_file(TOPOLOGY_MANIFEST)
    for graph_index, graph in enumerate(graphs):
        for dt in (0.1, 0.05, 0.025):
            name = f"n{graph_index:02d}-on-dt{dt:.3f}"
            if validated_chunk(output, name, common) is None:
                budget.check(pending_bytes=4 * 1024**2)
                arrays = simulate_collection_chunk(graph, p, collection["repeated"], dt_ms=dt, apl_enabled=True)
                write_chunk_once(output, name, arrays, common)
                budget.check()
    for dt in (0.1, 0.05, 0.025):
        name = f"n00-off-dt{dt:.3f}"
        if validated_chunk(output, name, common) is None:
            budget.check(pending_bytes=4 * 1024**2)
            write_chunk_once(output, name, simulate_collection_chunk(real, p, collection["repeated"], dt_ms=dt, apl_enabled=False), common)
        for enabled, label in ((True, "on"), (False, "off")):
            zero_name = f"zero-{label}-dt{dt:.3f}"
            if validated_chunk(output, zero_name, common) is None:
                write_chunk_once(output, zero_name, simulate_zero_chunk(real, p, dt_ms=dt, apl_enabled=enabled), common)
        budget.check()
    _, permutation_seed = domain_seed("STONKFLY-V2/M1/FREEZE-V5/RQ/S2-SIGNFLIP/v1")
    science: dict[float, dict[str, object]] = {}
    paths: dict[str, dict[float, np.ndarray]] = {}
    for graph_index in range(40):
        paths[f"n{graph_index:02d}-on"] = {
            dt: npz_dict(output / "chunks" / f"n{graph_index:02d}-on-dt{dt:.3f}.npz")["response_integral"]
            for dt in (0.1, 0.05, 0.025)
        }
    paths["n00-off"] = {dt: npz_dict(output / "chunks" / f"n00-off-dt{dt:.3f}.npz")["response_integral"] for dt in (0.1, 0.05, 0.025)}
    for dt in (0.1, 0.05, 0.025):
        b_on = npz_dict(output / "chunks" / f"n00-on-dt{dt:.3f}.npz")
        b_off = npz_dict(output / "chunks" / f"n00-off-dt{dt:.3f}.npz")
        controls = [npz_dict(output / "chunks" / f"n{i:02d}-on-dt{dt:.3f}.npz") for i in range(1, 40)]
        science[dt] = science_statistics(b_on, b_off, controls, collection["base"], permutation_seed=permutation_seed)
    vectors = {dt: tuple(bool(science[dt][name]) for name in ("S1_B", "S2_B", "S3_B", "S4_B_APL", "S5_B_VS_39C")) for dt in science}
    i8 = i8_summary(paths, vectors)
    zero_pass = True
    for dt in (0.1, 0.05, 0.025):
        for label in ("on", "off"):
            item = npz_dict(output / "chunks" / f"zero-{label}-dt{dt:.3f}.npz")
            zero_pass &= bool(item["finite"].all() and np.array_equal(item["response_integral"], np.zeros_like(item["response_integral"])) and np.array_equal(item["state_error"], np.zeros_like(item["state_error"])))
    reference = science[0.025]
    implementation_pass = bool(zero_pass and all(npz_dict(path)["finite"].all() for path in sorted((output / "chunks").glob("*.npz"))))
    if not implementation_pass:
        verdict = "M1_IMPLEMENTATION_FAILURE"
    elif not i8["pass"]:
        verdict = "M1_NUMERICAL_NO_GO"
    elif not all(bool(reference[name]) for name in ("S1_B", "S2_B", "S3_B", "S4_B_APL")):
        verdict = "M1_REPRESENTATION_NO_GO"
    else:
        verdict = "M1_RQ_QUALIFIED"
    formal = {
        "schema": "stonkfly-m1-freeze-v5-rq-formal-v1", "domain": "M1_RQ",
        "verdict": verdict,
        "topology_substatus": "SUPPORTED_AT_RQ_ONLY" if reference["S5_B_VS_39C"] else "NOT_DEMONSTRATED_AT_RQ",
        "rt_accessed": False, "m2_accessed": False, "market_used": False,
        "learning_used": False, "execution_used": False,
        "science": jsonable(science),
        "i8": i8, "zero_pass": zero_pass, "resource_usage": budget.check(), "identity": identity,
    }
    exclusive_json(output / "m1_formal.json", formal)
    raw_summary = {"science_dt": np.asarray([0.1, 0.05, 0.025]), "reference_control_t_sep": reference["control_t_sep"], "reference_apl_deltas": reference["apl_deltas"], "reference_covered_fraction": reference["covered_fraction"], "reference_all_2045_fraction": reference["all_2045_fraction"]}
    raw_path = output / "m1_raw.npz"
    write_npz_once(raw_path, raw_summary)
    report = output / "m1_report.md"
    descriptor = os.open(report, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o640)
    with os.fdopen(descriptor, "w") as handle:
        handle.write(f"# M1-RQ formal report\n\nVerdict: `{verdict}`\n\nRT remains sealed.\n")
    exclusive_json(output / "stdout.json", {"verdict": verdict})
    exclusive_json(output / "launch.json", {
        "argv": [str(item) for item in sys.argv], "pid": os.getpid(), "uid": os.getuid(),
        "cwd": str(Path.cwd()), "authorization_sha256": sha256_file(args.authorization),
    })
    artifacts = {path.name: sha256_file(path) for path in (
        run_record, output / "m1_formal.json", raw_path, report,
        output / "stdout.json", output / "launch.json")}
    manifest = {"schema": "stonkfly-m1-freeze-v5-rq-result-manifest-v1", "identity": identity,
                "source_hashes_start": bundle["files"], "source_hashes_end": verify_bundle(ROOT, BUNDLE)["files"],
                "topology_hashes": hashes, "artifacts": artifacts, "resource_usage": budget.check()}
    exclusive_json(output / "m1_manifest.json", manifest)
    exclusive_json(output / "COMPLETE.json", {"verdict": verdict, "manifest_sha256": sha256_file(output / "m1_manifest.json")})
    return formal


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--authorization", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = run(args)
    except BaseException as error:
        if args.out.exists() and not (args.out / "COMPLETE.json").exists():
            usage = CURRENT_BUDGET.usage() if CURRENT_BUDGET is not None else {}
            verdict = "M1_BUDGET_EXHAUSTED" if isinstance(error, BudgetExceeded) else "M1_IMPLEMENTATION_FAILURE"
            append_failure(args.out, {"error": repr(error), "traceback": traceback.format_exc(),
                                      "run_id": args.run_id, "verdict": verdict,
                                      "resource_usage": usage})
        raise
    print(json.dumps({"verdict": result["verdict"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
