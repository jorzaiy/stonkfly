"""Analyze KC (Kenyon Cell) activation overlap across training days.

S2 diagnostic from Opus 5.5 review: if KC sets activated by different days
have high Jaccard overlap, then any plasticity rule cannot learn discriminative
memories because reward/punishment will affect the same edges.

This script:
1. Loads events from a training run
2. Reconstructs RGB frames from observation data
3. Runs neural network to get KC spike counts
4. Computes pairwise Jaccard overlap between KC active sets
5. Reports statistics separately for reward days, punishment days, and cross-comparisons
"""

import argparse
import hashlib
import json
import os
import sys
import tempfile
from collections import defaultdict
from decimal import Decimal
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from stonkfly.config import Settings
from stonkfly.neural.sensory import retinal_samples
from stonkfly.neural.v51_controller import V51Controller


def D(value):
    return Decimal(str(value))


def _stats(closes, n):
    """Compute return and volatility over last n periods."""
    if len(closes) < n + 1:
        return 0.0, 0.0
    recent = closes[-n:]
    base = closes[-n - 1]
    if base == 0:
        return 0.0, 0.0
    ret = float((recent[-1] / base) - 1)
    returns = [float((recent[i] / recent[i-1]) - 1) if recent[i-1] != 0 else 0.0 for i in range(1, len(recent))]
    vol = float(np.std(returns)) if returns else 0.0
    return ret, vol


def render_frame_from_observation(obs: dict) -> np.ndarray:
    """Reconstruct the exact production RGB frame from its audit payload."""
    visual = obs.get("visual")
    if not isinstance(visual, dict):
        raise ValueError("observation lacks visual audit payload")
    if visual.get("frame_schema") != "wc006-rgb-frame/v1":
        raise ValueError("unsupported visual frame schema")
    candidates = visual.get("candidates")
    header = visual.get("header")
    if not isinstance(candidates, list) or not candidates or not isinstance(header, dict):
        raise ValueError("visual audit payload is incomplete")
    date = str(header["date"])
    multiplier = D(header["multiplier"])
    cash_fraction = float(header["cash_fraction"])
    drawdown = float(header["drawdown"])
    
    image = Image.new("RGB", (320, 180), (235, 240, 249))
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, 319, 20), fill=(19, 36, 71))
    draw.text((4, 4), f"wc006v4 {date} M{multiplier:.1f} C{cash_fraction:.2f} D{drawdown:.2f}", fill=(219, 229, 249))
    
    if not candidates:
        return np.asarray(image, dtype=np.uint8)
    
    row_height = max(1, 150 // len(candidates))
    
    for index, cand in enumerate(candidates):
        code = cand.get("code", "")
        normalized = cand.get("plot_close_path_60")
        if not isinstance(normalized, list) or len(normalized) != 60:
            raise ValueError(f"{code}: invalid normalized close path")
        normalized = [float(v) for v in normalized]
        volume_norm = cand.get("plot_volume_path_60")
        if not isinstance(volume_norm, list) or len(volume_norm) != 60:
            raise ValueError(f"{code}: invalid normalized volume path")
        volume_norm = [float(v) for v in volume_norm]
        
        # Extract stats
        r5 = float(cand["return_5"])
        r20 = float(cand["return_20"])
        r60 = float(cand["return_60"])
        v5 = float(cand["volatility_5"])
        v20 = float(cand["volatility_20"])
        v60 = float(cand["volatility_60"])
        target_exposure = float(D(cand.get("target_exposure", "0")))
        current_exposure = float(D(cand.get("current_exposure", "0")))
        stale = int(bool(cand["stale"]))
        geometry = cand.get("plot_geometry", {})
        top = int(geometry.get("top", 24 + index * row_height))
        bottom = int(geometry.get("bottom", min(176, top + row_height - 4)))
        
        draw.text((3, top), f"{code} T{target_exposure:.3f} E{current_exposure:.3f} M{stale} r{r5:.2%}/{r20:.2%}/{r60:.2%}"[:38], fill=(28, 46, 82))
        
        for series, color in ((normalized, (0, 101, 183)), (volume_norm, (160, 90, 30))):
            if not series:
                continue
            lo, hi = min(series), max(series)
            span = max(hi - lo, 1e-12)
            points = []
            for pos, value in enumerate(series):
                x = 60 + pos * 255 / max(1, len(series) - 1)
                y = bottom - (value - lo) / span * max(2, row_height - 18)
                points.append((x, y))
            if len(points) > 1:
                draw.line(points, fill=color, width=1)
        
        draw.text((4, bottom - 10), f"v{v5:.3f}/{v20:.3f}/{v60:.3f}", fill=(92, 65, 42))
    
    frame = np.asarray(image, dtype=np.uint8)
    actual = hashlib.sha256(frame.tobytes()).hexdigest()
    if actual != visual.get("frame_sha256"):
        raise ValueError(
            f"frame hash mismatch: reconstructed={actual} recorded={visual.get('frame_sha256')}"
        )
    return frame


def jaccard_similarity(set1: set, set2: set) -> float:
    """Compute Jaccard similarity between two sets."""
    if not set1 and not set2:
        return 1.0
    if not set1 or not set2:
        return 0.0
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union > 0 else 0.0


def numeric_summary(values: np.ndarray) -> dict:
    """Compact finite distribution summary for a sensory vector."""
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if not array.size or not np.isfinite(array).all():
        raise ValueError("sensory diagnostic values must be nonempty and finite")
    return {
        "n": int(array.size),
        "mean": float(np.mean(array)),
        "std": float(np.std(array)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
        "p05": float(np.quantile(array, 0.05)),
        "p50": float(np.quantile(array, 0.50)),
        "p95": float(np.quantile(array, 0.95)),
    }


def layer_spike_summary(counts: np.ndarray, indices: np.ndarray, seconds: float) -> dict:
    indices = np.asarray(indices, dtype=np.int64)
    values = np.asarray(counts)[indices]
    return {
        "cell_count": int(indices.size),
        "active_cells": int(np.count_nonzero(values)),
        "active_fraction": float(np.count_nonzero(values) / indices.size) if indices.size else 0.0,
        "spikes": int(values.sum()),
        "mean_hz": float(np.mean(values) / seconds) if indices.size else 0.0,
        "max_spikes": int(values.max()) if indices.size else 0,
    }


def analyze_kc_overlap(
    run_path: Path,
    sample_size: int = 60,
    seed: int = 42,
) -> dict:
    """Analyze KC activation overlap from a training run.
    
    Args:
        run_path: Path to training run directory
        sample_size: Number of days to sample (default 60 as suggested by Opus 5.5)
        seed: Random seed for sampling
    
    Returns:
        Dictionary with overlap statistics
    """
    events_path = run_path / "events.jsonl"
    
    if not events_path.exists():
        raise FileNotFoundError(f"events.jsonl not found at {run_path}")
    
    # Load events
    events = []
    with open(events_path) as f:
        for line in f:
            events.append(json.loads(line))
    
    print(f"Loaded {len(events)} events from {run_path.name}")
    
    # Categorize by reward/punishment based on reward.stimulus field
    reward_days = []
    punishment_days = []
    neutral_days = []
    
    for i, event in enumerate(events):
        stimulus = event.get("reward", {}).get("stimulus", "none")
        if stimulus == "reward":
            reward_days.append(i)
        elif stimulus == "aversive":
            punishment_days.append(i)
        else:
            neutral_days.append(i)
    
    print(f"Reward days: {len(reward_days)}, Punishment days: {len(punishment_days)}, Neutral: {len(neutral_days)}")
    
    # Sample days
    rng = np.random.default_rng(seed)
    
    def sample_days(day_list, n):
        if len(day_list) <= n:
            return day_list
        return list(rng.choice(day_list, size=n, replace=False))
    
    # Sample proportionally from each category
    n_reward = min(sample_size // 3, len(reward_days))
    n_punishment = min(sample_size // 3, len(punishment_days))
    n_neutral = min(sample_size - n_reward - n_punishment, len(neutral_days))
    
    sampled_reward = sample_days(reward_days, n_reward)
    sampled_punishment = sample_days(punishment_days, n_punishment)
    sampled_neutral = sample_days(neutral_days, n_neutral)
    
    all_sampled = sorted(set(sampled_reward + sampled_punishment + sampled_neutral))
    print(f"Sampled {len(all_sampled)} days: {len(sampled_reward)} reward, {len(sampled_punishment)} punishment, {len(sampled_neutral)} neutral")
    
    # Initialize controller (fresh, no checkpoint needed for KC analysis)
    # We only need to observe frames and record KC activity, not train
    settings = Settings(neural_ms=500, pulse_ms=200, learning=False)
    controller = V51Controller(settings)
    
    # Get KC indices
    kc_indices = np.asarray(controller.brain.circuit["kc"], dtype=np.int64)
    print(f"KC neurons: {len(kc_indices)}")
    superclass = np.asarray(controller.brain.superclass).astype(str)
    layer_indices = {
        "r1_r6": np.asarray(controller.brain.retina, dtype=np.int64),
        "r8": np.asarray(controller.brain.r8, dtype=np.int64),
        "lamina": np.asarray(controller.brain.lamina, dtype=np.int64),
        "ol_sensory": np.flatnonzero(superclass == "ol_sensory"),
        "ol_intrinsic": np.flatnonzero(superclass == "ol_intrinsic"),
        "visual_projection": np.flatnonzero(superclass == "visual_projection"),
        "cb_intrinsic": np.flatnonzero(superclass == "cb_intrinsic"),
        "kc": kc_indices,
        "mbon07": np.asarray(controller.brain.circuit["reward_mb"], dtype=np.int64),
        "mbon11": np.asarray(controller.brain.circuit["aversive_mb"], dtype=np.int64),
    }
    
    # Restore the complete clean checkpoint for every frame; resetting only
    # weights/traces leaves membrane, conductance and refractory state behind.
    kc_active_sets = {}
    frame_hashes = {}
    repeatability = None
    with tempfile.TemporaryDirectory(prefix="stonkfly-kc-overlap-") as temp_dir:
        clean_checkpoint = Path(temp_dir) / "brain-clean.npz"
        controller.save(clean_checkpoint)

        for i, day_idx in enumerate(all_sampled):
            event = events[day_idx]
            obs = event.get("observation", {})
        
        # Reconstruct frame
            frame = render_frame_from_observation(obs)
            expected_hash = event["neural"]["input_sha256"]
            actual_hash = hashlib.sha256(frame.tobytes()).hexdigest()
            if actual_hash != expected_hash:
                raise RuntimeError(
                    f"day {day_idx} neural/visual frame hashes differ: "
                    f"{actual_hash} != {expected_hash}"
                )
            frame_hashes[day_idx] = actual_hash

            controller.restore(clean_checkpoint)
            retinal_input = retinal_samples(frame, controller.brain.uv)
            h, w = frame.shape[:2]
            r8_x = np.minimum(
                (controller.brain.r8_uv[:, 0] * (w - 1)).astype(int), w - 1
            )
            r8_y = np.minimum(
                (controller.brain.r8_uv[:, 1] * (h - 1)).astype(int), h - 1
            )
            r8_input = (
                frame[r8_y, r8_x, controller.brain.r8_channel].astype(np.float32)
                / 255.0
            )
            r8_input = np.where(
                r8_input <= 0.04045,
                r8_input / 12.92,
                ((r8_input + 0.055) / 1.055) ** 2.4,
            )
            result = controller.observe(frame, reinforcement="none")
            counts = controller.brain.counts
            kc_counts = counts[kc_indices]
            active_kc = set(np.flatnonzero(kc_counts > 0))

            if repeatability is None:
                controller.restore(clean_checkpoint)
                repeated = controller.observe(frame, reinforcement="none")
                repeated_counts = controller.brain.counts[kc_indices]
                repeatability = {
                    "date": event["date"],
                    "spike_sha256_match": result["spike_sha256"] == repeated["spike_sha256"],
                    "kc_counts_match": bool(np.array_equal(kc_counts, repeated_counts)),
                }
                if not all(repeatability[key] for key in ("spike_sha256_match", "kc_counts_match")):
                    raise RuntimeError("clean-checkpoint repeated observation is not deterministic")

            kc_active_sets[day_idx] = {
                "date": event["date"],
                "active_kc": active_kc,
                "n_active": len(active_kc),
                "total_kc_spikes": int(kc_counts.sum()),
                "category": "reward" if day_idx in sampled_reward else ("punishment" if day_idx in sampled_punishment else "neutral"),
                "sensory_diagnostic": {
                    "pixels": {
                        "all": numeric_summary(frame.astype(np.float32) / 255.0),
                        "red": numeric_summary(frame[:, :, 0].astype(np.float32) / 255.0),
                        "green": numeric_summary(frame[:, :, 1].astype(np.float32) / 255.0),
                        "blue": numeric_summary(frame[:, :, 2].astype(np.float32) / 255.0),
                    },
                    "r1_r6_sample": numeric_summary(retinal_input),
                    "r1_r6_filtered": numeric_summary(controller.brain.luminance),
                    "r1_r6_current": numeric_summary(
                        30 * controller.brain.luminance / (0.02 + controller.brain.luminance)
                    ),
                    "r8_sample": numeric_summary(r8_input),
                    "r8_filtered": numeric_summary(controller.brain.r8_light),
                    "r8_current": numeric_summary(
                        30 * controller.brain.r8_light / (0.02 + controller.brain.r8_light)
                    ),
                    "layers": {
                        name: layer_spike_summary(
                            counts, indices, settings.neural_ms / 1000.0
                        )
                        for name, indices in layer_indices.items()
                    },
                },
            }

            if (i + 1) % 10 == 0:
                print(f"  Processed {i + 1}/{len(all_sampled)} days")
    
    print(f"Successfully processed {len(kc_active_sets)} days")
    
    # Compute pairwise Jaccard similarities
    def compute_pairwise_jaccard(day_indices):
        similarities = []
        for i, idx1 in enumerate(day_indices):
            if idx1 not in kc_active_sets:
                continue
            for idx2 in day_indices[i+1:]:
                if idx2 not in kc_active_sets:
                    continue
                sim = jaccard_similarity(
                    kc_active_sets[idx1]["active_kc"],
                    kc_active_sets[idx2]["active_kc"]
                )
                similarities.append(sim)
        return similarities

    def compute_cross_jaccard(left_indices, right_indices):
        similarities = []
        for left in left_indices:
            if left not in kc_active_sets:
                continue
            for right in right_indices:
                if right not in kc_active_sets:
                    continue
                similarities.append(
                    jaccard_similarity(
                        kc_active_sets[left]["active_kc"],
                        kc_active_sets[right]["active_kc"],
                    )
                )
        return similarities
    
    # Compute within-group and between-group similarities
    reward_within = compute_pairwise_jaccard(sampled_reward)
    punishment_within = compute_pairwise_jaccard(sampled_punishment)
    
    # Cross-group: reward vs punishment
    cross_similarities = compute_cross_jaccard(
        sampled_reward, sampled_punishment
    )
    all_indices = sorted(kc_active_sets)
    overall_similarities = compute_pairwise_jaccard(all_indices)
    
    def stats(values):
        if not values:
            return {"n": 0, "mean": None, "std": None, "min": None, "max": None, "median": None}
        arr = np.array(values)
        return {
            "n": len(values),
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "median": float(np.median(arr)),
            "quantiles": {
                str(label): float(value)
                for label, value in zip(
                    ("p05", "p25", "p50", "p75", "p95"),
                    np.quantile(arr, (0.05, 0.25, 0.50, 0.75, 0.95)),
                )
            },
            "identical_count": int(np.count_nonzero(np.isclose(arr, 1.0))),
            "identical_fraction": float(np.mean(np.isclose(arr, 1.0))),
            "near_disjoint_count": int(np.count_nonzero(arr <= 0.01)),
            "near_disjoint_fraction": float(np.mean(arr <= 0.01)),
        }

    active_counts = np.asarray(
        [item["n_active"] for item in kc_active_sets.values()], dtype=np.int64
    )
    active_set_frequencies = defaultdict(int)
    for item in kc_active_sets.values():
        active_set_frequencies[tuple(sorted(item["active_kc"]))] += 1
    frequency_values = sorted(active_set_frequencies.values(), reverse=True)
    per_day = [
        {
            "event_index": int(index),
            "date": item["date"],
            "category": item["category"],
            "frame_sha256": frame_hashes[index],
            "active_kc_count": item["n_active"],
            "total_kc_spikes": item["total_kc_spikes"],
            "active_kc_indices": sorted(int(value) for value in item["active_kc"]),
            "sensory_diagnostic": item["sensory_diagnostic"],
        }
        for index, item in sorted(kc_active_sets.items())
    ]
    
    results = {
        "run_path": str(run_path),
        "total_events": len(events),
        "sampled_days": len(kc_active_sets),
        "kc_neuron_count": len(kc_indices),
        "frame_hashes_verified": len(frame_hashes),
        "repeatability": repeatability,
        "active_kc_stats": {
            "mean": float(np.mean(active_counts)),
            "std": float(np.std(active_counts)),
            "min": int(np.min(active_counts)),
            "max": int(np.max(active_counts)),
            "quantiles": {
                str(label): float(value)
                for label, value in zip(
                    ("p05", "p25", "p50", "p75", "p95"),
                    np.quantile(active_counts, (0.05, 0.25, 0.50, 0.75, 0.95)),
                )
            },
            "days_at_most_10": int(np.count_nonzero(active_counts <= 10)),
            "days_at_least_1000": int(np.count_nonzero(active_counts >= 1000)),
        },
        "active_set_repetition": {
            "unique_sets": len(active_set_frequencies),
            "most_common_set_days": frequency_values[0] if frequency_values else 0,
            "sets_repeated_on_multiple_days": sum(
                count > 1 for count in frequency_values
            ),
        },
        "jaccard_overlap": {
            "all_pairs": stats(overall_similarities),
            "reward_within": stats(reward_within),
            "punishment_within": stats(punishment_within),
            "reward_vs_punishment": stats(cross_similarities),
        },
        "per_day": per_day,
        "interpretation": None,
    }
    
    # Interpretation
    results["interpretation"] = (
        "DESCRIPTIVE_ONLY: inspect activation-size quantiles, repeated-set counts, "
        "and the full overlap distribution before any mechanism gate decision."
    )
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Analyze KC activation overlap")
    parser.add_argument("run_path", type=Path, help="Path to training run directory")
    parser.add_argument("--sample-size", type=int, default=60, help="Number of days to sample")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", type=Path, help="Output JSON path")
    args = parser.parse_args()
    
    results = analyze_kc_overlap(args.run_path, args.sample_size, args.seed)
    
    # Print summary
    print("\n" + "=" * 60)
    print("KC OVERLAP ANALYSIS RESULTS")
    print("=" * 60)
    print(f"Run: {results['run_path']}")
    print(f"Sampled days: {results['sampled_days']}")
    print(f"KC neurons: {results['kc_neuron_count']}")
    print(f"Active KCs per day: {results['active_kc_stats']['mean']:.0f} ± {results['active_kc_stats']['std']:.0f}")
    print()
    print("Jaccard Overlap (0=no overlap, 1=identical):")
    for key, stat in results["jaccard_overlap"].items():
        if stat["mean"] is not None:
            print(f"  {key}: {stat['mean']:.3f} ± {stat['std']:.3f} (n={stat['n']})")
    print()
    print(f"Interpretation: {results['interpretation']}")
    
    # Save results
    if args.output:
        output_path = args.output
    else:
        output_path = args.run_path / "kc_overlap_analysis.json"
    
    with open(output_path, "w") as f:
        # Convert sets to lists for JSON serialization
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
