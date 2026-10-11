#!/usr/bin/env python3
"""Build extended QFQ cache for 2021 baseline by prepending verified 2020 lookback.

Lineage & Provenance:
- 1,867 SZ universe stocks from universe_5y_days161.top2000tag.ok1867.unique.txt
- Base QFQ cache (2021-01-04 .. 2026-04-03) from data/cache_canonical_202603
- Raw 2020 history (2020-01-01 .. 2020-12-31) from data/cache_hist_2018_2020
- Raw anchor prices from data/tdx_full/normalized/sz_security
- Official corporate actions from AmazingData BaseData.get_adj_factor (8,732 dates x 1,867 codes)
    Consolidated at runs/wc006_v5_1_2021_baseline/inputs/corporate_actions/amazingdata_adj_factor_all1867.csv
- Mathematically proven QFQ backward-propagation formula:
    ratio(t) = ratio(anchor) / prod_{t < s <= anchor} adj_factor(s)
- Provides 243 trading days of 2020 history, strictly exceeding the 120-day min_periods
  required by make_features (proximity_52w).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import sys
import time
import pandas as pd
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
TRADING_ROOT = Path("/home/ubuntu/trading_agent_20260408")

UNIVERSE_PATH = TRADING_ROOT / "project/universe_5y_days161.top2000tag.ok1867.unique.txt"
CANONICAL_CACHE_DIR = TRADING_ROOT / "data/cache_canonical_202603"
HIST_CACHE_DIR = TRADING_ROOT / "data/cache_hist_2018_2020"
TDX_SZ_DIR = TRADING_ROOT / "data/tdx_full/normalized/sz_security"

OUTPUT_DIR = REPO_ROOT / "runs/wc006_v5_1_2021_baseline/inputs/cache"
CORP_DIR = REPO_ROOT / "runs/wc006_v5_1_2021_baseline/inputs/corporate_actions"
ADJ_FACTOR_CSV = CORP_DIR / "amazingdata_adj_factor_all1867.csv"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def build_extended_cache() -> dict[str, object]:
    t0 = time.time()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CORP_DIR.mkdir(parents=True, exist_ok=True)

    if not ADJ_FACTOR_CSV.exists():
        raise FileNotFoundError(f"Missing consolidated adj_factor CSV: {ADJ_FACTOR_CSV}")

    print("Loading AmazingData consolidated adj_factor table...", flush=True)
    adj_df = pd.read_csv(ADJ_FACTOR_CSV, index_col=0)

    universe = [l.strip() for l in UNIVERSE_PATH.read_text().splitlines() if l.strip()]
    if len(universe) != 1867:
        raise ValueError(f"Expected 1867 universe codes, got {len(universe)}")

    file_manifests: dict[str, dict[str, object]] = {}
    ipo_count = 0
    extended_count = 0

    for idx, code in enumerate(universe, 1):
        canon_file = CANONICAL_CACHE_DIR / f"{code}_5y_2021-01-01_2026-04-03.csv"
        if not canon_file.exists():
            raise FileNotFoundError(f"Missing canonical cache file for {code}: {canon_file}")

        out_file = OUTPUT_DIR / f"{code}_5y_2021-01-01_2026-04-03.csv"
        hist_file = HIST_CACHE_DIR / f"{code}_tdxvipdoc_2018-01-01_2020-12-31.csv"

        if not hist_file.exists():
            # Post-2020 IPO stock: copy directly
            shutil.copyfile(canon_file, out_file)
            ipo_count += 1
            file_manifests[code] = {
                "type": "ipo_post_2020",
                "sha256": sha256_file(out_file),
                "rows": sum(1 for _ in out_file.open()) - 1,
            }
            continue

        # Load canonical 2021+ data
        df_canon = pd.read_csv(canon_file)
        # Load raw 2020 data
        df_hist = pd.read_csv(hist_file, comment="#")
        # Filter for full calendar year 2020 (2020-01-01 .. 2020-12-31)
        mask_2020 = (df_hist["date"] >= "2020-01-01") & (df_hist["date"] <= "2020-12-31")
        df_2020_raw = df_hist[mask_2020].copy()

        if len(df_2020_raw) == 0:
            shutil.copyfile(canon_file, out_file)
            ipo_count += 1
            file_manifests[code] = {
                "type": "no_2020_trading",
                "sha256": sha256_file(out_file),
                "rows": len(df_canon),
            }
            continue

        # Determine anchor raw and canonical prices
        tdx_raw_file = TDX_SZ_DIR / f"sz{code}.csv"
        df_tdx = pd.read_csv(tdx_raw_file, comment="#")
        row_tdx_0104 = df_tdx[df_tdx["date"] == "2021-01-04"]
        if len(row_tdx_0104) > 0 and float(row_tdx_0104["volume"].iloc[0]) > 0:
            anchor_date = "2021-01-04"
            p_raw_anchor = float(row_tdx_0104["close"].iloc[0])
            p_canon_anchor = float(df_canon[df_canon["date"] == "2021-01-04"]["close"].iloc[0])
            ratio_anchor = p_canon_anchor / p_raw_anchor
        else:
            # Fallback for suspended stocks on 2021-01-04: find first active date in 2021
            df_canon_act = df_canon[df_canon["volume"] > 0]
            df_tdx_act = df_tdx[df_tdx["volume"] > 0]
            merged_act = pd.merge(df_canon_act, df_tdx_act, on="date", suffixes=("_canon", "_tdx"))
            if len(merged_act) > 0:
                first_act = merged_act.iloc[0]
                anchor_date = str(first_act["date"])
                ratio_anchor = float(first_act["close_canon"]) / float(first_act["close_tdx"])
            else:
                anchor_date = "last_hist"
                ratio_anchor = float(df_canon["close"].iloc[0]) / float(df_hist["close"].iloc[-1])

        # Compute QFQ adjustment ratio per 2020 row using AmazingData factors
        col_name = f"{code}.SZ"
        if col_name in adj_df.columns:
            s_adj = adj_df[col_name]
        else:
            s_adj = pd.Series(1.0, index=adj_df.index)

        # Slice adj factors between 2020-01-01 and anchor_date
        s_slice = s_adj[(s_adj.index >= "2020-01-01") & (s_adj.index <= anchor_date)].fillna(1.0)
        # Cumulative product from future back to present:
        # On date t, ratio(t) = ratio_anchor / prod(adj(s) for s in (t, anchor_date])
        # prod_future[t] is prod(adj(s) for s > t)
        cumprod_rev = s_slice.iloc[::-1].cumprod().iloc[::-1]
        prod_future = cumprod_rev.shift(-1).fillna(1.0)
        ratio_map = (ratio_anchor / prod_future).to_dict()

        # Map to df_2020_raw dates (defaulting to ratio_anchor if date not in calendar)
        ratios = df_2020_raw["date"].map(ratio_map).fillna(ratio_anchor).values

        df_2020_qfq = pd.DataFrame({
            "date": df_2020_raw["date"].values,
            "open": df_2020_raw["open"].values * ratios,
            "high": df_2020_raw["high"].values * ratios,
            "low": df_2020_raw["low"].values * ratios,
            "close": df_2020_raw["close"].values * ratios,
            "volume": df_2020_raw["volume"].astype(np.int64).values,
        })

        # Combine 2020 QFQ with canonical 2021+
        df_combined = pd.concat([df_2020_qfq, df_canon], ignore_index=True)
        # Ensure no duplicate dates
        df_combined = df_combined.drop_duplicates(subset=["date"]).sort_values("date").reset_index(drop=True)

        # Basic validations
        assert not df_2020_qfq.isna().any().any(), f"NaNs found in 2020 QFQ for {code}"
        assert not df_combined[["date", "open", "high", "low", "close"]].isna().any().any(), f"NaNs in OHLC for {code}"
        assert df_combined["date"].is_monotonic_increasing, f"Dates not monotonic in {code}"
        assert df_combined["date"].nunique() == len(df_combined), f"Duplicate dates in {code}"

        # Write to output
        df_combined.to_csv(out_file, index=False, float_format="%.8f")
        extended_count += 1
        file_manifests[code] = {
            "type": "extended_with_2020_qfq",
            "sha256": sha256_file(out_file),
            "rows": len(df_combined),
            "ratio_anchor": ratio_anchor,
            "anchor_date": anchor_date,
            "qfq_2020_rows": len(df_2020_qfq),
        }

    elapsed = round(time.time() - t0, 2)
    manifest = {
        "schema": "wc006-extended-qfq-cache-manifest/v2",
        "description": "Full calendar 2020 lookback prepended to canonical cache using AmazingData official adjustment factors",
        "created_at_epoch": time.time(),
        "elapsed_seconds": elapsed,
        "total_files": len(file_manifests),
        "extended_files": extended_count,
        "ipo_or_unextended_files": ipo_count,
        "lookback_period": "2020-01-01 .. 2020-12-31",
        "canonical_cache_dir": str(CANONICAL_CACHE_DIR),
        "hist_cache_dir": str(HIST_CACHE_DIR),
        "output_cache_dir": str(OUTPUT_DIR),
        "adj_factor_source": str(ADJ_FACTOR_CSV),
        "files": file_manifests,
    }
    manifest_path = OUTPUT_DIR / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Extended cache complete: {extended_count} extended, {ipo_count} IPO/unextended, elapsed={elapsed}s")
    print(f"Manifest written to {manifest_path}")
    return manifest


if __name__ == "__main__":
    build_extended_cache()
