# Purged WC006 selector continuation, 2024–2025

Status: research input build only. Frozen on 2026-09-27 before any 2024–2025
Fly/Base returns are computed. No neural replay or broker execution is part of
this step.

- Start from a byte-for-byte copy of the completed 2021–2023 purged selector
  mirror at `runs/wc006_v3_2021_inputs/20260925T-purged-full-2021-2023-stateful`.
  Its checkpoint SHA256 is
  `e9188ffd44ea1296c2759de6d56752965ad0cfa97c13b9bd10c684dfb41d1641`.
  Verify every copied source and prior watch artifact against its checkpoint.
  The external trading repository's exporter and walkforward sources have
  since changed. The continuation explicitly allows that historical source
  drift only after checking the frozen mirror hashes and prior artifacts; it
  executes the frozen mirror files, never the external newer sources. The
  external repository must remain unchanged during the continuation.
  Execute with the existing experiment interpreter
  `/home/ubuntu/trading_agent_20260408/project/venv_exp_clean/bin/python`,
  which has the frozen selector's LightGBM dependency. Record its package
  versions in the extension provenance. Do not install or replace model code.
- Append the existing contiguous 20-session hold schedule from 2024-01-02
  through 2025-12-29 (36 signal dates: 18 in each year). Keep the exact
  source calendar, universe, config, cache tag
  `5y_2021-01-01_2026-04-03`, turnover/state overlays and 20-session
  forward-label purge. Prior 46 signal artifacts are immutable.
- Run the exporter only in the new isolated mirror. The trading repository is
  read-only and its watched outputs must hash unchanged before and after each
  date. Any missing causal cache bar, changed artifact, schedule mismatch or
  incomplete selector state stops the build.
- Build a new, separately hash-pinned candidate stream from the new applied
  artifacts only after all 36 dates pass. It must have complete lookback and
  next-session timing proof. No selector result may be chosen or dropped based
  on realized return.
- XYSZ raw Kline, official status/limits, adjustment and point-in-time
  corporate-action coverage for the new target union is a separate mandatory
  gate before any Fly/Base replay. QFQ remains observation-only.
