#!/usr/bin/env bash
set -euo pipefail

# Safe bridge for the x86 AmazingData collector.  Codes are passed as a
# Bash array; no command substitution or newline-to-space rewriting is used.
repo_dir=/home/ubuntu/stonkfly-lab
project_dir=/home/ubuntu/trading_agent_20260408/project
credential_dir=/home/ubuntu/trading_agent_20260408/xysz
prepared_dir=$repo_dir/runs/wc006_v5_market_inputs/manual-bridge/prepared
codes_file=$prepared_dir/codes_125.txt
output_name=${1:-history_stock_status_125_attempt2}

if [[ ! -f $codes_file ]]; then
    printf 'missing codes file: %s\n' "$codes_file" >&2
    exit 2
fi
if [[ $output_name != history_stock_status_125_* || $output_name == */* ]]; then
    printf 'output name must match history_stock_status_125_* and contain no slash\n' >&2
    exit 2
fi
if [[ -e $prepared_dir/$output_name ]]; then
    printf 'refusing to reuse nonempty/previous output path: %s\n' "$prepared_dir/$output_name" >&2
    exit 2
fi

mapfile -t codes < "$codes_file"
if (( ${#codes[@]} != 125 )); then
    printf 'expected 125 codes, found %d\n' "${#codes[@]}" >&2
    exit 2
fi

declare -A seen=()
for code in "${codes[@]}"; do
    if [[ ! $code =~ ^[0-9]{6}\.(SZ|SH)$ ]]; then
        printf 'invalid market code: %s\n' "$code" >&2
        exit 2
    fi
    if [[ ${seen[$code]+yes} ]]; then
        printf 'duplicate market code: %s\n' "$code" >&2
        exit 2
    fi
    seen[$code]=1
done

exec sudo -n docker run --rm \
    --platform linux/amd64 \
    --network bridge \
    -v "$project_dir:/project:ro" \
    -v "$credential_dir:/run/secrets/xysz:ro" \
    -v "$prepared_dir:/out" \
    -w /project \
    amazingdata-x86:1.1.9 \
    python scripts/collect_xysz_v4_dataset.py \
    --dataset history_stock_status \
    --output "/out/$output_name" \
    --codes "${codes[@]}" \
    --begin-date 20210101 \
    --end-date 20240102 \
    --hard-exit-after-complete
