#!/usr/bin/env bash
set -euo pipefail

repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
python_bin=$repo_dir/.venv/bin/python
candidate=$repo_dir/runs/wc006_v3_2021_inputs/20260918T-candidate-full-2021-2023-stateful/manifest.json
market=$repo_dir/runs/wc006_v5_market_inputs/manual-bridge/authorized-v5-development-20260919T194757Z-controller-lock/manifest.json
output_root=$repo_dir/runs
run_id="v5dev-$(date -u +%Y%m%dT%H%M%SZ)"
seed=17
max_days=0
dry_run=0
resume=0

usage() {
    sed -n '1,34p' "$0"
    cat <<'EOF'

Options:
  --seed N                 Required fixed non-negative seed.
  --run-id ID              Single-component unique run id (default: UTC timestamp).
  --manifest PATH          Frozen 46-record candidate manifest.
  --market-manifest PATH   Pinned development authorization manifest.
  --output-root PATH       Must contain runs/wc006_v5_<arm>/<run-id>.
  --max-days N             Optional bounded run for an operator-controlled pause/resume.
  --resume                 Resume existing arm directories after preflight.
  --dry-run                Preflight and print exact arm commands; do not train.
  -h, --help               Show this help.
EOF
}

while (($#)); do
    case "$1" in
        --seed) seed=$2; shift 2 ;;
        --run-id) run_id=$2; shift 2 ;;
        --manifest) candidate=$2; shift 2 ;;
        --market-manifest) market=$2; shift 2 ;;
        --output-root) output_root=$2; shift 2 ;;
        --max-days) max_days=$2; shift 2 ;;
        --resume) resume=1; shift ;;
        --dry-run) dry_run=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done

if [[ -z "$seed" || ! "$seed" =~ ^[0-9]+$ ]]; then
    echo "--seed is required and must be a non-negative integer" >&2
    exit 2
fi
if [[ ! "$run_id" =~ ^[A-Za-z0-9._-]+$ ]]; then
    echo "--run-id must be one path component" >&2
    exit 2
fi
if [[ ! -x "$python_bin" ]]; then
    echo "missing project venv: $python_bin" >&2
    exit 2
fi

export PYTHONPATH=$repo_dir
output_root=$(realpath -m "$output_root")
mkdir -p "$output_root"
candidate=$(realpath "$candidate")
market=$(realpath "$market")
preflight_cmd=("$python_bin" "$repo_dir/tools/v5_formal_preflight.py" --manifest "$candidate" --market-manifest "$market" --output-root "$output_root" --run-id "$run_id" --seed "$seed")
if (( resume )); then preflight_cmd+=(--resume); fi
"${preflight_cmd[@]}" > "$output_root/.v5-preflight-$run_id.json"

if (( dry_run )); then
    echo "preflight: PASS"
    echo "seed: $seed"
    echo "run_id: $run_id"
    for arm in attribution-only state-only combined; do
        output=$output_root/wc006_v5_$arm/$run_id
        printf 'command: '
        printf '%q ' "$python_bin" "$repo_dir/experiments/a_share/wc006_male_cns_v5_adapter.py" --manifest "$candidate" --out "$output" --market-manifest "$market" --arm "$arm" --seed "$seed" --max-days "$max_days"
        printf '\n'
    done
    exit 0
fi

# Code-level parity/resume/isolation tests are a required preflight gate before
# any formal arm.  They use temporary fixtures and never open a broker.
"$python_bin" -m pytest -q "$repo_dir/tests/test_wc006_male_cns_v5_adapter.py" -k 'resume_parity_gate_covers_all_arms or input_and_output_scope_gates or formal_accepts_pinned_development_authorization'

for arm in attribution-only state-only combined; do
    output=$output_root/wc006_v5_$arm/$run_id
    if (( ! resume )); then mkdir -p "$output"; fi
    log=$output/training.log
    tmp=$(mktemp -d)
    trap 'rm -rf "$tmp"' EXIT
    cmd=("$python_bin" "$repo_dir/experiments/a_share/wc006_male_cns_v5_adapter.py" --manifest "$candidate" --out "$output" --market-manifest "$market" --arm "$arm" --seed "$seed" --max-days "$max_days")
    if (( resume )); then
        {
            printf '\n[launcher] resume_started_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
            printf '[launcher] arm=%s seed=%s run_id=%s\n' "$arm" "$seed" "$run_id"
            printf '[launcher] exact_command='; printf '%q ' "${cmd[@]}"; printf '\n'
            printf '[launcher] stdout/stderr captured below after process exit\n'
        } >> "$log"
    else
        {
            printf '[launcher] started_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
            printf '[launcher] arm=%s seed=%s run_id=%s\n' "$arm" "$seed" "$run_id"
            printf '[launcher] exact_command='; printf '%q ' "${cmd[@]}"; printf '\n'
            printf '[launcher] stdout/stderr captured below after process exit\n'
        } > "$log"
    fi
    set +e
    "${cmd[@]}" > "$tmp/stdout" 2> "$tmp/stderr"
    rc=$?
    set -e
    {
        printf '\n----- stdout -----\n'
        sed 's/^/[stdout] /' "$tmp/stdout"
        printf '\n----- stderr -----\n'
        sed 's/^/[stderr] /' "$tmp/stderr"
        printf '\n[launcher] exit_code=%s\n' "$rc"
    } >> "$log"
    if (( rc != 0 )); then
        printf '[launcher] final_status=FAILED; stopping subsequent arms\n' >> "$log"
        exit "$rc"
    fi
    post_cmd=("$python_bin" "$repo_dir/tools/v5_formal_postflight.py" --output "$output" --arm "$arm")
    {
        printf '\n[launcher] postflight_command='; printf '%q ' "${post_cmd[@]}"; printf '\n'
    } >> "$log"
    set +e
    "${post_cmd[@]}" >> "$log" 2>&1
    post_rc=$?
    set -e
    if (( post_rc != 0 )); then
        printf '[launcher] final_status=POSTFLIGHT_FAILED; stopping subsequent arms\n' >> "$log"
        exit "$post_rc"
    fi
    printf '[launcher] final_status=PASS\n' >> "$log"
    rm -rf "$tmp"
    trap - EXIT
    echo "completed arm=$arm output=$output"
done
