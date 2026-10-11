#!/usr/bin/env bash
set -euo pipefail

repo="/home/ubuntu/stonkfly-lab"
gate_parent="$repo/runs/wc006_v5_resume_parity_v2"
manifest="$repo/runs/wc006_v3_2021_inputs/20260918T-candidate-full-2021-2023-stateful/manifest.json"
market_manifest="$repo/runs/wc006_v5_market_inputs/manual-bridge/authorized-v5-development-20260919T194757Z-controller-lock/manifest.json"
run_id="${1:?run-id required}"
root="$gate_parent/$run_id"

if [[ "$run_id" == */* || "$run_id" == *\\* || -z "$run_id" ]]; then
  echo "invalid run-id" >&2
  exit 2
fi
if [[ -e "$root" ]]; then
  echo "refusing existing parity output: $root" >&2
  exit 3
fi

mkdir -p "$gate_parent"
tmp_base="$gate_parent/.${run_id}.driver"
tmp_stdout="$tmp_base.stdout"
tmp_stderr="$tmp_base.stderr"
tmp_meta="$tmp_base.meta"
started_utc="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
python_bin="$repo/.venv/bin/python"
script="$repo/tools/run_v5_resume_parity.py"
args=(
  env "PYTHONPATH=$repo" "$python_bin" "$script"
  --manifest "$manifest"
  --market-manifest "$market_manifest"
  --gate-root "$gate_parent"
  --run-id "$run_id"
  --seed 17
  --execute
)

printf '%s\n' "driver_pid=$$" > "$tmp_meta"
printf '%s\n' "started_utc=$started_utc" >> "$tmp_meta"
printf '%q ' "${args[@]}" > "$tmp_base.command"
printf '\n' >> "$tmp_base.command"

"${args[@]}" > "$tmp_stdout" 2> "$tmp_stderr" &
child_pid=$!
printf '%s\n' "child_pid=$child_pid" >> "$tmp_meta"

while kill -0 "$child_pid" 2>/dev/null && [[ ! -e "$root" ]]; do
  sleep 1
done
if [[ ! -e "$root" ]]; then
  mkdir "$root"
fi

mv "$tmp_meta" "$root/driver.meta"
mv "$tmp_base.command" "$root/exact_command.txt"
mv "$tmp_stdout" "$root/stdout.log"
mv "$tmp_stderr" "$root/stderr.log"
printf '%s\n' "$root/stdout.log" > "$root/stdout_path.txt"
printf '%s\n' "$root/stderr.log" > "$root/stderr_path.txt"
printf '%s\n' "resume_parity_gate_only" > "$root/purpose.txt"
printf '%s\n' "formal_development=false" > "$root/formal_flag.txt"
printf '%s\n' "paper_only=true" > "$root/paper_flag.txt"
printf '%s\n' "broker_orders_enabled=false" >> "$root/paper_flag.txt"

wait "$child_pid"
status=$?
printf '%s\n' "$status" > "$root/driver_exit_code.txt"
exit "$status"
