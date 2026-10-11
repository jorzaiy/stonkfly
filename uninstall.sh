#!/usr/bin/env bash
set -euo pipefail

readonly TARGET=/home/ubuntu/stonkfly-lab
readonly UNIT=/home/ubuntu/.config/systemd/user/stonkfly-paper.service
readonly COMMAND_LINK=/home/ubuntu/.local/bin/stonkflyctl

if [[ "${1:-}" != '--yes' ]]; then
  printf '%s\n' \
    'This permanently removes the Stonkfly source, virtualenv, MaleCNS data,' \
    'paper ledger, checkpoints, logs stored in the run directory, and user service.'
  read -r -p 'Type DELETE-STONKFLY to continue: ' answer
  [[ "$answer" == 'DELETE-STONKFLY' ]] || { printf 'Cancelled.\n'; exit 1; }
fi

systemctl --user disable --now stonkfly-paper.service 2>/dev/null || true
rm -f -- "$UNIT" "$COMMAND_LINK"
systemctl --user daemon-reload
systemctl --user reset-failed stonkfly-paper.service 2>/dev/null || true
rm -rf -- "$TARGET"
printf 'Stonkfly was removed. The existing quant project was not touched.\n'
