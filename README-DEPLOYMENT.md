# Stonkfly isolated paper-trading deployment

This installation is confined to `/home/ubuntu/stonkfly-lab` plus two small
user-level integration files:

- `/home/ubuntu/.config/systemd/user/stonkfly-paper.service`
- `/home/ubuntu/.local/bin/stonkflyctl` (a symlink)

It does not modify `/home/ubuntu/trading_agent_20260408`, install apt packages,
or contain a `.env`, Coinbase key, portfolio ID, or `--live` service argument.

## Commands

```sh
stonkflyctl status
stonkflyctl fly-status
stonkflyctl logs
stonkflyctl stop
stonkflyctl start
stonkflyctl restart
stonkflyctl verify
```

The service uses real public BTC-USDC quotes but simulated fills. Runtime state
is under `/home/ubuntu/stonkfly-lab/runs/paper`.

## Remove everything

Interactive removal:

```sh
stonkflyctl uninstall
```

Non-interactive removal (irreversible):

```sh
stonkflyctl uninstall --yes
```

The uninstall command removes only the paths listed above.

## Pinned provenance

- Repository: `https://github.com/nftechie/stonkfly.git`
- Commit: `78ef3e05ab0fa086032098558d893667068944a0`
- Python: 3.11
- Architecture: aarch64
- Data directory: `/home/ubuntu/stonkfly-lab/data`
- Paper run directory: `/home/ubuntu/stonkfly-lab/runs/paper`

On this host, the indirect `bcl==2.3.1` dependency initially built an incomplete
ARM64 wheel because its build-system requirements omit CFFI. It was rebuilt
inside this virtualenv with `--no-build-isolation`; no system package was added.
