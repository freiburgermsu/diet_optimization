#!/usr/bin/env bash
# Download and verify pinned external data per data_sources.yaml.
# Exits non-zero if any checksum diverges — safe to call in CI.
#
# Usage: scripts/fetch_data.sh [--update-hashes]
#   --update-hashes    compute and print new sha256 for each file without
#                      verifying (used when bumping a dataset version)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
MANIFEST="$ROOT/data_sources.yaml"

UPDATE_HASHES=false
[[ "${1:-}" == "--update-hashes" ]] && UPDATE_HASHES=true

command -v python3 >/dev/null || { echo "python3 required" >&2; exit 1; }
command -v curl    >/dev/null || { echo "curl required"    >&2; exit 1; }
command -v shasum  >/dev/null || { echo "shasum required"  >&2; exit 1; }

python3 -c "import yaml" 2>/dev/null || { echo "python3 -m pip install pyyaml" >&2; exit 1; }

python3 - "$MANIFEST" "$ROOT" "$UPDATE_HASHES" <<'PY'
import hashlib, os, subprocess, sys, yaml

manifest_path, root, update_hashes_str = sys.argv[1:4]
update_hashes = update_hashes_str == "True"

with open(manifest_path) as f:
    manifest = yaml.safe_load(f)

failed = []
for name, entry in manifest.items():
    local = os.path.join(root, entry["local_path"])
    url = entry["url"]
    pinned = entry.get("sha256", "TBD")

    if not os.path.exists(local):
        os.makedirs(os.path.dirname(local) or ".", exist_ok=True)
        print(f"[fetch] {name}: {url} -> {local}")
        subprocess.check_call(["curl", "-fsSL", "-o", local, url])

    h = hashlib.sha256()
    with open(local, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    actual = h.hexdigest()

    if update_hashes or pinned == "TBD":
        print(f"[hash ] {name}: {actual}")
        continue

    if actual != pinned:
        failed.append(f"{name}: expected {pinned}, got {actual}")
    else:
        print(f"[ok   ] {name}")

if failed:
    print("\nCHECKSUM MISMATCH:", *failed, sep="\n  ", file=sys.stderr)
    sys.exit(2)
PY
