#!/bin/bash
# Run the isolated Greek PDF build with the migration gh credentials.
# Reads the token from ~/docs/gh-auth-migration (private, 0700) and passes it
# to the docker build via env only - never prints or persists it.
set -e

MIG_DIR="${MIG_DIR:-/home/zoon/docs/gh-auth-migration}"
HOSTS="$MIG_DIR/.config/gh/hosts.yml"

if [ ! -f "$HOSTS" ]; then
    echo "ERROR: migration credentials not found at $HOSTS" >&2
    exit 1
fi

# Extract oauth_token for github.com without echoing it
TOKEN=$(python3 - "$HOSTS" <<'PY'
import sys, yaml
d = yaml.safe_load(open(sys.argv[1]))
print(d["github.com"]["oauth_token"])
PY
)

export GH_TOKEN="$TOKEN"
exec "$(dirname "$0")/build_greek_pdf.sh" "$@"
