#!/bin/bash
# Build 42_greek_latest.pdf in an isolated Docker environment.
#
# Usage:
#   GH_TOKEN=<your-token> ./build_greek_pdf.sh          # build + commit + push
#   ./build_greek_pdf.sh                                # build + commit locally (no push)
#   REPO=/path/to/42 ./build_greek_pdf.sh               # use a different checkout
#
# The token is passed ONLY via env at runtime - it never lands in the image,
# and the container has no access to host gh/ssh credentials.
set -e
REPO="${REPO:-/home/zoon/projects/42}"
cd "$(dirname "$0")"

echo "== building pdf-build image =="
docker build -q -t 42-pdf-build .

echo "== running build (isolated) =="
exec docker run --rm \
    -e GH_TOKEN="${GH_TOKEN:-}" \
    -e GIT_USER_NAME="${GIT_USER_NAME:-}" \
    -e GIT_USER_EMAIL="${GIT_USER_EMAIL:-}" \
    -v "$REPO:/repo" \
    -w /repo/resources \
    42-pdf-build ./updateGreek.sh
