#!/bin/bash
# Entrypoint: set up GitHub auth ONLY from the injected GH_TOKEN env var.
# Nothing is read from the host (no mounts of ~/.config/gh, ~/.ssh, or git configs).
set -e

# the mounted repo belongs to the host user; allow git ops inside the container
git config --global --add safe.directory /repo

if [ -n "$GH_TOKEN" ]; then
    # GH_TOKEN env already authenticates all gh calls (auth login --with-token
    # would HANG here: gh detects the env token and blocks on stdin)
    gh auth setup-git
    git config --global user.name "${GIT_USER_NAME:-42 Builder}"
    git config --global user.email "${GIT_USER_EMAIL:-42@voiceofhumanity.foundation}"
else
    echo "GH_TOKEN not set - running without push capability"
fi

exec "$@"
