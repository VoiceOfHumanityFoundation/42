# Isolated PDF build environment for the 42 book (esp. Greek edition).
#
# REPRODUCIBILITY & SECURITY
# - Base image is pinned by DIGEST (immutable) - every rebuild is identical.
# - Apt packages come from the immutable bookworm release (versions fixed).
# - This image contains NO credentials, NO tokens, NO host paths:
#   * no ENV/ARG secrets - docker history is clean
#   * credentials are injected ONLY at runtime: `docker run -e GH_TOKEN=...`
#   * entrypoint.sh authenticates gh purely from that env var
#   * /root/.config/gh is written at runtime, inside the container, ephemeral
#   * the host's ~/.config/gh, ~/.ssh, git credential helpers are never mounted
# - Build anywhere: `docker build -t 42-pdf-build .` (context = this dir only)
FROM debian:bookworm-slim@sha256:7c7b2c966bc9ee8cedfeef67e0e279108992c77681fa595db4a9d65c06ccc587

RUN apt-get update && apt-get install -y --no-install-recommends \
    pandoc \
    texlive-xetex \
    texlive-latex-extra \
    texlive-fonts-recommended \
    texlive-fonts-extra \
    fonts-noto-core \
    fonts-noto \
    fonts-noto-cjk \
    git \
    gh \
    ca-certificates \
    bash \
    && rm -rf /var/lib/apt/lists/*

COPY entrypoint.sh /usr/local/bin/entrypoint.sh
COPY pandoc-template.latex /opt/pandoc-template.latex
RUN chmod +x /usr/local/bin/entrypoint.sh

WORKDIR /repo
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
