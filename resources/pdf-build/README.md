# 42 PDF build environment (isolated, credential-free, reproducible)

Builds the book PDFs (esp. the Greek edition) in an isolated Docker container.

## Why Docker

- The PDF pipeline needs pandoc + XeLaTeX + Greek fonts (Noto Serif) + git/gh.
- The container is **isolated**: it has no access to your host's `~/.config/gh`,
  `~/.ssh`, or git credential helpers.
- **No credentials are baked in** — the image history contains no secrets,
  and the token is injected only at runtime via `GH_TOKEN` (see below).

## Reproducibility

- Base image pinned by **digest** (`debian:bookworm-slim@sha256:...`):
  identical image every rebuild.
- apt packages come from the immutable bookworm release (fixed versions).
- Build context is only this directory:
  ```bash
  docker build -t 42-pdf-build .
  ```
- The `pandoc-template.latex` (stock pandoc template, minus `lmodern` which
  does not exist in Debian bookworm) is committed in this directory and
  copied into the image — no external fetch at build time.

## Usage

Build the Greek PDF, commit, and push (push only if `GH_TOKEN` is set):

```bash
# build + commit + push with YOUR token (injected at runtime, never stored)
GH_TOKEN=<your-token> ./build_greek_pdf.sh

# build + commit locally, no push
./build_greek_pdf.sh
```

Inside the container, `entrypoint.sh` performs
`gh auth login --with-token` from `$GH_TOKEN`, then `updateGreek.sh` runs
`composeBook_greek.sh` → `pandoc (xelatex + Noto Serif)` → `book/42_greek_latest.pdf`
→ `git commit` → `git push`.

## Security model

| Vector | Status |
|---|---|
| Image layers / docker history | no secrets (verified) |
| env vars in image | none (verified) |
| /root config in image | absent (verified) |
| token persistence | runtime env only, container is `--rm` |
| host gh/ssh/git creds | never mounted |

Verify yourself after building:

```bash
docker run --rm 42-pdf-build env | grep -iE "token|secret|key|github"   # nothing
docker history 42-pdf-build | grep -iE "token|secret"                    # nothing
```

## Greek pipeline scripts (repo)

- `resources/composeBook_greek.sh` — assembles the book from `translate/translated/greek/`
- `resources/updateGreek.sh` — compose → pandoc → commit → push (if `GH_TOKEN`)
- `resources/pdf-build/run_pdf_build.sh` — wrapper that reads a token from a
  private local path (`~/docs/gh-auth-migration/.config/gh/hosts.yml`) and
  passes it to docker as `GH_TOKEN` — convenient on this machine, not part of
  the image.
