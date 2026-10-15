#!/bin/bash
# Build the Greek edition PDF.
# Greek needs a Unicode-capable engine and font -> xelatex + Noto Serif.
# Push happens only when GH_TOKEN is provided (docker build env injects it).
set -e
# strip any echoed batch markers from translated files (retry-path artifact)
for f in ./translate/translated/greek/*.txt; do
    sed -i '1{/^=== CHAPTER:/d}' "$f"
done
rm -f ./translate/translated/greek/greek.txt
bash composeBook_greek.sh > ./translate/translated/greek/greek.txt
pandoc --pdf-engine=xelatex --template=/opt/pandoc-template.latex \
    -V mainfont="Noto Serif" -V geometry:margin=1in \
    -o ./book/42_greek_latest.pdf ./translate/translated/greek/greek.txt
git add ./book/42_greek_latest.pdf
git commit -m "Updated Greek edition"
if [ -n "$GH_TOKEN" ]; then
    git push
else
    echo "GH_TOKEN not set - committed locally, skipping push"
fi
