#!/bin/bash
# Build the Turkish edition PDF (Latin script - xelatex + Noto Serif works fine).
# Push happens only when GH_TOKEN is provided (docker build env injects it).
set -e
# strip any echoed batch markers from translated files (retry-path artifact)
for f in ./translate/translated/turkish/*.txt; do
    sed -i '1{/^=== CHAPTER:/d}' "$f"
done
rm -f ./translate/translated/turkish/turkish.txt
bash composeBook_turkish.sh > ./translate/translated/turkish/turkish.txt
pandoc --pdf-engine=xelatex --template=/opt/pandoc-template.latex \
    -V mainfont="Noto Serif" -V geometry:margin=1in \
    -o ./book/42_turkish_latest.pdf ./translate/translated/turkish/turkish.txt
git add ./book/42_turkish_latest.pdf
git commit -m "Updated Turkish edition"
if [ -n "$GH_TOKEN" ]; then
    git push
else
    echo "GH_TOKEN not set - committed locally, skipping push"
fi
