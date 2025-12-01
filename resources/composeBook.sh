#!/bin/bash
cat ../draft/title.md
hash=$(git rev-parse --verify HEAD)
echo -e "Commit-hash: ${hash}"
echo -e "\n"
echo "\newpage"
echo -e "\n"
cat ../draft/disclaimer.txt
echo -e "\n"
echo "\tableofcontents"
echo -e "\n"
echo "\newpage"
echo -e "\n"
cat ../README.md
echo -e "\n"
mapfile -t title < "../draft/index.txt"
j=0
for i in $(cat ../draft/index.txt)
do
j=$((j+1))
echo -e "# ${j}. ${i}   \n   \n"
cat "../draft/${i}.txt"  | sed 's/\// \/ /g'
echo -e "   \n"
done
