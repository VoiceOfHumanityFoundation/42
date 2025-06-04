#!/bin/bash
cat ../manifest/title.md
hash=$(git rev-parse --verify HEAD)
echo -e "Commit-hash: ${hash}"
echo -e "\n"
cat ../manifest/disclaimer.txt
echo -e "\n"
echo "\tableofcontents"
for i in $(seq 1 19)
do
echo -e ".\n"
done
echo -e "\n"
cat ../README.md
echo -e "\n"
mapfile -t title < "../manifest/index.txt"
j=0
for i in $(cat ../manifest/index.txt)
do
j=$((j+1))
echo -e "# ${j}. ${i}   \n   \n"
cat "../manifest/${i}.txt"
echo -e "   \n"
done
