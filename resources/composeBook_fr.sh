#!/bin/bash
cat ./translate/translation/french/title.md
hash=$(git rev-parse --verify HEAD)
echo -e "Commit-hash: ${hash}"
echo -e "\n"
cat  ./translate/translation/french/disclaimer.txt
echo -e "\n"
echo "\tableofcontents"
for i in $(seq 1 19)
do
echo -e ".\n"
done
echo -e "\n"
cat ./translate/translation/french/README.md
echo -e "\n"
mapfile -t title < "./translate/translation/french/index.txt"
j=0
for i in $(cat ../draft/index.txt)
do
j=$((j+1))
echo -e "# ${j}. ${title[j-1]}    \n   \n"
cat "./translate/translation/french/${i}.txt"
echo -e "   \n"
done
