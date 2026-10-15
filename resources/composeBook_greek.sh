#!/bin/bash
# Compose the Greek edition from the translated files (translated/greek/).
cat ./translate/translated/greek/title.txt
hash=$(git rev-parse --verify HEAD)
echo -e "Commit-hash: ${hash}"
echo -e "\n"
cat  ./translate/translated/greek/disclaimer.txt
echo -e "\n"
echo "\tableofcontents"
echo "\newpage"
echo -e "\n"
cat ./translate/translated/greek/README.txt
echo -e "\n"
mapfile -t title < "./translate/translated/greek/index.txt"
j=0
for i in $(cat ../draft/index.txt)
do
j=$((j+1))
echo -e "# ${j}. ${title[j-1]}?    \n   \n"
cat "./translate/translated/greek/${i}.txt" | sed 's/\// \/ /g'
echo -e "   \n"
done
echo "\newpage"
echo -e "\n"
echo -e "# Backcover   \n   \n"
cat ./translate/translated/greek/backcover.txt
echo -e "\n"
