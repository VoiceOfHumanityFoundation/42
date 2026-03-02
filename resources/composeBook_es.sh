#!/bin/bash
cat ./translate/translation/spanish/title.md
hash=$(git rev-parse --verify HEAD)
echo -e "Commit-hash: ${hash}"
echo -e "\n"
cat  ./translate/translation/spanish/disclaimer.txt
echo -e "\n"
echo "\tableofcontents"
echo -e "\newpage\n"
echo -e "\n"
cat ./translate/translation/spanish/README.md
echo -e "\n"
mapfile -t title < "./translate/translation/spanish/index.txt"
j=0
for i in $(cat ../draft/index.txt)
do
j=$((j+1))
echo -e "# ${j}. ${title[j-1]}?    \n   \n"
cat "./translate/translation/spanish/${i}.txt"
echo -e "   \n"
done
