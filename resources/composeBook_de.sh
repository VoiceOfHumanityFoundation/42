#!/bin/bash
cat ./translate/translation/german/title.md
hash=$(git rev-parse --verify HEAD)
echo -e "Commit-hash: ${hash}"
echo  "\newpage"
echo -e "\n"
cat  ./translate/translation/german/disclaimer.txt
echo -e "\n"
echo "\tableofcontents"
echo "\newpage"
echo -e "\n"
cat ./translate/translation/german/README.md
echo -e "\n"
mapfile -t title < "./translate/translation/german/index.txt"
j=0
for i in $(cat ../draft/index.txt)
do
j=$((j+1))
echo -e "# ${j}. ${title[j-1]}?    \n   \n"
cat "./translate/translation/german/${i}.txt" | sed 's/\// \/ /g'
echo -e "   \n"
done
