#!/bin/bash
# Compose the Turkish edition from the translated files (translated/turkish/).
cat ./translate/translated/turkish/title.txt
hash=$(git rev-parse --verify HEAD)
echo -e "Commit-hash: ${hash}"
echo -e "\n"
cat  ./translate/translated/turkish/disclaimer.txt
echo -e "\n"
echo "\tableofcontents"
echo "\newpage"
echo -e "\n"
cat ./translate/translated/turkish/README.txt
echo -e "\n"
mapfile -t title < "./translate/translated/turkish/index.txt"
j=0
for i in $(cat ../draft/index.txt)
do
j=$((j+1))
echo -e "# ${j}. ${title[j-1]}?    \n   \n"
cat "./translate/translated/turkish/${i}.txt" | sed 's/\// \/ /g'
echo -e "   \n"
done
echo "\newpage"
echo -e "\n"
echo -e "# Backcover   \n   \n"
cat ./translate/translated/turkish/backcover.txt
echo -e "\n"
