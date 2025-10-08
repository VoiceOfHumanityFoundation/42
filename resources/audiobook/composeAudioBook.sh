#!/bin/bash
echo -e "Title: 42. Subtitle: An open source feedback framework for continous integration and application of ancient wisdom to modern knowledge to inspire peace in the age of artificial intelligence in the form of an artists manifesto. author: KennyAwesome"
cat ../README.md
echo -e "\n"
mapfile -t title < "../index.txt"
j=0
for i in $(cat ../index.txt)
do
j=$((j+1))
echo -e "# ${j}. ${i}   \n   \n"
cat "../${i}.txt"
echo -e "   \n"
done
