rm ./translate/translation/french/french.txt
bash composeBook_fr.sh > ./translate/translation/french/french.txt
pandoc -V geometry:margin=1in -o ./book/42_french_latest.pdf translate/translation/french/french.txt
#pandoc en.txt -o ./book/42_en_latest.pdf
git add ./book/42_french_latest.pdf
git commit -m "Updated edition"
git push
