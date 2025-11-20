rm english.txt
bash composeBook.sh | sed 's/\// \/ /g'  > english.txt #need to break links for the margin
#pandoc en.txt -o ./book/42_en_latest.pdf
#pandoc -V geometry:margin=1in -o ./book/42_en_latest.pdf english.txt
pandoc -V papersize:a5  -V geometry:margin=1in -o ./book/42_en_latest_A5.pdf english.txt
#git add ./book/42_en_latest.pdf
#git commit -m "Updated edition"
#git push
