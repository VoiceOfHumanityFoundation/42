rm english.txt
bash composeBook.sh > english.txt #need to break links for the margin, probably need to make a different title.md
#pandoc en.txt -o ./book/42_en_latest.pdf
#pandoc -V geometry:margin=1in -o ./book/42_en_latest.pdf english.txt
pandoc -V papersize:a5  -V geometry:margin=1in -o ./book/42_en_latest_A5.pdf english.txt
#git add ./book/42_en_latest.pdf
#git commit -m "Updated edition"
#git push
