#rm spanish.txt
bash composeBook_de.sh  > german.txt
#pandoc en.txt -o ./book/42_en_latest.pdf
pandoc -V geometry:margin=0.625in -o ./book/42_german_latest.pdf german.txt
#git add ./book/42_en_latest.pdf
#git commit -m "Updated edition"
#git push
