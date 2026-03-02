#rm spanish.txt
#bash composeBook_es.sh  > spanish.txt
#pandoc en.txt -o ./book/42_en_latest.pdf
pandoc -V geometry:margin=0.625in -o ./book/42_spanish_latest.pdf spanish.txt
#git add ./book/42_en_latest.pdf
#git commit -m "Updated edition"
#git push
