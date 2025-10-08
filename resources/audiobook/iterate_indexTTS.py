import os
import inflect

from indextts.infer import IndexTTS
tts = IndexTTS(model_dir="checkpoints",cfg_path="checkpoints/config.yaml")
voice="kenny_short.wav"

inputPath="../OnTheWayTo.Pro/"
outputPath="./output/"
p = inflect.engine()

def generateText(text, outputName):
	output_path=outputPath+outputName+".wav"
	tts.infer(voice, text, output_path)
'''      
with open(inputPath+"README.md", 'r') as file:
        rawTitle="""
        title: "42"
subtitle: "An open source feedback framework for continous integration and application of ancient wisdom to modern knowledge to inspire peace in the age of artificial intelligence in the form of an artists manifesto."
author: "KennyAwesome"
        """
        rawTitle += file.read()
        generateText(rawTitle, "00_title")
'''
with open(inputPath+"disclaimer.txt", 'r') as file:
        rawTitle = file.read()
        generateText(rawTitle, "99_disclaimer")

with open(inputPath+"index.txt", "r") as index_file:
    titles = [line.strip() for line in index_file]
    # Process each title
    for j, title in enumerate(titles, 1):
        with open(inputPath+title+".txt", 'r') as file:
                text ="Chapter "+p.number_to_words(j)+". {title}"
                text += file.read()
                print (text)
                output_path=outputPath+f'{j:02}_'+title+".wav"
                if(False):
                        tts.infer(voice, text, output_path)
