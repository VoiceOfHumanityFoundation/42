import torch
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer
import soundfile as sf
import os
import nltk
from nltk.tokenize import sent_tokenize
nltk.download('punkt')


os.environ['TORCH_USE_CUDA_DSA'] = '1'
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

device = "cuda:0" if torch.cuda.is_available() else "cpu"
model = ParlerTTSForConditionalGeneration.from_pretrained("PHBJT/french_parler_tts_mini_v0.1").to(device)
tokenizer = AutoTokenizer.from_pretrained("PHBJT/french_parler_tts_mini_v0.1")
description = "A french male speaker delivers a slightly expressive and animated speech with a moderate speed and pitch. The recording is of very high quality, with the speaker's voice sounding clear and very close up suitable for an audiobook."
input_ids = tokenizer(description, return_tensors="pt").input_ids.to(device)

inputPath="../translate/translated/french/"
outputPath="../translate/translated/french/audio/"

def generateText(text, outputName):
        prompt_input_ids = tokenizer(text, return_tensors="pt").input_ids.to(device)
        generation = model.generate(input_ids=input_ids, prompt_input_ids=prompt_input_ids)
        audio_arr = generation.cpu().numpy().squeeze()
        sf.write(outputPath+outputName+".wav", audio_arr, model.config.sampling_rate)

'''
with open(inputPath+"README.md", 'r') as file:
        rawTitle="""
        title: "42"
subtitle: "An open source feedback framework for continous integration and application of ancient wisdom to modern knowledge to inspire peace in the age of artificial intelligence in the form of an artists manifesto."
author: "KennyAwesome"
        """
        rawTitle += file.read()
        generateText(rawTitle, "00_title")

with open(inputPath+"disclaimer.txt", 'r') as file:
        rawTitle = file.read()
        generateText(rawTitle, "99_disclaimer")
        '''

with open("../../draft/index.txt", "r") as index_file:
    titles = [line.strip() for line in index_file]
    # Process each title
    for j, title in enumerate(titles, 1):
        counter = 0
        with open(inputPath+title+".txt", 'r') as file:
                text ="Chapitres "+str(j)+". {title}."
                text += file.read()
                print (text)
                sentences = sent_tokenize(text)
                for i, sentence in enumerate(sentences):
                        print(f"Sentence {i+1}: {sentence}")
                        generateText(text, f'{j:02}_'+f'{counter:03}_'+title)