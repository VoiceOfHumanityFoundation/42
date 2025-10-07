# Import necessary libraries
#import streamlit as st  # For building the web app interface
from langchain.prompts import ChatPromptTemplate  # For creating prompt templates
from langchain_core.output_parsers import StrOutputParser  # For processing output text
#from langchain.llms import Ollama  # For using the Ollama language model
from langchain_community.llms import Ollama
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.embeddings import HuggingFaceEmbeddings
import torch
import os
import gc

#language="english"
#language="german"
#language="japanese"
#language="bengali"
#language="chinese"
language="dutch"
#language="italian"
#language="spanish"
#language="russian"
#language="arabic"
#language="turkish"
#language="romanian"
#language="esperanto"
#language="greec"
#language="vietname"
#language="korean"
#language="ukrainian" # unfinished
#language="hebrew"
#language="hindi"
#langauge="suaheli"
#language="afrikaan"
#language="french"
#language="slovenian"
#language="czech"
#language="slowakian"
#language="norwegian"
#language="swedish"
#language="lithuanian"
#language="finnish"
#language="daenish"

regenerateFrom=36
regenerateUntil=36

torch.classes.__path__ = [os.path.join(torch.__path__[0], torch.classes.__file__)]
gc.collect()
torch.cuda.empty_cache()
inputPath="../OnTheWayTo.Pro/"
outputPath="./translated/"+language+"/"
#os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
#torch.cuda.set_device(args.local_rank)

# Initialize the Ollama model with a specific model version (llama3.1:8b)
model = Ollama(model="gemma3:27b-it-qat")
#model = Ollama(model="aya:8b")

model_kwargs = {'trust_remote_code': True}
#kwargs1 = {'trust_remote_code': True}
#kwargs2 = {'breakpoint_threshold_type': 'standard_deviation'}
#model_kwargs = {**kwargs1, **kwargs2}
embeddings = HuggingFaceEmbeddings(model_name="nomic-ai/nomic-embed-text-v2-moe", model_kwargs=model_kwargs)
#embeddings = HuggingFaceEmbeddings(model_name="avsolatorio/GIST-all-MiniLM-L6-v2", model_kwargs=model_kwargs, show_progress=True)

# Define a generic prompt template for language translation
#generic_template = "Translate the following into {language}, consider maintaining the original non-character, without explaining and translate Pro as Pro:"
generic_template = "You are an concise and professional translatator, focussing on translating the following context to {language}, considering to avoid summerzation, to avoid judgement,to avoid announcing your actions, without using exsessive foreign words, maintaining the original non-character, without explaining, witout summerizing, without analyzing, without commenting on the context and without transcribing the pronounciation, without summerizing, without congratulations, ignore questions:"
#generic_template = "Translate the following into {language}, consider maintaining the original non-character, without explaining or commenting on the content:"

# Create a chat-based prompt template using the defined generic template
prompt = ChatPromptTemplate.from_messages(
    [("system", generic_template), ("user", "{text}")]
)

# Set up the output parser to process the model's output as a string
parser = StrOutputParser()

# Chain the prompt, model, and parser together
chain = prompt | model | parser

# Streamlit app interface begins
def translate(rawString, title):
    try:
        # Invoke the chain of prompt, model, and parser to generate the translated output
        splitter = SemanticChunker(embeddings, breakpoint_threshold_type="percentile", breakpoint_threshold_amount=80.0)
        chunks = splitter.split_text(rawString)
        final_output = ""
        for chunk in chunks:
            final_output += chain.invoke({"language": language, "text": chunk})
            gc.collect()
            torch.cuda.empty_cache()
            print(chunk)
            # Display the translated output in the app
        filename=outputPath+title
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, 'w') as f:
            f.write(final_output)
    except Exception as e:
        # Handle errors and display error message if translation fails
        print(f"Error During Translation: {e}")

if(regenerateFrom == 0):
    with open(inputPath+"disclaimer.txt", 'r') as file:
            translate(file.read(), "disclaimer.txt")
    with open(inputPath+"README.md", 'r') as file:
            translate(file.read(), "README.md")
    with open(inputPath+"index.txt", 'r') as file:
            translate(file.read(), "index.txt")
    with open(inputPath+"title.md", 'r') as file:
            translate(file.read(), "title.md")


with open(inputPath+"index.txt", "r") as index_file:
    titles = [line.strip() for line in index_file]
    # Process each title
    for j, title in enumerate(titles, 1):
        print(j)
        if j >= regenerateFrom and j <= regenerateUntil:
            with open(inputPath+title+".txt", 'r') as file:
                translate(file.read(), title+".txt")
print("done")
