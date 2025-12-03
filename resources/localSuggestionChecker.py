# -*- coding: utf-8 -*-

from fastapi import FastAPI, Response
from pydantic import BaseModel
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains.llm import LLMChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import PyPDFLoader
import re
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.embeddings import HuggingFaceEmbeddings
#import torch
import os
import gc
import json
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from langchain_community.embeddings import OllamaEmbeddings

promptpath = "../draft/prompt.txt"

model_kwargs = {'trust_remote_code': True, 'device': 'cuda:0'}
#model_kwargs = {'trust_remote_code': True, 'device': 'cpu'}
embedder = HuggingFaceEmbeddings(model_name="nomic-ai/nomic-embed-text-v2-moe", model_kwargs=model_kwargs)
#embedder = OllamaEmbeddings(model="embeddinggemma")
print("Loading document...")
loader = PyPDFLoader("./book/42_en_latest.pdf")
docs = loader.load()
print("Loading model ok")
# Split into chunks
text_splitter = SemanticChunker(embedder)
print("Splitting document...")
documents = text_splitter.split_documents(docs)
print("Splitting document ok")
# Instantiate the embedding model
#embedder = HuggingFaceEmbeddings()
# Create the vector store and fill it with embeddings
print("Loading vector...")
vector = FAISS.from_documents(documents, embedder)
print("Loading vector ok")
print("Loading Retriever...")
retriever = vector.as_retriever(search_type="similarity", search_kwargs={"k": 3})
print("Loading Retriever ok")
# Define llm
#llm = Ollama(model="hf.co/lmstudio-community/Qwen3-30B-A3B-GGUF:Q3_K_L", keep_alive="-1m")
llm = Ollama(model="gemma3n:e4b", keep_alive="-1m")
#llm = Ollama(model="hf.co/lmstudio-community/Qwen3-30B-A3B-GGUF:Q4_K_M", keep_alive="-1m")
#llm = Ollama(model="mistral-small3.2:24b-instruct-2506-q4_K_M", keep_alive="-1m")
print("Loading model...")
#llm = Ollama(model="sammcj/qwen2.5-coder-7b-instruct:q8_0", keep_alive="-1m")
# Define the prompt
print("Loading model ok")
prompt = """
You are an expert content suggestion checker. Your task is to evaluate a user's input based on the provided context (a book or document).

**Evaluation Rules:**
1. Evaluate the user input by iterating over every relevant chapter from the context. Determine if the user input might be in logical contradiction of the context.
2. If the user input is logically coherent or resolvable as a paradox, reply with a truly random 4-digit number wrapped in the tag: <success>XXXX</success>.
3. If the suggestion is rejected, give the reason as short as possible (less than 100 characters) wrapped in the tag: <reason>the reason for rejection</reason>.

Context:
{context}

User Input to Evaluate: {question}
"""


QA_CHAIN_PROMPT = PromptTemplate.from_template(prompt)
llm_chain = LLMChain(
    llm=llm,
    prompt=QA_CHAIN_PROMPT,
    callbacks=None,
    verbose=True)

document_prompt = PromptTemplate(
    input_variables=["page_content", "source"],
    template="Context:\ncontent:{page_content}\nsource:{source}",
)

combine_documents_chain = StuffDocumentsChain(
    llm_chain=llm_chain,
    document_variable_name="context",
    document_prompt=document_prompt,
    callbacks=None)

qa = RetrievalQA(
    combine_documents_chain=combine_documents_chain,
    verbose=True,
    retriever=retriever,
    return_source_documents=True)

app = FastAPI()

print("ready")

class Message(BaseModel):
    message: str

@app.get("/")
async def root():
    return {"message": "How might might weall be on the way to Pro?"}

@app.post("/echo")
async def handle_echo(ms: Message):
    ms.message = ms.message + " " + ms.message
    return ms

@app.post("/chunker")
async def handle_echo(ms: Message):
    #ms.message = ms.message
    # Split the string into semantic chunks
    chunks = text_splitter.create_documents([ms.message])
    # Print the resulting chunksfor chunk in chunks:
    print(chunks)
    return {'message': chunks}

@app.post("/rag")
async def handle_echo(ms: Message):
    #ms.message = ms.message
    # Split the string into semantic chunks
    result = []
    chunks = text_splitter.create_documents([ms.message])
    for chunk in chunks:
        print (chunk.page_content)
        res = (re.sub(r"<think>.*?</think>\s*", "", qa(chunk.page_content)["result"], flags=re.DOTALL))
        result.append(res)
        result.append("{" + chunk.page_content +"}")
    # Print the resulting chunksfor chunk in chunks:
    json_compatible_item_data = jsonable_encoder(result)
    return JSONResponse(content=json_compatible_item_data)
    """
    headers = {"Content-Disposition": 'inline; filename="out.json"'}
    return Response(
        json.dumps(result, ensure_ascii=False),
        headers=headers,
        media_type="application/json",
    )"""
#return json.dumps(result)
