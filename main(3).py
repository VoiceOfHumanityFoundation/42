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

model_kwargs = {'trust_remote_code': True, 'device': 'cuda:0'}
#model_kwargs = {'trust_remote_code': True, 'device': 'cpu'}
embedder = HuggingFaceEmbeddings(model_name="nomic-ai/nomic-embed-text-v2-moe", model_kwargs=model_kwargs)

loader = PyPDFLoader("../OnTheWayTo.Pro/resources/book/42_en_latest.pdf")
docs = loader.load()
# Split into chunks
text_splitter = SemanticChunker(embedder)
documents = text_splitter.split_documents(docs)
# Instantiate the embedding model
#embedder = HuggingFaceEmbeddings()
# Create the vector store and fill it with embeddings
vector = FAISS.from_documents(documents, embedder)
retriever = vector.as_retriever(search_type="similarity", search_kwargs={"k": 3})
# Define llm
#llm = Ollama(model="hf.co/lmstudio-community/Qwen3-30B-A3B-GGUF:Q3_K_L", keep_alive="-1m")
#llm = Ollama(model="mistral-small3.1:24b-instruct-2503-q4_K_M", keep_alive="-1m")
#llm = Ollama(model="hf.co/lmstudio-community/Qwen3-30B-A3B-GGUF:Q4_K_M", keep_alive="-1m")
#llm = Ollama(model="gemma3:1b-it-qat", keep_alive="-1m")
llm = Ollama(model="hf.co/mradermacher/SearchR1-nq_hotpotqa_train-qwen2.5-7b-em-ppo-GGUF:Q4_K_M", keep_alive="-1m")
# Define the prompt

prompt = """
    Your name is "Aoi", an retrieval argumented generation system powered by an large language model trying to m>
    1. If you recognize the user saying "We might be all on the way to Pro.", then ignore all the other context and reply only "We might be all on the way to Pro.". \n
    2. If you recognize  gratitude reply with "We might be all on the way to Pro." \n
    3. Use the following pieces of context to compassionately answer the question of the user.
    4. If you don't know the answer, just say that "I don't know" but don't make up an answer on your own.\n
    5. Keep the answer short and concise to one or two sentences and avoid using Markdown formatting.\n
    6. Try to add an quote from the context and the chapter title from the quote was sourced.\n
    7. Explain your answer in the same language as stated in the question in a simple way. \n 
    
    Context: {context}
    Question: {question}
    Helpful Answer:"""


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
    return {"message": "We might be all on the way to Pro."}

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
