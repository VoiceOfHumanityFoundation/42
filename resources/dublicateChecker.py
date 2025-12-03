# -*- coding: utf-8 -*-

from fastapi import FastAPI, Response
from pydantic import BaseModel
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain.docstore.document import Document
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
import os
import json
import datetime
import subprocess
import sys

# --- Configuration ---
SUGGESTIONS_FILE = "suggestions.jsonl"
EXTERNAL_SCRIPT_PATH = "./validator_script.py" # The local linux script to call
SIMILARITY_THRESHOLD = 0.3 # Lower value = stricter match for L2 distance (adjust based on metric)

# --- Initialize Embedder ---
model_kwargs = {'trust_remote_code': True, 'device': 'cuda:0'} 
# Fallback to cpu if cuda not available
# model_kwargs = {'trust_remote_code': True, 'device': 'cpu'}

embedder = HuggingFaceEmbeddings(
    model_name="nomic-ai/nomic-embed-text-v2-moe", 
    model_kwargs=model_kwargs
)

# --- Helper Functions ---

def load_suggestions_from_file():
    """Reads the JSONL file and returns a list of LangChain Documents."""
    docs = []
    if os.path.exists(SUGGESTIONS_FILE):
        with open(SUGGESTIONS_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    # We embed the suggestion, and store the response/date in metadata
                    doc = Document(
                        page_content=data['suggestion'], 
                        metadata={
                            "response": data.get('response'),
                            "date": data.get('date')
                        }
                    )
                    docs.append(doc)
                except json.JSONDecodeError:
                    continue
    return docs

def append_suggestion_to_file(suggestion, response):
    """Appends a new record to the JSONL file."""
    record = {
        "date": datetime.datetime.now().isoformat(),
        "suggestion": suggestion,
        "response": response
    }
    with open(SUGGESTIONS_FILE, 'a', encoding='utf-8') as f:
        f.write(json.dumps(record) + "\n")
    return record

def call_external_validator(suggestion_text):
    """
    Calls a local python script. 
    Expects the script to print the result to stdout.
    """
    try:
        # Calling a local python script in linux
        result = subprocess.check_output(
            [sys.executable, EXTERNAL_SCRIPT_PATH, suggestion_text],
            stderr=subprocess.STDOUT
        )
        return result.decode('utf-8').strip()
    except subprocess.CalledProcessError as e:
        return f"<reason>Script Error: {e.output.decode('utf-8')}</reason>"
    except Exception as e:
        return f"<reason>Execution Error: {str(e)}</reason>"

# --- Vector Store Initialization ---

print("Loading existing suggestions...")
existing_docs = load_suggestions_from_file()

if existing_docs:
    print(f"Found {len(existing_docs)} existing suggestions. Building vector store...")
    vector = FAISS.from_documents(existing_docs, embedder)
else:
    print("No existing suggestions found. Initializing empty vector store...")
    # FAISS needs at least one text to initialize the index. 
    # We add a placeholder that won't match real queries easily.
    vector = FAISS.from_texts(["__INITIALIZATION_TOKEN__"], embedder)

print("System ready.")

app = FastAPI()

class Message(BaseModel):
    message: str

@app.get("/")
async def root():
    return {"message": "Suggestion Checker System Online"}

@app.post("/rag")
async def handle_suggestion(ms: Message):
    user_suggestion = ms.message.strip()
    
    # 1. Check for duplicates using Vector Similarity
    # Note: FAISS L2 distance: 0 is identical. Higher is less similar.
    results_with_scores = vector.similarity_search_with_score(user_suggestion, k=1)
    
    is_duplicate = False
    duplicate_content = ""
    score = 1.0 # Initialize score
    
    if results_with_scores:
        best_doc, score = results_with_scores[0]
        # If the distance is very low, it's a duplicate. 
        if score < SIMILARITY_THRESHOLD and best_doc.page_content != "__INITIALIZATION_TOKEN__":
            is_duplicate = True
            duplicate_content = best_doc.page_content

    if is_duplicate:
        # If duplicate, return the rejection response and EXIT the function.
        print(f"Duplicate detected (Score: {score}): {duplicate_content}")
        return JSONResponse(content={
            "status": "duplicate",
            "message": "This suggestion has already been made.",
            "original_suggestion": duplicate_content,
            "previous_response": best_doc.metadata.get("response")
        })

    # --- UNIQUE SUGGESTION PATH (Execution continues here ONLY if NOT duplicate) ---
    
    # 2. Process New Suggestion
    print(f"<new>{user_suggestion}</new>")
    
    # 3. Call local linux script to validate the unique suggestion
    # Expected output from validator_script.py: <success>1234</success> or <reason>...</reason>
    script_response = call_external_validator(user_suggestion)
    
    # 4. Save to JSONL
    saved_record = append_suggestion_to_file(user_suggestion, script_response)
    
    # 5. Update Vector Store in Memory
    new_doc = Document(
        page_content=user_suggestion,
        metadata={"response": script_response, "date": saved_record['date']}
    )
    vector.add_documents([new_doc])
    
    return JSONResponse(content={
        "status": "processed",
        "result": script_response,
        "record": saved_record
    })