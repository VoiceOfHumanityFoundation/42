# -*- coding: utf-8 -*-

from fastapi import FastAPI
from pydantic import BaseModel
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain.docstore.document import Document
from langchain.prompts import PromptTemplate
from langchain.chains.llm import LLMChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains import RetrievalQA
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
import os
import json
import datetime
import re
import sys
import subprocess
from langchain_community.document_loaders import PyPDFLoader
from langchain_experimental.text_splitter import SemanticChunker

# --- Configuration ---
SUGGESTIONS_FILE = "suggestions.jsonl"
SIMILARITY_THRESHOLD_DUPLICATE = 0.05 # VERY low L2 distance = STOLEN/DUPLICATE
SIMILARITY_THRESHOLD_RAG = 0.5        # Medium L2 distance = SIMILAR, needs LLM check
EXTERNAL_SCRIPT_PATH = "./validator_script.py"

# --- Initialize Embedder (Shared) ---
model_kwargs = {'trust_remote_code': True, 'device': 'cuda:0'}
embedder = HuggingFaceEmbeddings(
    model_name="nomic-ai/nomic-embed-text-v2-moe",
    model_kwargs=model_kwargs
)

# ==========================================
# SYSTEM 1: HISTORY MANAGEMENT
# ==========================================

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

print("--- Initializing History RAG Source... ---")
existing_docs = load_suggestions_from_file()

if existing_docs:
    print(f"Found {len(existing_docs)} existing suggestions. Building history vector store...")
    suggestion_vector = FAISS.from_documents(existing_docs, embedder)
else:
    print("No existing suggestions found. Initializing empty history vector store...")
    # Initialize with a placeholder
    suggestion_vector = FAISS.from_texts(["__INITIALIZATION_TOKEN__"], embedder)


# ==========================================
# SYSTEM 2: LLM DISCRIMINATION SETUP
# ==========================================

def check_gemini_connection() -> bool:
    """
    Attempts a minimal API call to check connectivity and authentication.
    """
    try:
        # 1. Initialize the client
        # Replace GOOGLE_API_KEY with your actual key or environment variable reference
        client = genai.Client()

        # 2. Make a minimal call (e.g., listing models)
        print("Attempting to connect and list models...")
        
        # Iterating over the response ensures the connection is actually opened and data retrieved
        models = client.models.list()
        next(models) # Try to fetch the first model to confirm a successful response

        print("✅ Connection successful. API key is valid.")
        return True
    
    except APIError as e:
        # Catches common issues like 400 (Bad Request), 403 (Permission Denied/Invalid API Key), 
        # 429 (Rate Limit), or 503 (Service Unavailable)
        print(f"❌ API Error: A connection was made, but the API key or request failed.")
        print(f"   Details: {e}")
        return False
        
    except Exception as e:
        # Catches network-level errors (ConnectionError, Timeout, DNS failure, etc.)
        # These are often wrapped in a client-level Exception
        print(f"❌ Connection Failed: A network or client-side error occurred.")
        print(f"   Details: {e}")
        return False

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

def call_internal_validator(suggestion_text):
    return(re.sub(r"<think>.*?</think>\s*", "", qa(suggestion_text)["result"], flags=re.DOTALL))

# 1. Setup LLM
print("Loading LLM...")
llm = Ollama(model="gemma3n:e4b", keep_alive="-1m")

# 2. Setup Prompt for Discrimination
DISCRIMINATION_PROMPT_TEMPLATE = """
You are an expert idea discriminator. Your task is to evaluate a NEW user suggestion based on the context of SIMILAR PAST suggestions.

Context (Similar Past Suggestions):
{context}

New User Suggestion to Evaluate: {question}

**Evaluation Rules:**
1. Check if the New Suggestion is fundamentally different, provides a major improvement, or resolves a known issue in the Context.
2. If the New Suggestion is significantly unique or valuable, reply only with <new>original user input</new> and do not explain.
3. If the New Suggestion is just a minor rephrasing, a subtle variation without new value, or is already covered, give the reason as short as possible (less than 100 characters) wrapped in the tag: <reason>the reason for rejection</reason>.
"""

DISCRIMINATION_PROMPT = PromptTemplate.from_template(DISCRIMINATION_PROMPT_TEMPLATE)

llm_chain = LLMChain(
    llm=llm,
    prompt=DISCRIMINATION_PROMPT,
    verbose=True
)

document_prompt = PromptTemplate(
    input_variables=["page_content", "source"],
    template="Past Suggestion: {page_content}",
)

# Combine documents chain to pass the retrieved documents into the LLM chain
combine_documents_chain = StuffDocumentsChain(
    llm_chain=llm_chain,
    document_variable_name="context",
    document_prompt=document_prompt,
)

# RetrievalQA chain (renamed for clarity but using same structure)
discriminator_qa = RetrievalQA(
    combine_documents_chain=combine_documents_chain,
    verbose=True,
    retriever=suggestion_vector.as_retriever(search_type="similarity", search_kwargs={"k": 2}),
    return_source_documents=True
)


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

QA_CHAIN_PROMPT = """
You are an expert content suggestion checker. Your task is to evaluate a user's input based on the provided context (a book or document).

**Evaluation Rules:**
1. Evaluate the user input by iterating over every relevant chapter from the context. Determine if the user input might be in logical contradiction of the context.
2. If the user input is logically coherent or resolvable as a paradox, reply with a truly random 4-digit number wrapped in the tag: <success>XXXX</success>.
3. If the suggestion is rejected, give the reason as short as possible (less than 100 characters) wrapped in the tag: <reason>the reason for rejection</reason>.

Context:
{context}

User Input to Evaluate: {question}
"""

rag_chain = LLMChain(
    llm=llm,
    prompt=QA_CHAIN_PROMPT,
    callbacks=None,
    verbose=True)

rag_prompt = PromptTemplate(
    input_variables=["page_content", "source"],
    template="Context:\ncontent:{page_content}\nsource:{source}",
)

combine_rag_chain = StuffDocumentsChain(
    llm_chain=rag_chain,
    document_variable_name="context",
    document_prompt=rag_prompt,
    callbacks=None)

qa = RetrievalQA(
    combine_documents_chain=combine_rag_chain,
    verbose=True,
    retriever=retriever,
    return_source_documents=True)


print("--- All Systems Ready ---")

# ==========================================
# API ENDPOINTS
# ==========================================

app = FastAPI()

class Message(BaseModel):
    message: str

@app.get("/")
async def root():
    return {"message": "History-Based Suggestion Discriminator Online"}

@app.post("/rag")
async def handle_suggestion(ms: Message):
    user_suggestion = ms.message.strip()
    
    # ---------------------------------------------------------
    # 1. Check for STRICT Duplicate (using a very low threshold)
    # ---------------------------------------------------------
    results_with_scores = suggestion_vector.similarity_search_with_score(user_suggestion, k=1)
    
    is_duplicate = False
    best_doc = None

    if results_with_scores:
        best_doc, score = results_with_scores[0]
        # Check if the score indicates a near-perfect match (strict duplicate)
        print(f"Score: {score}")
        if score < SIMILARITY_THRESHOLD_DUPLICATE and best_doc.page_content != "__INITIALIZATION_TOKEN__":
            is_duplicate = True
    if is_duplicate:
        print(f"Duplicate detected (Score: {score}): {best_doc.page_content}")
        # MODIFICATION: Return the specific message requested: <reason>not new</reason>
        return JSONResponse("<reason>not new</reason>")

    # ---------------------------------------------------------
    # 2. Process New Suggestion (Run Discrimination RAG)
    # ---------------------------------------------------------
    print(f"<new>Running discrimination for: {user_suggestion}</new>")

    #script_response = call_external_validator(user_suggestion)
    script_response = call_internal_validator(user_suggestion)
    
    # Run the Discrimination RAG chain
    # It retrieves similar past suggestions and passes them to the LLM for evaluation
    #chain_output = discriminator_qa(user_suggestion)["result"]
    
    # Clean <think> tags if using reasoning models like Gemma
    #final_response = re.sub(r"<.+?>.*?</think>\s*", "", chain_output, flags=re.DOTALL).strip()
    final_response = script_response
    # ---------------------------------------------------------
    # 3. Save to File & History Vector
    # ---------------------------------------------------------
    
    # Save to JSONL
    saved_record = append_suggestion_to_file(user_suggestion, final_response)
    
    # Update History Vector Store in Memory
    new_doc = Document(
        page_content=user_suggestion,
        metadata={"response": final_response, "date": saved_record['date']}
    )
    suggestion_vector.add_documents([new_doc])
    
    # 4. Return the result
    return JSONResponse(final_response)