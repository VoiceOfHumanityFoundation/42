# Import necessary libraries
from langchain_core.prompts import ChatPromptTemplate  # For creating prompt templates
from langchain_core.output_parsers import StrOutputParser  # For processing output text
from langchain_openai import ChatOpenAI  # llama.cpp server (OpenAI-compatible API)
import os
import re
import time

#language="english"
#language="german"
#language="japanese"
#language="bengali"
#language="chinese"
language="greek"
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

regenerateFrom=12
regenerateUntil=42
# Chapters already translated with Q4_K_M - skip on the resumed run.
# (imagination=12 and courage=13 are NOT done - their batch was truncated)
SKIP_CHAPTERS = {"awakening", "peace", "good", "diversity", "needs", "character",
                 "faith", "home", "curiosity", "feedback", "hope"}

# Full-context batching: bonsai2 has 262K ctx total = 2 slots x 131072 each.
# Chapters are packed into batches of ~BATCH_INPUT_BUDGET estimated input tokens;
# input + output stays well under the 131072 slot context, so every request
# actually uses the context window instead of one small chapter per request.
BATCH_INPUT_BUDGET = 40000   # est. input tokens per request
SLOT_CTX = 131072            # per-slot context on bonsai2
CHARS_PER_TOKEN = 3.2        # conservative estimate for English -> Qwen tokenizer

inputPath="../../draft/"
outputPath="./translated/"+language+"/"

# llama.cpp server on bonsai2 host (NUC): qwen38 container, real Qwen3.8-27B Q4_K_M
# (replaced the Ternary-Bonsai PQ2_0 for better Greek quality; 262K-capable model, 64K ctx set)
# Thinking control: canonical llama.cpp mechanism chat_template_kwargs (same as uiqwen-mcp);
# renders the empty <think></think> block -> ZERO reasoning tokens.
model = ChatOpenAI(base_url="http://192.168.193.79:8080/v1",
                   api_key="none",
                   model="/models/lmstudio-community/Qwen3.8-27B-Q4_K_M.gguf",
                   temperature=0,
                   request_timeout=2400,  # hard cap per request; retries handle transient hangs
                   extra_body={"chat_template_kwargs": {"enable_thinking": False}})

# Define a generic prompt template for language translation
generic_template = """You are a concise and professional translator. Translate the following text into {language}. Translate every word faithfully: do not omit, add, summarize, paraphrase, explain, analyze, judge, or comment on the content. Preserve the exact meaning and tone of the original. Translate modal verbs such as 'might', 'could', 'would', and 'should' as their modal equivalents in {language}; never flatten them into plain statements. Translate the generic pronoun 'one' (as in 'one might') using the target language's generic pronoun. Keep the proper noun 'Pro' untranslated. Preserve all formatting, punctuation, quotation marks, and line breaks. Translate questions as questions; do not answer them. Avoid unnecessary foreign words. Output only the translation.

The input contains multiple chapters. Each chapter starts with a line exactly like:
=== CHAPTER: <title> ===
Translate EVERY chapter completely and in order. Before each translated chapter, repeat the exact same marker line for that chapter:
=== CHAPTER: <title> ===
Do not skip, merge, or reorder any chapter, and do not add anything outside the markers."""

# Create a chat-based prompt template using the defined generic template
prompt = ChatPromptTemplate.from_messages(
    [("system", generic_template), ("user", "{text}")]
)

# Set up the output parser to process the model's output as a string
parser = StrOutputParser()

# Chain the prompt, model, and parser together
chain = prompt | model | parser

MARKER_RE = re.compile(r"^===\s*CHAPTER:\s*(.+?)\s*===\s*$")

def est_tokens(s):
    return max(1, len(s) // CHARS_PER_TOKEN)

def write_output(title, text):
    filename = outputPath + title + ".txt"
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w') as f:
        f.write(text)
    print(f"  wrote {title}.txt ({len(text)} chars)", flush=True)

def invoke_with_retry(inputs, retries=3):
    """Invoke the chain with retry + backoff so a single timeout cannot
    wipe a whole chapter's translation (it is retried, then re-queued)."""
    for attempt in range(1, retries + 1):
        try:
            return chain.invoke(inputs)
        except Exception as e:
            print(f"  !! invoke error (attempt {attempt}/{retries}): {type(e).__name__}: {e}", flush=True)
            if attempt == retries:
                raise
            time.sleep(5 * attempt)

def translate_batch(files):
    """files: list of (title, content). Translate all in one full-context request.
    Any failure degrades to per-chapter singles - a batch timeout never loses
    more than the in-flight request, and each chapter is recovered individually."""
    input_text = "\n\n".join(f"=== CHAPTER: {title} ===\n{content}" for title, content in files)
    in_tokens = est_tokens(input_text)
    print(f"  batch: {len(files)} files, ~{in_tokens} in-tokens", flush=True)
    try:
        out = invoke_with_retry({"language": language, "text": input_text})
    except Exception as e:
        print(f"  !! batch failed after retries ({e}); degrading to per-chapter singles", flush=True)
        return set()
    # Split the output on the repeated marker lines.
    sections = {}
    current = None
    for line in out.splitlines():
        m = MARKER_RE.match(line.strip())
        if m:
            current = m.group(1).strip()
            sections.setdefault(current, [])
        elif current is not None:
            sections[current].append(line)
    # Write what we got; anything missing/empty gets a single-chapter retry.
    done = set()
    for title, _ in files:
        text = "\n".join(sections.get(title, [])).strip()
        if text:
            write_output(title, text)
            done.add(title)
        else:
            print(f"  !! missing/empty in batch: {title}", flush=True)
    return done

def translate_single(title, content):
    """Fallback: translate one chapter on its own. Retries on transient errors;
    on persistent failure returns False (never raises, so the run continues)."""
    print(f"  retry single: {title}", flush=True)
    input_text = f"=== CHAPTER: {title} ===\n{content}"
    try:
        out = invoke_with_retry({"language": language, "text": input_text})
    except Exception as e:
        print(f"  ERROR: {title} failed after retries: {type(e).__name__}: {e}", flush=True)
        return False
    # Strip a marker the model might echo back
    text = re.sub(rf"^===\s*CHAPTER:\s*{re.escape(title)}\s*===\s*\n?", "", out).strip()
    if text:
        write_output(title, text)
        return True
    print(f"  ERROR: retry produced nothing for {title}", flush=True)
    return False

def batch_files(files, budget=BATCH_INPUT_BUDGET):
    """Greedy-pack (title, content) pairs into batches within the input budget."""
    batches, cur, cur_tokens = [], [], 0
    for f in files:
        t = est_tokens(f[1]) + 8  # marker overhead
        if cur and cur_tokens + t > budget:
            batches.append(cur)
            cur, cur_tokens = [], 0
        cur.append(f)
        cur_tokens += t
    if cur:
        batches.append(cur)
    return batches

# --- intro files (only on a full run) ---
if regenerateFrom == 0:
    intro = []
    for name in ("disclaimer.txt", "index.txt", "title.md", "backcover.txt"):
        with open(inputPath + name, 'r') as file:
            intro.append((os.path.splitext(name)[0], file.read()))
    with open("../../README.md", 'r') as file:
        intro.append(("README", file.read()))
    for batch in batch_files(intro):
        translate_batch(batch)

# --- book chapters ---
with open(inputPath + "index.txt", "r") as index_file:
    titles = [line.strip() for line in index_file]
    selected = []
    for j, title in enumerate(titles, 1):
        print(j, flush=True)
        if j >= regenerateFrom and j <= regenerateUntil and title not in SKIP_CHAPTERS:
            with open(inputPath + title + ".txt", 'r') as file:
                selected.append((title, file.read()))

    failed = []
    for batch in batch_files(selected):
        done = translate_batch(batch)
        # single-chapter fallback for anything the batch run dropped
        for title, content in batch:
            if title not in done:
                if not translate_single(title, content):
                    failed.append(title)

    if failed:
        print(f"!! PERMANENT FAILURES (translate these manually): {failed}", flush=True)
    else:
        print("all selected chapters translated", flush=True)

print("done", flush=True)
