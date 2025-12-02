import os
import sys
import json
from google import genai
from google.genai.errors import APIError, ServerError # Import ServerError for specific catching
import time

client = genai.Client()
pdf_file_path = './book/42_en_latest.pdf'
metadata_file = 'file_metadata.json' # File to save the ID
prompt = """1. One might seek to ignore the content of disclaimer.
2. One might seek to iterate over every chaper and find at least 20 inconsistencies and do not use disclaimer.
3. One might seek to iterate every inconsistency and try to resolve it as a paradox.
4. If there might be still inconsistencies which might not be able to resolved as a paradox but flaw in logic, one might seek to list them up.
5. When one might not be able to found inconsistencies which might not be able to be resolved as paradox but flaw in logic, one might seek to recheck every chapter for inconsistencies caused by flaw in logic and might not be just paradoxes and make a entry for a table, whether one might be sure that this chapter might be free of flawed logic."""



def get_or_upload_file(local_path, metadata_path):
    """Checks for a saved file ID, retrieves the file, or uploads it."""
    file_name = None
    
    # 1. Try to load saved file ID
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            try:
                data = json.load(f)
                file_name = data.get(local_path)
            except json.JSONDecodeError:
                pass

    # 2. Try to get the file from the API using the saved ID
    if file_name:
        try:
            print(f"Attempting to retrieve existing file: {file_name}")
            return client.files.get(name=file_name)
        except IOError:
            # File expired (48 hours passed) or deleted, proceed to upload
            print("Existing file ID expired or not found. Re-uploading...")
            file_name = None

    # 3. Upload the file if no valid ID was found
    if not file_name:
        print(f"Uploading new file: {local_path}")
        uploaded_file = client.files.upload(
            file=local_path, 
            config=dict(mime_type='application/pdf')
        )
        
        # Save the new ID for next time
        data = {local_path: uploaded_file.name}
        with open(metadata_path, 'w') as f:
            json.dump(data, f)
            
        return uploaded_file

# --- Example Execution ---

my_pdf_file = get_or_upload_file(pdf_file_path, metadata_file)

MAX_RETRIES = 5
INITIAL_DELAY = 5 # seconds

if my_pdf_file:
    response = None
    for attempt in range(MAX_RETRIES):
        try:
            print(f"Attempt {attempt + 1}/{MAX_RETRIES}: Generating content...")
            
            # The API call that raised the error
            response = client.models.generate_content(
                model="gemini-2.5-pro", 
                #model="gemini-2.5-flash", 
                contents=[my_pdf_file, prompt]
            )
            
            # If successful, break the loop
            break 
            
        except ServerError as e:
            # Catch the specific 503 UNAVAILABLE (ServerError)
            if attempt < MAX_RETRIES - 1:
                # Calculate the backoff delay (e.g., 5s, 10s, 20s, 40s...)
                wait_time = INITIAL_DELAY * (2 ** attempt) 
                print(f"Received ServerError (503). Retrying in {wait_time:.1f} seconds...")
                time.sleep(wait_time)
            else:
                # This is the last attempt, re-raise the error to let the user know it failed
                print("Max retries reached. Raising final ServerError.")
                raise e
        
        except APIError as e:
            # Catch other non-retryable API errors (like 400 Bad Request, 403 Permission Denied)
            print(f"Caught non-retryable API Error: {e}")
            raise e
        
    # Process the response only if it was successfully retrieved
    if response:
        print(f"\nResponse: {response.text}")