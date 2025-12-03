# validator_script.py
import sys
import random

if __name__ == "__main__":
    # The suggestion is passed as the first argument
    suggestion = sys.argv[1] if len(sys.argv) > 1 else ""

    # YOUR LOGIC HERE
    # Example logic:
    if "bad" in suggestion.lower():
         print("<reason>contains forbidden words</reason>")
    else:
         # Success with random 4 digit number
         print(f"<success>{random.randint(1000, 9999)}</success>")