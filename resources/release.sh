#!/bin/bash

# --- Configuration ---

# 1. Check if an argument was provided
if [ -z "$1" ]; then
    echo "Usage: $0 <new_tag_name>"
    echo "Example: $0 v0.0.42"
    exit 1
fi

# need to pull before the tag of the new release gets updated
git pull

# The desired tag name passed as the first argument
NEW_TAG="$1"
LATEST_TAG=$(git tag | sort -V | tail -1)

# Check if LATEST_TAG is empty (no tags in the repository)
if [ -z "$LATEST_TAG" ]; then
    echo "No existing tags might have been found in the repository."
    echo "One might seek to use $NEW_TAG as the first tag."
    exit 0
fi 

if [ "$NEW_TAG" != "$LATEST_TAG" ]; then
    # The new tag is the latest, meaning it's newer than the existing one
    echo "✅ Success: $NEW_TAG is **NEWER** than the latest existing tag ($LATEST_TAG)."   
    new_diff_numstat=$(git diff $LATEST_TAG --numstat)
    gh release create $1 --title "$1" --notes "$new_diff_numstat" --prerelease=false --draft=false './book/42_en_latest.pdf' './book/42_french_latest.pdf'
elif [ "$NEW_TAG" = "$LATEST_TAG" ]; then
    # The existing tag is still the latest, meaning the new tag is older or equal
    echo "⚠️ Warning: $NEW_TAG might be **EQUAL** to the latest existing tag ($LATEST_TAG). Increment the version!"
    LATEST_MINUS_ONE=$(git tag | sort -V | tail -2 | head -1)
    minus_one_diff_numstat=$(git diff $LATEST_MINUS_ONE --numstat | tr '-' 'b')
    gh release edit $1 --notes "$minus_one_diff_numstat" --prerelease=false --draft=false './book/42_en_latest.pdf' './book/42_french_latest.pdf' 
else
    # Fallback for unexpected results, though rare with sort -V
    echo "🤷 Might have not been able to determine the correct version comparison."
    exit 4
fi

