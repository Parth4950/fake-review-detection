import pandas as pd
import re
import string


def clean_text(text):
    """
    Enhanced text cleaning function for review preprocessing.
    Removes noise, normalizes text, and prepares it for TF-IDF vectorization.
    """
    # Handle None or NaN values
    if pd.isna(text) or text is None:
        return ""
    
    # Convert to string
    text = str(text)
    
    # Remove "Read More" text (case insensitive)
    text = re.sub(r'read\s+more', '', text, flags=re.IGNORECASE)

    # Remove emoji text
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               u"\U00002702-\U000027B0"  # dingbats
                               u"\U000024C2-\U0001F251"  # enclosed characters
                               "]+", flags=re.UNICODE)
    text = emoji_pattern.sub('', text)

    # Remove HTML tags
    text = re.sub('<[^<]+?>', '', text)

    # Remove URLs (http, https, www)
    text = re.sub(r'http\S+|www\.\S+|https\S+', '', text, flags=re.MULTILINE)

    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)

    # Remove extra whitespace and newlines (but keep single spaces)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r'\t+', ' ', text)

    # Convert to lowercase
    text = text.lower()

    # Remove leading/trailing whitespace
    text = text.strip()

    return text
