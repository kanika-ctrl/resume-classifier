import re
import pandas as pd
from typing import List

def clean_text(text: str) -> str:
    """Clean resume text by removing special characters and normalizing whitespace."""
    if pd.isna(text):
        return ""
    
    # Convert to string and lowercase
    text = str(text).lower()
    
    # Remove special characters but keep spaces
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def clean_resume_data(df: pd.DataFrame, text_column: str = 'Resume') -> pd.DataFrame:
    """Clean the entire resume dataset."""
    df_clean = df.copy()
    df_clean['cleaned_text'] = df_clean[text_column].apply(clean_text)
    return df_clean

def remove_stopwords(text: str, stopwords: List[str] = None) -> str:
    """Remove common stopwords from text."""
    if stopwords is None:
        stopwords = ['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by']
    
    words = text.split()
    filtered_words = [word for word in words if word not in stopwords]
    return ' '.join(filtered_words) 