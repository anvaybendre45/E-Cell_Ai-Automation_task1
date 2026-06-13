import re
from datasets import load_dataset
import pandas as pd

def stream_and_label_data(target_rows=3000):
    # Streams data chunks until target rows are reached and applies labels.
    dataset = load_dataset("winterForestStump/10-K_sec_filings", streaming=True)
    collected_rows = []
    
    for chunk_id in dataset.keys():
        if len(collected_rows) >= target_rows:
            break
        for row in dataset[chunk_id]:
            collected_rows.append(row)
            if len(collected_rows) == target_rows:
                break
                
    df = pd.DataFrame(collected_rows)
    mda_col = 'Management’s Discussion and Analysis of Financial Condition and Results of Operations'
    
    # Filter valid rows
    df_filtered = df[df[mda_col].fillna("").str.len() > 100].copy()
    
    # Define sentiment scoring word sets
    positive_words = {'growth', 'increase', 'profitable', 'expansion', 'gain', 'success', 'strengthen', 'improvement'}
    negative_words = {'loss', 'decline', 'risk', 'deficit', 'decrease', 'adversely', 'litigation', 'failure'}
    
    def get_sentiment(text):
        words = str(text).lower().split()
        pos_count = sum(1 for w in words if w in positive_words)
        neg_count = sum(1 for w in words if w in negative_words)
        total = pos_count + neg_count
        if total == 0: return "Neutral"
        score = (pos_count - neg_count) / total
        if score > 0.1: return "Positive"
        elif score < -0.1: return "Negative"
        return "Neutral"
        
    df_filtered['label'] = df_filtered[mda_col].apply(get_sentiment)
    return df_filtered

def clean_financial_text(text):
    # Strips HTML, formatting, numbers, and symbols from raw text strings.
    text = str(text).lower()
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    return " ".join(text.split())
