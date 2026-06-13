from sklearn.feature_extraction.text import TfidfVectorizer

def extract_tfidf_features(cleaned_text_series, max_features=1000):
    # Initializes and applies a TF-IDF vectorizer configuration.
    vectorizer = TfidfVectorizer(max_features=max_features)
    X_matrix = vectorizer.fit_transform(cleaned_text_series).toarray()
    return X_matrix, vectorizer
