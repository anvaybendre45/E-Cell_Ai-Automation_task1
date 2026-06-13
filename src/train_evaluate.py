from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, f1_score
import joblib
import os

# Import module blocks from other files
from preprocess import stream_and_label_data, clean_financial_text
from features import extract_tfidf_features

def run_production_pipeline():
    print("Executing full pipeline...")
    
    # Load data
    df = stream_and_label_data(target_rows=3000)
    
    # Preprocess text
    mda_col = 'Management’s Discussion and Analysis of Financial Condition and Results of Operations'
    df['cleaned_text'] = df[mda_col].apply(clean_financial_text)
    
    # Engineer Features
    X_matrix, vectorizer = extract_tfidf_features(df['cleaned_text'])
    
    label_map = {'Positive': 1, 'Negative': 0, 'Neutral': 2}
    y = df['label'].map(label_map).values
    
    # Split and Train
    X_train, X_test, y_train, y_test = train_test_split(X_matrix, y, test_size=0.2, random_state=42)
    
    print(f"Training CatBoost on {X_train.shape[0]} samples...")
    cat_model = CatBoostClassifier(verbose=0)
    cat_model.fit(X_train, y_train)
    
    # Evaluate
    preds = cat_model.predict(X_test)
    print(f"\nFinal Test Accuracy: {accuracy_score(y_test, preds):.4f}")
    print(f"Final Test F1-Score: {f1_score(y_test, preds, average='macro'):.4f}")
    
    # Save assets to parent root directory
    joblib.dump(cat_model, '../catboost_sentiment_model.pkl')
    joblib.dump(vectorizer, '../tfidf_vectorizer.pkl')
    print("\nProduction models successfully serialized to root directory folder!")

if __name__ == "__main__":
    run_production_pipeline()
