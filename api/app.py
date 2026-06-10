from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import re
import uvicorn

# 1. Initialize the FastAPI application
app = FastAPI(
    title="10-K Document Intelligence API", 
    description="FastAPI endpoint for E-Cell AI & Automation Task 1 - Financial Sentiment Classification"
)

# 2. Load the high-performance model and vectorizer saved from your notebook
try:
    model = joblib.load('catboost_sentiment_model.pkl')
    vectorizer = joblib.load('tfidf_vectorizer.pkl')
    print("Models loaded successfully into the API!")
    
except FileNotFoundError:
    print("Error: Model files not found! Make sure you ran your serialization notebook cell.")

# 3. Define the strict JSON input requirement (Pydantic model) 
class TextRequest(BaseModel):
    text: str

# Helper text cleaning logic from Stage 1 [cite: 82]
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'<[^>]+>', ' ', text)  # Strip HTML tags
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)  # Strip numbers/punctuation
    return " ".join(text.split())

# 4. Create the required /predict POST endpoint 
@app.post("/predict")
def predict_sentiment(request: TextRequest):
    # 1. Clean the incoming raw text
    cleaned = clean_text(request.text)
    
    # 2. Convert text to numeric features using the loaded TF-IDF vectorizer
    features = vectorizer.transform([cleaned]).toarray()
    
    # 3. Predict using CatBoost and safely force it to a flat, raw integer
    raw_prediction = model.predict(features)
    
    # Safely extract the inner integer regardless of array shape
    if hasattr(raw_prediction, "item"):
        prediction_code = int(raw_prediction.item())
    else:
        prediction_code = int(raw_prediction[0])
    
    # 4. Extract confidence probabilities safely
    probabilities = model.predict_proba(features)[0]
    confidence = float(probabilities[prediction_code])
    
    # 5. Map numerical codes back to our 3 human-readable string labels
    label_map_inverse = {1: "Positive", 0: "Negative", 2: "Neutral"}
    predicted_label = label_map_inverse.get(prediction_code, "Neutral")
    
    # Return structured JSON result
    return {
        "label": predicted_label,
        "confidence": round(confidence, 2)
    }

# Entry point to run the app directly if needed
if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)