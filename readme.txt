================================================================================
                                 MODEL REPORT
        E-Cell AI & Automation Task 1: 10-K Document Intelligence API
================================================================================

1. PREPROCESSING DECISIONS AND TEXT CLEANING STEPS TAKEN
--------------------------------------------------------------------------------
* Target Column Selection:
  The Management’s Discussion and Analysis (MD&A) section was exclusively selected 
  as the target column for text extraction. This decision is justified because the 
  MD&A contains narrative prose regarding corporate performance expectations, forward-
  looking insights, and market trend forecasts, making it highly suitable for 
  sentiment classification, unlike unparsed numerical financial statement tables.

* Data Filtering:
  Filings with empty or extremely short MD&A text segments (less than 100 characters) 
  were filtered out to reduce noise and remove skeletal or corrupt document records.

* Automated Rule-Based Labeling Setup:
  A financial lexicon scoring approach was engineered to generate the classification 
  target. Occurrences of standard financial positive indicators (growth, profitable, 
  expansion, etc.) vs. negative indicators (loss, risk, deficit, litigation, etc.) 
  were tabulated. A continuous net score was computed and bounded using clear mathematical 
  thresholds to yield distinct multi-class ground-truth labels:
  - High/Positive Outlook (Score > 0.1) -> Class 1
  - Low/Negative Outlook (Score < -0.1) -> Class 0
  - Neutral Outlook (-0.1 <= Score <= 0.1) -> Class 2

* Text Cleaning Operations (via Python 're' and 'src/utils.py'):
  - Normalization: String text was cast completely to lowercase characters to avoid 
    vocabulary duplication.
  - HTML/XML Extraction: Leftover regulatory format tags (e.g., '<[^>]+>') were stripped.
  - Non-Alpha Removal: All numeric variables, special punctuation symbols, and arithmetic 
    operators were stripped (keeping strictly '^[a-zA-Z\s]') to minimize over-fitting on 
    arbitrary digits and specific submission fiscal years.
  - Whitespace Splitting: Consecutive spacing configurations, paragraph tabs, and trailing 
    newlines ('\n') were unified down into standard single spaces.


2. FEATURES USED AND WHY THEY WERE CHOSEN
--------------------------------------------------------------------------------
* TF-IDF (Term Frequency - Inverse Document Frequency) Vectorization:
  Text strings were converted into numerical features using 'TfidfVectorizer' bounded 
  to the top 1000 most predictive vocabulary terms ('max_features=1000').

* Why TF-IDF Was Chosen over Raw Token Counts:
  - Term Frequency (TF) scores words based on local density in a specific text document, 
    surfacing essential topic themes.
  - Inverse Document Frequency (IDF) scales down ubiquitous terms found across all 
    filings (e.g., 'the', 'company', 'financial', 'section'), ensuring that common industry 
    jargon does not overwhelm the predictive capabilities of the classifier.
  - The combination generates highly distinct, normalized continuous matrix weights bounded 
    between 0.0 and 1.0, enabling gradient boosting tree split points to separate numeric 
    feature boundaries cleanly.


3. MODEL COMPARISON, METRICS TABLE AND INTERPRETATION
--------------------------------------------------------------------------------
The performance metrics of the three mandated boosting algorithms trained across a robust 
sample of 3,000 streamed filings with a strict 80/20 train-test split are outlined below:

+--------------------+------------+------------+------------+------------+
| Model              | Accuracy   | Precision  | Recall     | F1-Score   |
+--------------------+------------+------------+------------+------------+
| XGBoost            | 0.8306     | 0.8294     | 0.8293     | 0.8290     |
| AdaBoost           | 0.7500     | 0.7620     | 0.7459     | 0.7490     |
| CatBoost (Winner)  | 0.8495     | 0.8533     | 0.8502     | 0.8478     |
+--------------------+------------+------------+------------+------------+

* Results Interpretation & Misclassification Analysis:
  - CatBoost performed exceptionally well, leading across all metrics ($84.95\%$ accuracy). 
    Its internal structure showed superior boundary parsing, misclassifying only 3 
    negative targets as positive, and maintaining a robust distribution on the neutral 
    class.
  - XGBoost achieved a highly competitive accuracy profile ($83.06\%$), but suffered minor 
    degradations around neutral margins, misclassifying several high-risk profiles 
    as neutral due to heavy semantic overlaps in textual vocabulary.
  - AdaBoost significantly underperformed ($75.00\%$ accuracy). Because AdaBoost 
    exponentially penalizes misclassified instances sequentially, it overfit heavily on 
    linguistic noise and structural vocabulary outposts present within the regulatory texts, 
    skewing its decision boundaries.


4. BEST MODEL SELECTION WITH JUSTIFICATION
--------------------------------------------------------------------------------
* Selection: CatBoostClassifier (Serialized as 'catboost_sentiment_model.pkl')

* Engineering Justification:
  1. Top Generalization Metrics: CatBoost secured the absolute highest macro F1-Score 
     (0.8478), showing balanced performance across positive, negative, and neutral targets.
  2. Native Optimization: CatBoost utilizes optimized symmetric tree splits that reduce 
     variance and prevent overfitting on sparse data layouts generated by text-based 
     vectorization parameters.
  3. Feature Robustness: It effectively balanced vocabulary weights without needing aggressive 
     hyperparameter tuning, making it the most resilient asset to transition from exploratory 
     notebook environments into a live, low-latency web deployment environment.

* Production Integration:
  The optimized model was serialized alongside its dedicated 'tfidf_vectorizer.pkl' object. 
  Both assets are successfully connected to a high-speed FastAPI backend ('app.py') serving 
  a live, auto-documented validation endpoint at '/predict' with clean JSON responses.
================================================================================
