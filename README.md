# NLP-Auto-secretary

## Dataset Overview

The dataset contains **77 intents** and **13,083 user utterances**, covering a wide range of banking-related customer service queries.

### Key Observations from EDA:
- The dataset is **moderately imbalanced**: the most frequent intent (`card_arrival`) has 285 samples, while the rarest intents (e.g., `exchange_via_app`, `fiat_currency_support`) have only 45–50 samples each.
- Common words include: *“card”*, *“payment”*, *“account”*, *“blocked”*, and *“transaction”*, reflecting typical banking support topics.
- Despite the imbalance, all intentions contain at least 45 examples, and this is enough to tune the model.

### Recommendation:
If you're using simpler machine learning models (like SVM TF-IDF or logistic regression), it’s a good idea to **balance the data**—for example, by adding more examples for rare intents or telling the model to pay extra attention to them. This helps the model understand less common questions better.

=======
## Pipeline

1. **Data ingestion:** `train.csv`, `test.csv` (plus `label_names.json`).
2. **Preprocessing:** Lowercase, punctuation removal, tokenization, lemmatization, stopwords filtering (with `nltk`).
3. **Vectorization:** TF-IDF with n-grams (uni- & bigrams).
4. **Train/validation split:** Stratified, `train_indices.csv`/`val_indices.csv`.
5. **Modeling:** Logistic Regression, Multinomial Naive Bayes, Random Forest.
6. **Evaluation:** Full metrics (accuracy, macro/weighted F1), classification reports, confusion matrices (val & test splits).
7. **Inference:** `predict_intent()` for any user text, plus chatbot logic (`chatbot_logic.py`) returning human-readable answers.

---

## Metrics (example)

| Model           | Val acc | Val F1  | Test acc | Test F1 |
|-----------------|---------|---------|----------|---------|
| LogisticRegression | ≈0.83  | ≈0.83  | (see eval) | (see eval) |
| Naive Bayes        | ...    | ...    | ...      | ...     |
| Random Forest      | ...    | ...    | ...      | ...     |

Full reports (per-class) and confusion matrices are in `data/banking77/eval_artifacts/`.

---

## Quickstart — How to Run

1. 
   - **(Optional) Setup environment(for Mac)**
    ```bash
    python3 -m venv venv
    source venv/bin/activate 
    pip install -r requirements.txt
    ```
    - **Setup environment(for Windows)**

    ```bash
    python3 -m venv venv
    venv\Scripts\activate 
    ```

2. **Download/prepare data**  
    (If using provided files, skip. Otherwise, run to fetch from HuggingFace:)
    ```bash
    python3 src/load_data.py
    ```

3. **Preprocess dataset (tokenize, clean, build TF-IDF)**
    ```bash
    python3 src/preprocessing.py
    ```

4. **Train LogisticRegression**
    ```bash
    python3 src/train_logreg.py
    ```

5. **Evaluate models and save metrics/reports**
    ```bash
    python3 src/evaluate.py
    ```

6. **Test predictions and chatbot answer**
    ```bash
    python3 -m src.predict "How do I order a new card?"
    python3 -c "from src.chatbot_logic import reply; print(reply('When will my card arrive?'))"
    ```

7. **(Optional) Edit `responses.json` to add/edit chatbot replies for each intent.**

---

## Project Files

- `src/`  
  - `preprocessing.py` — all text preprocessing and vectorization.
  - `train_logreg.py` — Logistic Regression model training.
  - `evaluate.py` — evaluation of 3 models and metrics summary.
  - `predict.py` — prediction function (intent classifier, top-k).
  - `chatbot_logic.py` — reply logic for chatbot (intent → answer mapping).

- `data/banking77/`
  - `train.csv`, `test.csv` — raw dataset splits.
  - `label_names.json` — intent names by label index.
  - `responses.json` — intent → answer mapping for chatbot.
  - Artifacts (ignored from git): TF-IDF, models, splits, reports.

---

## Results, Limitations, Improvements

- **Results:** Classic models with TF-IDF achieve up to 83% accuracy/F1 on a fine-grained real-world intent dataset.
- **Limitations:**  
  - Struggles with ambiguous/rare intents; quality is tied to the coverage of training data.
  - No neural embeddings/modern LLM integration yet.
- **Opportunities for improvement:**  
  - Add embeddings (Word2Vec/BERT/Transformer) for better context capture.
  - Fine-tune on user-specific queries and handle misspellings.
  - Build a full web/chatbot demo (Streamlit/Telegram).

---

### Authors & Credits

- **Daniil Istomin & Kirill Tarasov**  
- Data: [BANKING77 on HuggingFace](https://huggingface.co/datasets/legacy-datasets/banking77)  
- Pipeline/code: University of Debrecen, Department of Data Science and Visualization, 2025.
