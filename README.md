# NLP-Auto-secretary

## Dataset Overview

The dataset contains **77 intents** and **13,083 user utterances**, covering a wide range of banking-related customer service queries.

### Key Observations from EDA:
- The dataset is **moderately imbalanced**: the most frequent intent (`card_arrival`) has 285 samples, while the rarest intents (e.g., `exchange_via_app`, `fiat_currency_support`) have only 45–50 samples each.
- Common words include: *“card”*, *“payment”*, *“account”*, *“blocked”*, and *“transaction”*, reflecting typical banking support topics.
- Despite the imbalance, all intentions contain at least 45 examples, and this is enough to tune the model.

### Recommendation:
If you're using simpler machine learning models (like SVM TF-IDF or logistic regression), it’s a good idea to **balance the data**—for example, by adding more examples for rare intents or telling the model to pay extra attention to them. This helps the model understand less common questions better.

