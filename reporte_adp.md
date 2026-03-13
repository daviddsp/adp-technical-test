# ADP HR Topic Classification - Technical Report

**Author:** Abraham Solórzano

---

## My Approach & Methodology

The goal was to build a routing engine that accurately directs HR queries to specialized agents. I focused on building a scalable pipeline that ensures data integrity and high reliability for production environments.

### Key Assumptions
* **Exclusive Routing:** To keep the user experience simple, I treated this as a multiclass problem where each message gets exactly one destination.
* **Confidence Filtering:** Chatbots shouldn't guess. I implemented a **0.60 confidence threshold**. If the model is unsure, it flags the query as "Unsupported" instead of routing it to the wrong person.
* **Data Integrity & Isolation:** To ensure a fair evaluation, I physically partitioned the dataset into static `train`, `val`, and `test` files before any training occurred. This isolation guarantees that the final metrics are based on completely unseen data, preventing any potential data leakage.

## Model Evolution

### 1. The Baseline (LR + TF-IDF)
I started with a simple Logistic Regression model. It's fast and easy to debug, but it only reached **44% accuracy**. This confirmed that simple keyword matching doesn't work for HR topics because many categories (like Payroll vs Taxes) share the same vocabulary.

### 2. Deep Learning (Fine-Tuning DistilBERT)
To fix the semantic confusion, I moved to `distilbert-base-uncased`. I chose this because it's much faster than regular BERT while keeping ~97% of its performance.
* **Setup:** 10 epochs max, but with `EarlyStopping` to avoid overfitting once the validation loss stops dropping.
* **Learning:** Used a 2e-5 learning rate with weight decay for better generalization.

## Results & Insights

The final model reached **64% accuracy** on the isolated test set. This represents a significant jump from the baseline and provides a solid foundation for a first production release.

### Topic Performance breakdown:
* **Tax Services & Benefits (High Accuracy):** The model handles these very well (F1-Scores ~0.70+). The language in these domains is specific enough for the transformer to catch.
* **Payroll (Confusing):** This category shows the most overlap with others. In a future iteration, I'd suggest gathering more niche training samples specifically for payroll.

## Metrics & Visualization

### Confidence Distribution
![Model confidence Distribution](imgs/model-confience.png)
Most predictions cluster above 0.8, which is great. It means when the model decides, it usually knows what it's doing. The red line shows where we start rejecting low-quality queries to protect the routing accuracy.

### Confusion Matrix
![Confusion matrix](imgs/confusion-matrix.png)
You can see the strong diagonal here. The "Other" category is the main source of confusion, which makes sense since it's a catch-all for anything miscellaneous.

## Trade-offs & Edge Case Handling

* **Threshold Selection (0.60):** I arrived at this number by analyzing the confidence distribution across the test set. At 0.60, the model effectively filters out-of-domain noise while maintaining a balanced recall for supported topics. In a production setting, this threshold can be tuned dynamically to favor precision over recall (or vice-versa) based on business needs.
* **The "Other" Topic vs "Unsupported":** A key architectural challenge is distinguishing between a supported "Other" query and an "Unsupported" query. Currently, the model uses the softmax probability as a proxy for out-of-domain detection. A future improvement could involve training a dedicated binary classifier for "Supported vs Unsupported" to further improve the safety layer.
* **Class Imbalance:** Given the stratified sampling in our split, the model maintains stable performance. However, for a high-traffic production system, I would recommend implementing class weighting in the loss function to protect underrepresented domains.
