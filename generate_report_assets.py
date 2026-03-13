import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from predict import TopicPredictor
import os

# Set style
sns.set_theme(style="whitegrid")

def generate_assets():
    print("Generating updated plots for the report...")
    predictor = TopicPredictor(model_dir="./saved_model", topics_path="data/available_topics.csv")
    
    # Load test data and topics
    test_df = pd.read_csv('data/split/test.csv')
    df_topics = pd.read_csv('data/available_topics.csv')
    test_df = test_df.merge(df_topics, on='topic_id')
    
    # Run predictions
    y_true = test_df['topic_name'].tolist()
    y_pred = []
    confidences = []
    
    for msg in test_df['message']:
        result = predictor.predict(msg, threshold=0.0) # threshold 0 to see raw distribution
        y_pred.append(result['topic'])
        confidences.append(result['confidence'])
    
    # 1. Confusion Matrix
    labels = sorted(df_topics['topic_name'].unique().tolist())
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title('Confusion Matrix - DistilBERT (Latest Run)')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('imgs/confusion-matrix.png')
    plt.close()
    
    # 2. Confidence Distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(confidences, bins=20, kde=True, color='green')
    plt.axvline(0.60, color='red', linestyle='--', label='Confidence Threshold (0.60)')
    plt.title('Distribution of Prediction Confidence')
    plt.xlabel('Confidence Score')
    plt.ylabel('Frequency')
    plt.legend()
    plt.tight_layout()
    plt.savefig('imgs/model-confience.png')
    plt.close()
    
    print("Plots saved successfully in 'imgs/' folder.")

if __name__ == "__main__":
    os.makedirs('imgs', exist_ok=True)
    generate_assets()
