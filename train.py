import pandas as pd
import numpy as np
import torch
import transformers
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, EarlyStoppingCallback
from datasets import Dataset
import os

# Suppress visual warnings
transformers.logging.set_verbosity_error()

def load_pre_split_data(split_dir="data/split"):
    # Check if files exist, otherwise run preparation
    if not os.path.exists(os.path.join(split_dir, "train.csv")):
        from prepare_data import prepare_data
        prepare_data()

    df_train = pd.read_csv(os.path.join(split_dir, "train.csv"))
    df_val = pd.read_csv(os.path.join(split_dir, "val.csv"))
    df_test = pd.read_csv(os.path.join(split_dir, "test.csv"))
    
    return (
        df_train['message'], df_val['message'], df_test['message'],
        df_train['topic_id'], df_val['topic_id'], df_test['topic_id']
    )

def main():
    print("Starting training pipeline...")
    
    # 1. Load data
    train_texts, val_texts, test_texts, train_labels, val_labels, test_labels = load_pre_split_data()
    
    # 2. Tokenization
    model_ckpt = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
    
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True)
    
    train_ds = Dataset.from_dict({"text": train_texts.tolist(), "label": train_labels.tolist()})
    val_ds = Dataset.from_dict({"text": val_texts.tolist(), "label": val_labels.tolist()})
    test_ds = Dataset.from_dict({"text": test_texts.tolist(), "label": test_labels.tolist()})
    
    tokenized_train = train_ds.map(tokenize_function, batched=True)
    tokenized_val = val_ds.map(tokenize_function, batched=True)
    tokenized_test = test_ds.map(tokenize_function, batched=True)
    
    # 3. Model Configuration
    model = AutoModelForSequenceClassification.from_pretrained(model_ckpt, num_labels=8)
    
    training_args = TrainingArguments(
        output_dir="./model_output",
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        num_train_epochs=10, 
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        save_total_limit=2,  # Keep only the 2 best/last checkpoints to save space
        report_to="none",
        dataloader_pin_memory=False
    )
    
    # 4. Training using VALIDATION for evaluation, not TEST
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val, 
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )
    
    print("Training model...")
    trainer.train()
    
    # 5. Final evaluation on TEST SET (totally blind to the model until now)
    print("\nEvaluating on Test Set (Isolated)...")
    predictions = trainer.predict(tokenized_test)
    preds = np.argmax(predictions.predictions, axis=-1)
    
    print("\nFinal Report (Test Set):")
    print(classification_report(test_labels, preds))
    
    # 6. Save final model and tokenizer
    # Using trainer.save_model ensures the weights from 'load_best_model_at_end' are the ones saved
    trainer.save_model("./saved_model")
    tokenizer.save_pretrained("./saved_model")
    print("Model saved successfully in './saved_model'")

if __name__ == "__main__":
    main()