import pandas as pd
import numpy as np
import torch
import transformers
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    Trainer, 
    TrainingArguments, 
    EarlyStoppingCallback
)
from datasets import Dataset
import os

# Clean logs
transformers.logging.set_verbosity_error()

# Custom Trainer to handle class weights in the loss function
class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        
        # Access weights passed via TrainingArguments
        weights = torch.tensor(self.args.class_weights, dtype=torch.float).to(labels.device)
        loss_fct = torch.nn.CrossEntropyLoss(weight=weights)
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        
        return (loss, outputs) if return_outputs else loss

def get_datasets(split_dir="data/split"):
    # Run split if CSVs are missing
    if not os.path.exists(f"{split_dir}/train.csv"):
        from prepare_data import prepare_data
        prepare_data()

    d_train = pd.read_csv(f"{split_dir}/train.csv")
    d_val = pd.read_csv(f"{split_dir}/val.csv")
    d_test = pd.read_csv(f"{split_dir}/test.csv")
    
    return (
        d_train['message'], d_val['message'], d_test['message'],
        d_train['topic_id'], d_val['topic_id'], d_test['topic_id']
    )

def main():
    print("Training pipeline started (with Class Weights)...")
    
    tr_txt, val_txt, ts_txt, tr_lbl, val_lbl, ts_lbl = get_datasets()
    
    # Calculate class weights to
    unique_labels = np.sort(np.unique(tr_lbl))
    weights = compute_class_weight(
        class_weight='balanced',
        classes=unique_labels,
        y=tr_lbl
    )
    
    ckpt = "distilbert-base-uncased"
    tkz = AutoTokenizer.from_pretrained(ckpt)
    
    def tokenize(ex):
        return tkz(ex["text"], padding="max_length", truncation=True)
    
    ds_train = Dataset.from_dict({"text": tr_txt.tolist(), "label": tr_lbl.tolist()})
    ds_val = Dataset.from_dict({"text": val_txt.tolist(), "label": val_lbl.tolist()})
    ds_test = Dataset.from_dict({"text": ts_txt.tolist(), "label": ts_lbl.tolist()})
    
    tk_train = ds_train.map(tokenize, batched=True)
    tk_val = ds_val.map(tokenize, batched=True)
    tk_test = ds_test.map(tokenize, batched=True)
    
    model = AutoModelForSequenceClassification.from_pretrained(ckpt, num_labels=8)
    
    args = TrainingArguments(
        output_dir="./model_output",
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        num_train_epochs=10, 
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        save_total_limit=1,
        report_to="none"
    )
    
    # Store weights in the args so they are accessible inside WeightedTrainer
    args.class_weights = weights.tolist()
    
    trainer = WeightedTrainer(
        model=model,
        args=args,
        train_dataset=tk_train,
        eval_dataset=tk_val, 
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )
    
    trainer.train()
    
    # Final evaluation on isolated test set
    print("\n--- Final Test Set Evaluation ---")
    out = trainer.predict(tk_test)
    preds = np.argmax(out.predictions, axis=-1)
    print(classification_report(ts_lbl, preds))
    
    # Export artifacts
    trainer.save_model("./saved_model")
    tkz.save_pretrained("./saved_model")
    print("Done. Model artifacts exported to ./saved_model")

if __name__ == "__main__":
    main()
