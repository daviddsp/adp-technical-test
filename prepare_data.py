import pandas as pd
from sklearn.model_selection import train_test_split
import os

def prepare_data(source="data/available_conversations.csv", out_dir="data/split"):
    """
    Splits the main dataset into train/val/test chunks (70/15/15).
    Using a fixed seed for reproducibility across the pipeline.
    """
    if not os.path.exists(source):
        print(f"Error: Source file {source} not found.")
        return

    df = pd.read_csv(source)
    
    # Stratified split to keep topic distribution consistent
    df_train, temp = train_test_split(
        df, test_size=0.3, random_state=42, stratify=df['topic_id']
    )
    df_val, df_test = train_test_split(
        temp, test_size=0.5, random_state=42, stratify=temp['topic_id']
    )
    
    os.makedirs(out_dir, exist_ok=True)
    
    df_train.to_csv(f"{out_dir}/train.csv", index=False)
    df_val.to_csv(f"{out_dir}/val.csv", index=False)
    df_test.to_csv(f"{out_dir}/test.csv", index=False)
    
    print(f"Data ready in {out_dir}/ (Train: {len(df_train)}, Val: {len(df_val)}, Test: {len(df_test)})")

if __name__ == "__main__":
    prepare_data()
