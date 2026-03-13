import pandas as pd
from sklearn.model_selection import train_test_split
import os

def prepare_data(data_path="data/available_conversations.csv", output_dir="data/split"):
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    
    # 1. Separate Train (70%) and Temp (30%)
    df_train, df_temp = train_test_split(
        df, test_size=0.3, random_state=42, stratify=df['topic_id']
    )
    
    # 2. Separate Temp into Validation (15%) and Test (15%)
    df_val, df_test = train_test_split(
        df_temp, test_size=0.5, random_state=42, stratify=df_temp['topic_id']
    )
    
    # Create directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save files
    df_train.to_csv(os.path.join(output_dir, "train.csv"), index=False)
    df_val.to_csv(os.path.join(output_dir, "val.csv"), index=False)
    df_test.to_csv(os.path.join(output_dir, "test.csv"), index=False)
    
    print(f"Data split successfully!")
    print(f" - Train: {len(df_train)} samples -> {output_dir}/train.csv")
    print(f" - Val:   {len(df_val)} samples -> {output_dir}/val.csv")
    print(f" - Test:  {len(df_test)} samples -> {output_dir}/test.csv")

if __name__ == "__main__":
    prepare_data()
