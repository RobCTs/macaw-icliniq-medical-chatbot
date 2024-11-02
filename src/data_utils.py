# src/data_utils.py

import re
import pandas as pd
from datasets import Dataset

def preprocess_dataset(df):
    """Preprocess the dataset and extract Input and Response tuples."""
    data_tuples = []

    for entry in df['text']:
        input_match = re.search(r'(?<=### Input: )(.*?)(?=### Response:)', entry, re.DOTALL)
        response_match = re.search(r'(?<=### Response: )(.*)', entry, re.DOTALL)

        input_text = input_match.group(0).strip() if input_match else ''
        response_text = response_match.group(0).strip() if response_match else ''

        data_tuples.append((input_text, response_text))

    return data_tuples

def create_datasets(train_df, validation_df):
    """Convert DataFrames to Hugging Face datasets."""
    train_tuples = preprocess_dataset(train_df)
    validation_tuples = preprocess_dataset(validation_df)

    train_df_tuples = pd.DataFrame(train_tuples, columns=['input_text', 'response_text'])
    validation_df_tuples = pd.DataFrame(validation_tuples, columns=['input_text', 'response_text'])

    train_dataset_tuples = Dataset.from_pandas(train_df_tuples)
    validation_dataset_tuples = Dataset.from_pandas(validation_df_tuples)

    return train_dataset_tuples, validation_dataset_tuples