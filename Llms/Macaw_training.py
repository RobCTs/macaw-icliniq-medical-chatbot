# Import required libraries
import torch
from transformers import (
    Trainer,
    TrainingArguments,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    T5Tokenizer,
    T5ForConditionalGeneration,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
)
from datasets import load_dataset, Dataset
import pandas as pd
import re
import numpy as np

## Load the Macaw Model and Tokenizer

# Load the Macaw model and tokenizer
model_name = "allenai/macaw-large"  # choose 'macaw-3b' or '11b' for higher RAM availability
tokenizer = T5Tokenizer.from_pretrained(model_name, clean_up_tokenization_spaces=True)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Print model structure and tokenizer configuration
print("Macaw Model Structure:")
print(model)
print("\nTokenizer Configuration:")
print(tokenizer)

## Inspect the iCliniq Dataset
# Load the iCliniq dataset
icliniq_dataset = load_dataset("ophycare/icliniq-dataset", split="train")

# Convert to DataFrame for easier manipulation
df = pd.DataFrame(icliniq_dataset)

# Display dataset structure and sample entries
print("iCliniq Dataset Structure:")
print(icliniq_dataset.column_names)
print("\nFirst Few Entries:")
print(df.head())

# Print detailed sample entries
print("\nDetailed Sample of the Dataset:")
for i in range(5):
    print(f"Entry {i + 1}:\n{df['text'].iloc[i]}\n{'=' * 50}")


## Identify Patterns in the Dataset
# Define patterns for extraction
instruction_pattern = "Instruction:"
input_pattern = "Input:"
response_pattern = "Response:"

# Count occurrences of patterns in the text
df['instruction_count'] = df['text'].str.count(instruction_pattern)
df['input_count'] = df['text'].str.count(input_pattern)
df['response_count'] = df['text'].str.count(response_pattern)

# Display counts for the first few entries
print("Pattern Counts in First Few Entries:")
print(df[['instruction_count', 'input_count', 'response_count']].head())

## Split Dataset into Train and Validation Sets

# Perform an 80-20 split for training and validation
train_test_split = icliniq_dataset.train_test_split(test_size=0.2, seed=42)
train_dataset = train_test_split['train']
validation_dataset = train_test_split['test']

# Convert to DataFrames
train_df = train_dataset.to_pandas()
validation_df = validation_dataset.to_pandas()

# Display sizes of train and validation datasets
print(f"Training Set Size: {len(train_dataset)}")
print(f"Validation Set Size: {len(validation_dataset)}")

## Preprocess the Dataset for Training
# Function to preprocess the dataset into (Input, Response) tuples
def preprocess_dataset(df):
    data_tuples = []
    for entry in df['text']:
        input_match = re.search(r'(?<=### Input: )(.*?)(?=### Response:)', entry, re.DOTALL)
        response_match = re.search(r'(?<=### Response: )(.*)', entry, re.DOTALL)
        input_text = input_match.group(0).strip() if input_match else ''
        response_text = response_match.group(0).strip() if response_match else ''
        data_tuples.append((input_text, response_text))
    return data_tuples

# Preprocess training and validation sets
train_tuples = preprocess_dataset(train_df)
validation_tuples = preprocess_dataset(validation_df)

# Convert tuples to DataFrames and Hugging Face Datasets
train_dataset_tuples = Dataset.from_pandas(pd.DataFrame(train_tuples, columns=['input_text', 'response_text']))
validation_dataset_tuples = Dataset.from_pandas(pd.DataFrame(validation_tuples, columns=['input_text', 'response_text']))

# ROUGE metrics
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    
    predictions[predictions == -100] = tokenizer.pad_token_id
    labels[labels == -100] = tokenizer.pad_token_id
    
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    
    return {k: round(v, 4) for k, v in result.items()}

## PATH B: Define Training Arguments & Data Collato (lookin at the training)
# Set up training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir='./results',
    eval_strategy="steps",    # evaluate at the end of each step/epoch
    eval_steps=1000,
    save_strategy="steps",          # save model at the end of each ste/epoch
    save_steps=4000,
    save_total_limit=1,
    learning_rate=3e-5, #as macaw
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=5,
    load_best_model_at_end=True,
    metric_for_best_model="loss", # "rouge"
    greater_is_better=False,
    remove_unused_columns=False,
    report_to=["none"],
    predict_with_generate=True,
    generation_max_length=512,
    gradient_accumulation_steps=3,
    gradient_checkpointing=False,
    fp16=True,  # Use mixed precision
    fp16_full_eval=True,
    metric=
    max_grad_norm=1.0  # Gradient clipping
)

# Custom collate function for Seq2Seq models
def collate_fn(batch):
    input_texts = [item['input_text'] for item in batch]
    response_texts = [item['response_text'] for item in batch]
    inputs = tokenizer(input_texts, padding=True, truncation=True, return_tensors="pt")
    outputs = tokenizer(response_texts, padding=True, truncation=True, return_tensors="pt")
    labels = outputs['input_ids'].clone()
    labels[labels == tokenizer.pad_token_id] = -100
    return {"input_ids": inputs["input_ids"], "attention_mask": inputs["attention_mask"], "labels": labels}

## Train Model
# Create the trainer instance
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset_tuples,
    eval_dataset=validation_dataset_tuples,
    data_collator=collate_fn,
    compute_metrics=compute_metrics,
)

# Train the model
trainer.train()

# Save the trained model
trainer.save_model('icliniq_model_wSeq2Seq')


## PATH A
#Parameters and training without looking at M. trainings (but finished the epoches and 
# "just" having CUDA memory problem when saving the final model.

# Set up training arguments
#training_args = Seq2SeqTrainingArguments(
#    output_dir='./results',
#    eval_strategy="steps",    # evaluate at the end of each step/epoch
#    eval_steps=3000,
#    save_strategy="steps",          # save model at the end of each ste/epoch
#    save_steps=9000,
#    save_total_limit=,2
#    learning_rate=5e-5, 
#    per_device_train_batch_size=2,
#    per_device_eval_batch_size=2,
#    num_train_epochs=5,
#    load_best_model_at_end=True,
#    metric_for_best_model="loss", 
#    greater_is_better=False,
#    remove_unused_columns=False,
#    report_to=["none"],
#    predict_with_generate=True,
#)

## Generate Predictions
# Function to generate a response from the model
def generate_response(query):
    # Construct input with clearer context for the model
    input_text = f"### Input: {query} ### Response:"

    # Tokenize the input
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, padding=True)

    # Generate response with no gradient calculations
    with torch.no_grad():
        output = model.generate(**inputs, max_length=450)

    # Decode and return the response
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response

"""
# Loop for user input
while True:
    query = input("Enter your medical question (or 'exit' to quit): ")
    if query.lower() == 'exit':
        break
    response = generate_response(query)
    print(f"Macaw Response: {response}\n")
"""
