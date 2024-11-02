# src/model_utils.py

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Trainer

def load_model_and_tokenizer(model_name):
    """Load the model and tokenizer from Hugging Face."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return model, tokenizer

def generate_response(model, tokenizer, query):
    """Generate a response using the loaded model."""
    input_text = f"### Input: {query} ### Response:"
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, padding=True)

    with torch.no_grad():
        output = model.generate(**inputs, max_length=450)

    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response

def collate_fn(tokenizer):
    """Custom collate function for preparing inputs for the model."""
    def inner(batch):
        input_texts = [item['input_text'] for item in batch]
        response_texts = [item['response_text'] for item in batch]

        inputs = tokenizer(input_texts, padding=True, truncation=True, return_tensors="pt")
        outputs = tokenizer(response_texts, padding=True, truncation=True, return_tensors="pt")

        labels = outputs["input_ids"].clone()
        labels[labels == tokenizer.pad_token_id] = -100

        return {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "labels": labels,
        }
    return inner
