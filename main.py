# main.py

import pandas as pd
from datasets import load_dataset
from src.data_utils import create_datasets
from src.model_utils import load_model_and_tokenizer, generate_response, collate_fn
from src.helpers import create_summary_writer, log_training_progress
from src.config_utils import load_yaml_config, load_json_config

# Load configurations
config = load_yaml_config('config/config.yaml')
training_args = load_json_config('config/training_args.json')

# Load the dataset
icliniq_dataset = load_dataset("path/to/your/icliniq_dataset")  # Modify with actual loading method
train_test_split = icliniq_dataset['train'].train_test_split(test_size=config['data']['train_test_split'], seed=42)
train_dataset = train_test_split['train']
validation_dataset = train_test_split['test']

train_df = train_dataset.to_pandas()
validation_df = validation_dataset.to_pandas()

# Create datasets for training and validation
train_dataset_tuples, validation_dataset_tuples = create_datasets(train_df, validation_df)

# Load model and tokenizer
model_name = config['model']['name']
model, tokenizer = load_model_and_tokenizer(model_name)

# Create a summary writer for logging
writer = create_summary_writer(training_args['logging_dir'])

# Create the Trainer instance
trainer = Trainer(
    model=model,
    args=training_args,  # Pass the training arguments directly from JSON
    train_dataset=train_dataset_tuples,
    eval_dataset=validation_dataset_tuples,
    data_collator=collate_fn(tokenizer)
)

# Training loop
for epoch in range(training_args['num_train_epochs']):
    trainer.train()

    # Log training loss
    train_loss = trainer.state.log_history[-1]['loss']  # Last logged loss
    log_training_progress(writer, epoch, train_loss, val_loss)

    # Evaluate and log validation loss
    eval_results = trainer.evaluate()
    val_loss = eval_results['eval_loss']
    log_training_progress(writer, epoch, train_loss, val_loss)

    print(f"Epoch {epoch}: Train Loss: {train_loss}, Validation Loss: {val_loss}")

# User interaction loop
while True:
    query = input("Enter your medical question (or 'exit' to quit): ")
    if query.lower() == 'exit':
        break
    response = generate_response(model, tokenizer, query)
    print(f"Macaw Response: {response}\n")
