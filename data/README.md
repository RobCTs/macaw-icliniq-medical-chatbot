# iCliniq Dataset

This folder should contain the iCliniq dataset used for training and evaluation. Please follow the instructions below:

1. Download the dataset from [iCliniq on Hugging Face](https://huggingface.co/datasets/ophycare/icliniq-dataset).
2. Place the downloaded dataset in this folder or refer to the Hugging Face `load_dataset` function in the code for direct loading.
3. A sample dataset `icliniq_dataset_sample.csv` is provided for quick testing.

## Data Format
Each entry in the dataset should contain:
- **Instruction**: Task instruction for the model.
- **Input**: User input question.
- **Response**: Model response to the question.
