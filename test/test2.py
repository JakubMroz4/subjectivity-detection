from transformers import XLMRobertaTokenizer, XLMRobertaForSequenceClassification
from transformers import Trainer, TrainingArguments
from datasets import load_dataset
import numpy as np
import torch
import pandas as pd
from datasets import Dataset

# Step 1: Install the required libraries if you haven't already
# !pip install transformers datasets torch

# Step 2: Load and preprocess the datasets
train_data = pd.read_csv("data/subtask-2-english/train_en.tsv",sep='\t')
eval_data = pd.read_csv("data/subtask-2-english/dev_en.tsv", sep='\t')

# Load XLM-RoBERTa tokenizer
tokenizer = XLMRobertaTokenizer.from_pretrained("xlm-roberta-large")

# Tokenize data
def tokenize_data(data):
    return tokenizer(data["sentence"].tolist(), padding=True, truncation=True), data["label"].tolist()

train_encodings, train_labels = tokenize_data(train_data)
eval_encodings, eval_labels = tokenize_data(eval_data)

# Define Dataset class
class CustomDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

# Convert data to Dataset object
#train_dataset = Dataset.from_dict({"input_ids": train_encodings["input_ids"], "attention_mask": train_encodings["attention_mask"], "labels": train_labels})
#eval_dataset = Dataset.from_dict({"input_ids": eval_encodings["input_ids"], "attention_mask": eval_encodings["attention_mask"], "labels": eval_labels})


train_dataset = CustomDataset(train_encodings, train_labels)
eval_dataset = CustomDataset(eval_encodings, eval_labels)

# Define custom Trainer class
class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fn = torch.nn.CrossEntropyLoss()
        loss = loss_fn(logits, labels)
        return (loss, outputs) if return_outputs else loss

# Step 3: Fine-tune the XLM-RoBERTa-Large model on the training data
model = XLMRobertaForSequenceClassification.from_pretrained("xlm-roberta-large", num_labels=2)

# Define Trainer
training_args = TrainingArguments(
    per_device_train_batch_size=8,
    num_train_epochs=3,
    evaluation_strategy="epoch",
    logging_dir="./logs",
    output_dir="./results",
)

# Define Trainer
trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

# Fine-tune the model
trainer.train()

# Step 4: Evaluate the model on the dev-test dataset
eval_results = trainer.evaluate()

print("Evaluation Results:", eval_results)