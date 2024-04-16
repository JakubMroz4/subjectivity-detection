import torch
from datasets import load_dataset
from transformers import (
    RobertaTokenizerFast,
    RobertaForSequenceClassification,
    TrainingArguments,
    Trainer,
    AutoConfig,
)
from huggingface_hub import HfFolder, notebook_login

model_id = "roberta-base"
dataset_id = "ag_news"

#part 1

# Load dataset
dataset = load_dataset(dataset_id)

# Training and testing datasets
train_dataset = dataset['train']
test_dataset = dataset["test"].shard(num_shards=2, index=0)

print(train_dataset) # ['text', 'label'],

# Validation dataset
val_dataset = dataset['test'].shard(num_shards=2, index=1)

# Preprocessing
tokenizer = RobertaTokenizerFast.from_pretrained(model_id)

# This function tokenizes the input text using the RoBERTa tokenizer.
# It applies padding and truncation to ensure that all sequences have the same length (256 tokens).
def tokenize(batch):
    return tokenizer(batch["text"], padding=True, truncation=True, max_length=256)

train_dataset = train_dataset.map(tokenize, batched=True, batch_size=len(train_dataset))
val_dataset = val_dataset.map(tokenize, batched=True, batch_size=len(val_dataset))
test_dataset = test_dataset.map(tokenize, batched=True, batch_size=len(test_dataset))

print(train_dataset) #['text', 'label', 'input_ids', 'attention_mask'],

# Set dataset format
train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])
val_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])
test_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])

print(train_dataset)