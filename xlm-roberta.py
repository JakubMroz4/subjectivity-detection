from transformers import XLMRobertaTokenizer, XLMRobertaForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score
import torch
import pandas as pd
# pip install sentencepiece

# Load data
train_data = pd.read_csv("data/subtask-2-english/train_en.tsv",sep='\t')  # Update with your train data file
dev_data = pd.read_csv("data/subtask-2-english/dev_en.tsv", sep='\t')  # Update with your dev data file

# Mapping label strings to integers
label_map = {"OBJ": 0, "SUBJ": 1}
train_data['label'] = train_data['label'].map(label_map)
train_data = train_data.drop('solved_conflict', axis=1)
dev_data['label'] = dev_data['label'].map(label_map)

print(train_data)

#train_data.set_format("torch", columns=["input_ids", "attention_mask", "label"])

# Load tokenizer
tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-large')

# Tokenize input sentences
train_encodings = tokenizer(train_data['sentence'].tolist(), truncation=True, padding=True)
dev_encodings = tokenizer(dev_data['sentence'].tolist(), truncation=True, padding=True)

print(train_encodings)

# Convert labels to tensors
train_labels = torch.tensor(train_data['label'].tolist())
dev_labels = torch.tensor(dev_data['label'].tolist())

#print(train_data['label'])
#print(train_data['sentence'])

# Define model
model = XLMRobertaForSequenceClassification.from_pretrained('xlm-roberta-large', num_labels=2)

# Define training arguments
training_args = TrainingArguments(
    output_dir='./results',
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    logging_dir='./logs',
    logging_steps=10,
    evaluation_strategy="epoch",
)

# Define Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_encodings,
    eval_dataset=dev_encodings,
    compute_metrics=lambda p: {"accuracy": accuracy_score(predictions.argmax(axis=1), dev_labels)},
)

# Train the model
trainer.train()

# Evaluate on dev set
results = trainer.evaluate()
print(results)

# Obtain predictions on dev set
with torch.no_grad():
    predictions = trainer.predict(dev_encodings)
    predicted_labels = torch.argmax(predictions.predictions["logits"], axis=1)

model.save_pretrained("model")  # Replace "model" with the directory where you want to save the model
