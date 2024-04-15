from transformers import XLMRobertaTokenizer, XLMRobertaForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score
import torch
import pandas as pd
import csv

def to_labels(data, threshold=0.5):
    ypred = []
    for pred in data:
        if pred >= threshold:
            ypred.append('SUBJ')
        else:
            ypred.append('OBJ')
    return ypred

def load_data(): # path to dev, dev-test and train
