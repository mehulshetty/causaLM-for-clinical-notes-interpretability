import pandas as pd
import numpy as np
from transformers import BertModel, BertTokenizerFast
import torch.nn as nn
import torch
import os
from torch.utils.data import Dataset, DataLoader
from sparsemax import Sparsemax

# Define the BERT-CF on Disease Prediction Model
class BertForDiseasePrediction(nn.Module):
    def __init__(self, bert_model, num_diseases):
        super(BertForDiseasePrediction, self).__init__()
        self.bert = bert_model  # Use the pre-trained BERT model
        for param in self.bert.parameters():
            param.requires_grad = False  # Freeze BERT parameters
        
        # Disease classification head
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_diseases)

        # Sparsemax Activation function
        self.sparsemax = Sparsemax(dim=1)

        # Initialize weights
        self.init_weights()

    def init_weights(self):
        """
        Initialize the weights of the classification layer.
        """
        nn.init.xavier_uniform_(self.classifier.weight)
        if self.classifier.bias is not None:
            nn.init.zeros_(self.classifier.bias)
    
    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None):
        
        outputs = self.bert(input_ids=input_ids, 
                            attention_mask=attention_mask, 
                            token_type_ids=token_type_ids,
                            return_dict=True)
        
        # Extract the [CLS] token's embedding
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        
        # Compute logits
        logits = self.classifier(cls_embedding)
        
        # Apply Sparsemax activation
        probs = self.sparsemax(logits)
        
        loss = None
        if labels is not None:
            loss_fct = nn.MSELoss()
            loss = loss_fct(probs, labels)
        
        return {'loss': loss, 'probs': probs}

    
class DiseasePredictionDataset(Dataset):
    def __init__(self, encodings, disease_labels):
        self.encodings = encodings
        self.disease_labels = disease_labels

    def __len__(self):
        return len(self.disease_labels)

    def __getitem__(self, idx):
        item = {key: val[idx].clone().detach() for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.disease_labels[idx], dtype=torch.float32)
        return item

    
# Load the pre-trained BERT model
pretrained_model = BertModel.from_pretrained('../models/pretrained_model')

# Initialize your disease prediction model
num_diseases = 49  # Adjust this if your number of diseases is different
model_disease = BertForDiseasePrediction(pretrained_model, num_diseases)

from safetensors.torch import load_file

# Load the state dictionary from the safetensors file
state_dict = load_file('../models/disease_prediction_model/model.safetensors')

# Load the state dictionary into your model
model_disease.load_state_dict(state_dict)

# Move the model to the appropriate device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_disease.to(device)

# Set the model to evaluation mode
model_disease.eval()

# Load your data
data = pd.read_csv('../../data/input.csv')

# Select the subset from index 1000 to 2000
data_test = data[100000:115000].reset_index(drop=True)

from transformers import BertTokenizerFast

# Initialize the tokenizer
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

# Tokenize the input texts
tokenized_inputs_test = tokenizer(
    data_test['X'].tolist(),
    truncation=True,
    padding='max_length',
    max_length=512,
    return_tensors='pt'
)

def parse_y_entry(y_str):
    try:
        # Remove square brackets and newlines
        y_str_clean = y_str.strip('[]').replace('\n', ' ')
        # Split the string by whitespace to get individual numbers
        y_list = y_str_clean.split()
        # Convert to floats
        y_floats = [float(num) for num in y_list]
        # Convert to NumPy array
        y_array = np.array(y_floats, dtype=np.float32)
        return y_array
    except Exception as e:
        print(f"Error parsing Y entry: {y_str}")
        raise e

# Apply the parsing function to get labels
disease_labels_test = data_test['Y'].apply(parse_y_entry).tolist()

# Create the dataset
dataset_test = DiseasePredictionDataset(tokenized_inputs_test, disease_labels_test)

# Create a DataLoader
test_loader = DataLoader(dataset_test, batch_size=8, shuffle=False)  # Adjust batch_size as needed

all_outputs = []
all_labels = []

with torch.no_grad():
    for batch in test_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        token_type_ids = batch['token_type_ids'].to(device)
        labels = batch['labels'].to(device)

        outputs = model_disease(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )

        probs = outputs['probs']  # Raw logits before sigmoid

        all_outputs.append(probs.cpu())
        all_labels.append(labels.cpu())

# Concatenate all outputs and labels
all_outputs = torch.cat(all_outputs, dim=0)
all_labels = torch.cat(all_labels, dim=0)


# Compute the differences between predicted probabilities and true labels
differences = all_outputs - all_labels

# Compute the absolute differences
abs_differences = differences.abs()

# Average the absolute differences for each column (disease)
mean_abs_differences = abs_differences.mean(dim=0)

# Convert to NumPy array for easier handling
mean_abs_differences = mean_abs_differences.numpy()

# Print the average absolute differences for each disease
for i, mean_diff in enumerate(mean_abs_differences):
    print(f'Disease {i}: Mean absolute difference: {mean_diff:.4f}')