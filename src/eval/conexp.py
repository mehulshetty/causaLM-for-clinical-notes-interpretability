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
    def __init__(self, encodings, disease_labels, concept_indicators):
        self.encodings = encodings
        self.disease_labels = disease_labels
        self.concept_indicators = concept_indicators

    def __len__(self):
        return len(self.disease_labels)

    def __getitem__(self, idx):
        item = {key: val[idx].clone().detach() for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.disease_labels[idx], dtype=torch.float32)
        # Add concept_indicator to the batch item
        item['concept_indicator'] = torch.tensor(self.concept_indicators[idx], dtype=torch.long)
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

# Create the concept_indicator column
# If 'chest' appears in the text, concept_indicator = 1, else 0
concept_indicator_array = np.where(data_test['X'].str.lower().str.contains('chest'), 1, 0)

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
        # Clean and parse labels
        y_str_clean = y_str.strip('[]').replace('\n', ' ')
        y_list = y_str_clean.split()
        y_floats = [float(num) for num in y_list]
        return np.array(y_floats, dtype=np.float32)
    except Exception as e:
        print(f"Error parsing Y entry: {y_str}")
        raise e
    
disease_labels_test = data_test['Y'].apply(parse_y_entry).tolist()

# Create the dataset with concept_indicator
dataset_test = DiseasePredictionDataset(tokenized_inputs_test, disease_labels_test, concept_indicator_array)

# Create DataLoader
test_loader = DataLoader(dataset_test, batch_size=8, shuffle=False)

all_outputs = []
all_labels = []
all_concept_presence = []

with torch.no_grad():
    for batch in test_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        token_type_ids = batch['token_type_ids'].to(device)
        labels = batch['labels'].to(device)
        concept_indicator = batch['concept_indicator']  # remains on CPU

        outputs = model_disease(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )

        probs = outputs['probs']

        all_outputs.append(probs.cpu())
        all_labels.append(labels.cpu())
        all_concept_presence.append(concept_indicator.cpu())

all_outputs = torch.cat(all_outputs, dim=0)
all_concept_presence = torch.cat(all_concept_presence, dim=0)

concept_present_indices = (all_concept_presence == 1)
concept_absent_indices = (all_concept_presence == 0)

probs_concept_present = all_outputs[concept_present_indices]
probs_concept_absent = all_outputs[concept_absent_indices]

mean_present = probs_concept_present.mean(dim=0)
mean_absent = probs_concept_absent.mean(dim=0)

conexp = mean_present - mean_absent
print("CONEXP:", conexp.numpy())