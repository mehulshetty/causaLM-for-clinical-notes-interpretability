# causal_dataset.py

import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer
import pandas as pd

class ClinicalNotesCausalDataset(Dataset):
    def __init__(self, csv_file, tokenizer, max_length, controlled_concepts=None, mask_prob=0.15):
        """
        Args:
            csv_file (str): Path to the csv file with clinical notes.
            tokenizer (BertTokenizer): Tokenizer for BERT.
            max_length (int): Maximum sequence length.
            controlled_concepts (list): List of controlled concept labels (optional).
            mask_prob (float): Probability of masking tokens for MLM.
        """
        self.data = pd.read_csv(csv_file)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.controlled_concepts = controlled_concepts
        self.mask_prob = mask_prob
        
        # Precompute all controlled concept labels if provided
        if self.controlled_concepts:
            self.controlled_concept_map = {concept: idx for idx, concept in enumerate(self.controlled_concepts)}
        else:
            self.controlled_concept_map = {}
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        note = row['note']
        chest_pain = row['chest_pain']
        controlled_concept = row.get('controlled_concept', None)  # Assuming you have this column
        
        # Tokenize
        encoding = self.tokenizer.encode_plus(
            note,
            add_special_tokens=True,
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()
        
        # Prepare MLM labels
        labels = input_ids.clone()
        
        # Create mask
        probability_matrix = torch.full(labels.shape, self.mask_prob)
        
        # Corrected special_tokens_mask without extra dimension
        special_tokens_mask = self.tokenizer.get_special_tokens_mask(input_ids.tolist(), already_has_special_tokens=True)
        special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        
        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens
        
        # Replace masked input tokens with [MASK] token
        input_ids[masked_indices] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)
        
        # Treated Concept Label
        chest_pain_label = torch.tensor(chest_pain, dtype=torch.long)
        
        # Controlled Concept Label (if any)
        if self.controlled_concepts and controlled_concept:
            cc_label = torch.tensor(self.controlled_concept_map.get(controlled_concept, 0), dtype=torch.long)
        else:
            cc_label = torch.tensor(0, dtype=torch.long)  # Default to 0 if no controlled concept
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'chest_pain': chest_pain_label,
            'controlled_concept': cc_label
        }
