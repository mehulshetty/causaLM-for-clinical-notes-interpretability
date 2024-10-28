import argparse
import os
import random
import torch
import re
import numpy as np
import pandas as pd
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import BertTokenizerFast, BertModel, BertPreTrainedModel, BertConfig, BertOnlyMLMHead, BertOnlyNSPHead, Trainer, TrainingArguments, DataCollatorForLanguageModeling

### DATA PREP ###
# Function to split Q&A into sentences
def split_qa(x_text):
    # Split based on 'Q:' and 'A:'
    qa_pairs = re.findall(r'Q:\s*(.*?)\s*A:\s*(.*?)(?=Q:|$)', x_text, re.DOTALL)
    return qa_pairs

def generate_negative_nsp_pairs(sentence_a, sentence_b, num_negatives=None):
    """
    Generates negative NSP pairs by pairing questions with random answers.
    
    Parameters:
    - sentence_a (list): List of sentences A (questions).
    - sentence_b (list): List of sentences B (answers).
    - num_negatives (int): Number of negative pairs to generate. If None, generates same number as positive.
    
    Returns:
    - tuple: (combined_sentence_a, combined_sentence_b, combined_labels_nsp)
    """
    if num_negatives is None:
        num_negatives = len(sentence_a)
    
    negative_sentence_a = []
    negative_sentence_b = []
    negative_labels_nsp = []
    
    for i in range(num_negatives):
        q = sentence_a[i]
        # Select a random answer that is not the correct one
        neg_a = random.choice(sentence_b)
        while neg_a == sentence_b[i]:
            neg_a = random.choice(sentence_b)
        negative_sentence_a.append(q)
        negative_sentence_b.append(neg_a)
        negative_labels_nsp.append(0)
    
    combined_sentence_a = sentence_a + negative_sentence_a
    combined_sentence_b = sentence_b + negative_sentence_b
    combined_labels_nsp = [1]*len(sentence_a) + negative_labels_nsp
    
    return combined_sentence_a, combined_sentence_b, combined_labels_nsp

# For MLM labels, we'll use the DataCollator
# Here, we prepare them manually

def mask_tokens(inputs, tokenizer, mlm_probability=0.15):
    """
    Prepare masked tokens inputs/labels for masked language modeling.
    """
    labels = inputs.input_ids.clone()
    
    # We sample a few tokens in each sequence for MLM training
    probability_matrix = torch.full(labels.shape, mlm_probability)
    special_tokens_mask = [
        tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
    ]
    special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
    probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = -100  # We only compute loss on masked tokens

    # 80% of the time, replace masked input tokens with [MASK]
    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
    inputs.input_ids[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

    # 10% of the time, replace masked input tokens with random word
    indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
    random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long)
    inputs.input_ids[indices_random] = random_words[indices_random]

    # The rest 10% of the time, keep the masked input tokens unchanged
    return inputs, labels


### MODEL ###

class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg()

class GradientReversal(nn.Module):
    def __init__(self):
        super(GradientReversal, self).__init__()
    
    def forward(self, x):
        return GradientReversalFunction.apply(x)

class CustomBertForPreTrainingWithAdversary(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config)
        
        # MLM Head
        self.cls = BertOnlyMLMHead(config)
        
        # NSP Head
        self.seq_relationship = BertOnlyNSPHead(config)
        
        # Adversarial Chest Pain Prediction Head
        self.grl = GradientReversal()
        self.adversary = nn.Linear(config.hidden_size, 1)  # Binary classification
        self.sigmoid = nn.Sigmoid()
        
        self.init_weights()
    
    def forward(self, input_ids, attention_mask=None, token_type_ids=None,
                labels_mlm=None, labels_nsp=None, chest_pain_labels=None, lambda_=1):
        outputs = self.bert(input_ids=input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            return_dict=True)
        
        sequence_output = outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]
        pooled_output = outputs.pooler_output        # [batch_size, hidden_size]
        
        # MLM Task
        prediction_scores = self.cls(sequence_output)  # [batch_size, seq_len, vocab_size]
        
        # NSP Task
        seq_relationship_scores = self.seq_relationship(pooled_output)  # [batch_size, 2]
        
        # Adversarial Chest Pain Prediction
        reversed_output = self.grl(pooled_output)             # [batch_size, hidden_size]
        chest_pain_logits = self.adversary(reversed_output)    # [batch_size, 1]
        chest_pain_probs = self.sigmoid(chest_pain_logits)     # [batch_size, 1]
        
        loss = None
        if labels_mlm is not None and labels_nsp is not None and chest_pain_labels is not None:
            # MLM Loss
            loss_fct_mlm = nn.CrossEntropyLoss()
            masked_lm_loss = loss_fct_mlm(prediction_scores.view(-1, self.config.vocab_size), 
                                         labels_mlm.view(-1))
            
            # NSP Loss
            loss_fct_nsp = nn.CrossEntropyLoss()
            nsp_loss = loss_fct_nsp(seq_relationship_scores.view(-1, 2), 
                                    labels_nsp.view(-1))
            
            # Adversarial Chest Pain Loss
            loss_fct_adversary = nn.BCELoss()
            chest_pain_loss = loss_fct_adversary(chest_pain_probs.view(-1), 
                                               chest_pain_labels.view(-1))
            
            # Combined Loss: MLM + NSP - Lambda * Adversarial Loss
            # The GRL ensures that minimizing this loss will maximize the adversarial loss
            loss = masked_lm_loss + nsp_loss - lambda_ * chest_pain_loss
        
        return {
            'loss': loss,
            'prediction_scores': prediction_scores,
            'seq_relationship_scores': seq_relationship_scores,
            'chest_pain_probs': chest_pain_probs
        }

# Define the BERT-CF on Disease Prediction Model
class BertForDiseasePrediction(nn.Module):
    def __init__(self, bert_model, num_diseases):
        super(BertForDiseasePrediction, self).__init__()
        self.bert = bert_model.bert  # Use the pre-trained BERT model
        for param in self.bert.parameters():
            param.requires_grad = False  # Freeze BERT parameters
        
        # Disease classification head
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_diseases)
    
    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None):
        with torch.no_grad():
            outputs = self.bert(input_ids=input_ids, 
                                attention_mask=attention_mask, 
                                token_type_ids=token_type_ids,
                                return_dict=True)
            pooled_output = outputs.pooler_output  # [batch_size, hidden_size]
        
        disease_logits = self.classifier(pooled_output)  # [batch_size, num_diseases]
        
        loss = None
        if labels is not None:
            loss_fct = nn.BCEWithLogitsLoss()
            loss = loss_fct(disease_logits, labels)
        
        return {'loss': loss, 'logits': disease_logits}


## DATASET DEFINITION ##
class MultiTaskDataset(Dataset):
    def __init__(self, encodings, labels_mlm, labels_nsp, chest_pain_labels):
        self.encodings = encodings
        self.labels_mlm = labels_mlm
        self.labels_nsp = labels_nsp
        self.chest_pain_labels = chest_pain_labels

    def __len__(self):
        return len(self.labels_nsp)

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels_mlm'] = self.labels_mlm[idx]
        item['labels_nsp'] = self.labels_nsp[idx]
        item['chest_pain_labels'] = self.chest_pain_labels[idx]
        return item
    
class DiseasePredictionDataset(Dataset):
    def __init__(self, encodings, disease_labels):
        self.encodings = encodings
        self.disease_labels = disease_labels

    def __len__(self):
        return len(self.disease_labels)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.disease_labels[idx], dtype=torch.float32)
        return item
    

## TRAINING DEFINITION ##
class MultiTaskTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels_mlm = inputs.pop("labels_mlm")
        labels_nsp = inputs.pop("labels_nsp")
        chest_pain_labels = inputs.pop("chest_pain_labels")
        outputs = model(**inputs, labels_mlm=labels_mlm, labels_nsp=labels_nsp, chest_pain_labels=chest_pain_labels)
        loss = outputs['loss']
        return (loss, outputs) if return_outputs else loss

class DiseaseTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs, labels=labels)
        loss = outputs['loss']
        return (loss, outputs) if return_outputs else loss
    

## MAIN ##
def main(args):
    # Seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    #Args
    lambda_ = args.lambda_
    device = args.device
    epochs = args.epochs
    train_csv = args.train_csv
    output_dir = args.output_dir

    data = pd.read_csv('input.csv')
    
    # 1. Split Q&A into Sentence Pairs
    sentence_a = []
    sentence_b = []
    labels_nsp = []
    
    for _, row in data.iterrows():
        qa_pairs = split_qa(row['X'])
        for qa in qa_pairs:
            q, a = qa
            sentence_a.append(q.strip())
            sentence_b.append(a.strip())
            labels_nsp.append(1)  # Positive pair
    
    # 2. Generate Negative NSP Pairs
    combined_sentence_a, combined_sentence_b, combined_labels_nsp = generate_negative_nsp_pairs(sentence_a, sentence_b)
    
    # 3. Tokenize Sentence Pairs
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    tokenized_inputs = tokenizer(
        combined_sentence_a,
        combined_sentence_b,
        truncation=True,
        padding='max_length',
        max_length=128,
        return_tensors='pt'
    )
    
    # 4. Mask Tokens for MLM
    masked_inputs, labels_mlm = mask_tokens(tokenized_inputs, tokenizer)
    
    # 5. Prepare Chest Pain Labels for NSP Pairs
    # Assuming chest pain labels correspond to the original positive pairs
    # For negative pairs, we can assign chest pain labels similar to positive pairs or set to 0
    # Here, we'll duplicate the chest pain labels for negative pairs
    chest_pain_labels_positive = data['TC'].values.repeat(len(sentence_a))[:len(combined_sentence_a)]
    # For simplicity, assign the same chest pain label to negative pairs as their corresponding positive pairs
    chest_pain_labels_neg = data['TC'].values[:len(combined_sentence_a) - len(sentence_a)]
    # Alternatively, you can set chest_pain_labels_neg = 0 if chest pain is unrelated
    # Here, we keep them the same
    combined_chest_pain_labels = np.concatenate([
        data['TC'].values[:len(sentence_a)],
        data['TC'].values[:len(combined_sentence_a) - len(sentence_a)]
    ])
    combined_chest_pain_labels = torch.tensor(combined_chest_pain_labels, dtype=torch.float32).unsqueeze(1)
    
    # 6. Create the Multi-Task Dataset
    dataset_pretraining = MultiTaskDataset(masked_inputs, labels_mlm, torch.tensor(combined_labels_nsp, dtype=torch.long), combined_chest_pain_labels)
    
    # 7. Initialize the Custom Model
    config = BertConfig.from_pretrained('bert-base-uncased')
    model_pretraining = CustomBertForPreTrainingWithAdversary.from_pretrained('bert-base-uncased', config=config, lambda_=lambda_)
    model_pretraining.to(device)
    
    # 8. Define Training Arguments for Pre-Training
    training_args_pretraining = TrainingArguments(
        output_dir='./results_pretraining',
        num_train_epochs=3,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=2,
        evaluation_strategy='epoch',
        save_strategy='epoch',
        logging_dir='./logs_pretraining',
        logging_steps=50,
        learning_rate=5e-5,
        weight_decay=0.01,
        save_total_limit=2
    )
    
    # 9. Initialize the Multi-Task Trainer
    trainer_pretraining = MultiTaskTrainer(
        model=model_pretraining,
        args=training_args_pretraining,
        train_dataset=dataset_pretraining,
        # eval_dataset=dataset_pretraining,  # Ideally, use a separate validation set
    )
    
    # 10. Train the Multi-Task Model
    print("Starting pre-training with MLM, NSP, and adversarial tasks...")
    trainer_pretraining.train()
    trainer_pretraining.save_model('./models/pretrained_model')
    
    # ------------------------------
    # Learning for Disease Prediction
    # ------------------------------
    
    # 1. Prepare Data for Disease Prediction
    # Tokenize the original sentence pairs without masking
    tokenized_inputs_disease = tokenizer(
        sentence_a,
        sentence_b,
        truncation=True,
        padding='max_length',
        max_length=128,
        return_tensors='pt'
    )
    
    # 2. Create Disease Prediction Labels
    # Assuming each sentence pair corresponds to a single clinical note
    # Here, we aggregate disease labels per clinical note
    # Adjust as per your data's actual correspondence
    # For simplicity, we map each positive NSP pair to its corresponding disease labels
    # This might need to be adjusted based on your actual data structure
    disease_labels = data['Y'].values.repeat(len(sentence_a)).astype(np.float32)
    disease_labels = torch.tensor(disease_labels)
    
    # 3. Create Disease Prediction Dataset
    dataset_disease = DiseasePredictionDataset(tokenized_inputs_disease, disease_labels)
    
    # 4. Initialize the Disease Prediction Model
    # Load the pre-trained model
    pretrained_model = BertModel.from_pretrained('./models/pretrained_model')
    model_disease = BertForDiseasePrediction(pretrained_model, num_diseases=len(all_diseases))
    model_disease.to(device)
    
    # 5. Define Training Arguments for Disease Prediction
    training_args_disease = TrainingArguments(
        output_dir='./results_disease',
        num_train_epochs=3,
        per_device_train_batch_size=16,
        evaluation_strategy='epoch',
        save_strategy='epoch',
        logging_dir='./logs_disease',
        logging_steps=50,
        learning_rate=1e-3,  # Higher learning rate since BERT is frozen
        weight_decay=0.01,
        save_total_limit=2
    )
    
    # 6. Initialize the Disease Trainer
    trainer_disease = DiseaseTrainer(
        model=model_disease,
        args=training_args_disease,
        train_dataset=dataset_disease,
        # eval_dataset=dataset_disease,  # Ideally, use a separate validation set
    )
    
    # 7. Train the Disease Prediction Model
    print("Starting fine-tuning for disease prediction...")
    trainer_disease.train()
    trainer_disease.save_model('./models/disease_prediction_model')
    
    print("Training complete. Models saved in './models/' directory.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train BERT-CF Model")
    parser.add_argument('--train_csv', type=str, required=True, help="Path to the training CSV file.")
    parser.add_argument('--output_dir', type=str, required=True, help="Directory to save the fine-tuned model.")
    parser.add_argument('--epochs', type=int, default=5, help="Number of training epochs.")
    parser.add_argument('--lambda_', type=float, default=1.0, help="Gradient reversal scaling factor.")
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help="Device to train on.")
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    main(args)