import argparse
import ast
import random
import torch
import re
import tqdm
import numpy as np
import pandas as pd
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import BertTokenizerFast, BertModel, BertPreTrainedModel, BertConfig, BertForPreTraining, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from sparsemax import Sparsemax

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

### DATA PREP ###
# Function to split Q&A into a group of two sentences
def split_qa(x_text):
    # Split based on 'Q:' and 'A:'
    qa_pairs = re.findall(r'(Q:.*?A:.*?)(?=Q:|$)', x_text)
    # If the number of matches is odd, append "<END>"
    if len(qa_pairs) % 2 != 0:
        qa_pairs.append("<END>")

    # Group every two consecutive Q&A pairs into tuples
    grouped_pairs = [(qa_pairs[i], qa_pairs[i + 1]) for i in range(0, len(qa_pairs), 2)]

    return grouped_pairs

def generate_negative_nsp_pairs(sentence_a, sentence_b, num_negatives=None):

    if num_negatives is None:
        num_negatives = len(sentence_a)
    
    negative_sentence_a = []
    negative_sentence_b = []
    negative_labels_nsp = []
    
    for i in range(num_negatives):
        q = sentence_a[i]
        # Select a random question that won't follow the first question
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

class CustomBertForPreTrainingWithAdversary(BertForPreTraining):
    def __init__(self, config):
        super().__init__(config)
        
        # Adversarial Chest Pain Prediction Head
        self.grl = GradientReversal()
        self.adversary = nn.Linear(config.hidden_size, 1)  # Binary classification
        self.sigmoid = nn.Sigmoid()
        
        self.init_weights()
    
    def forward(self, input_ids, attention_mask=None, token_type_ids=None,
                labels_mlm=None, labels_nsp=None, chest_pain_labels=None, lambda_=6.0):
        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            labels=labels_mlm,
            next_sentence_label=labels_nsp
        )
        
        # Extract the hidden states
        hidden_states = outputs.hidden_states  # Tuple of (layer_count + 1) tensors
        # Get the last hidden state (final layer)
        last_hidden_state = hidden_states[-1]  # Shape: (batch_size, sequence_length, hidden_size)
        # Extract the [CLS] token's embedding
        cls_embedding = last_hidden_state[:, 0, :]  # Shape: (batch_size, hidden_size)
        
        
        # Adversarial Chest Pain Prediction
        reversed_output = self.grl(cls_embedding)
        chest_pain_logits = self.adversary(reversed_output)
        chest_pain_probs = self.sigmoid(chest_pain_logits)
        
        loss = None
        if labels_mlm is not None and labels_nsp is not None and chest_pain_labels is not None:
            # Extract losses
            masked_lm_loss = outputs.loss  # Assuming the main loss is MLM + NSP
            loss_fct_adversary = nn.BCELoss()
            chest_pain_loss = loss_fct_adversary(chest_pain_probs.view(-1), chest_pain_labels.view(-1))

            global curr_step 
            if curr_step % 1000 == 0:
                print(f'cpl: {chest_pain_loss.item()} and mlm: {masked_lm_loss.item()}')
            
            curr_step += 1
            
            # Combined Loss: MLM + NSP + Lambda * Adversarial Loss
            # The negative aspect will be added 
            loss = masked_lm_loss + lambda_ * chest_pain_loss
        
        return {
            'loss': loss,
            'prediction_scores': outputs.prediction_logits,
            'seq_relationship_scores': outputs.seq_relationship_logits,
            'chest_pain_probs': chest_pain_probs
        }

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
        item = {key: val[idx].clone().detach() for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.disease_labels[idx], dtype=torch.float32)
        return item
    

## TRAINING DEFINITION ##
class MultiTaskTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels_mlm = inputs.pop("labels_mlm")
        labels_nsp = inputs.pop("labels_nsp")
        chest_pain_labels = inputs.pop("chest_pain_labels")
        outputs = model(**inputs, labels_mlm=labels_mlm, labels_nsp=labels_nsp, chest_pain_labels=chest_pain_labels)
        loss = outputs['loss']
        return (loss, outputs) if return_outputs else loss

class DiseaseTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs, labels=labels)
        loss = outputs['loss']
        return (loss, outputs) if return_outputs else loss
    
def parse_y_entry(y_str):
    try:
        # Step 1: Remove square brackets and newlines
        y_str_clean = y_str.strip('[]').replace('\n', ' ')
        
        # Step 2: Split the string by whitespace to get individual numbers
        y_list = y_str_clean.split()
        
        # Step 3: Convert the list of strings to a list of floats
        y_floats = [float(num) for num in y_list]
        
        # Step 4: Convert the list of floats to a NumPy array
        y_array = np.array(y_floats, dtype=np.float32)
        
        return y_array
    except Exception as e:
        print(f"Error parsing Y entry: {y_str}")
        raise e
    

## MAIN ##
def main(args):
    global timeforit
    timeforit = 0
    # Seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    global curr_step
    curr_step = 0

    #Args
    lambda_ = args.lambda_
    device = args.device
    epochs = args.epochs
    train_csv = args.train_csv
    output_dir = args.output_dir

    data = pd.read_csv('../data/input.csv')
    data = data[:65000]
    
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

    print("1")
    
    # 2. Generate Negative NSP Pairs
    combined_sentence_a, combined_sentence_b, combined_labels_nsp = generate_negative_nsp_pairs(sentence_a, sentence_b)

    print("2")
    
    # 3. Tokenize Sentence Pairs
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    tokenized_inputs = tokenizer(
        combined_sentence_a,
        combined_sentence_b,
        truncation=True,
        padding='max_length',
        max_length=512,
        return_tensors='pt'
    )

    print("3")
    
    # 4. Mask Tokens for MLM
    masked_inputs, labels_mlm = mask_tokens(tokenized_inputs, tokenizer)

    print("4")
    
    # 5. Prepare Chest Pain Labels for NSP Pairs

    chest_pain_labels = []
    for i in range(len(combined_sentence_a)):
        if "chest" in combined_sentence_a[i] or "chest" in combined_sentence_b[i]:
            chest_pain_labels.append(1)
        else:
            chest_pain_labels.append(0)
    
    chest_pain_labels = torch.tensor(chest_pain_labels, dtype=torch.float32).unsqueeze(1)

    print("5")
    
    # 6. Create the Multi-Task Dataset
    dataset_pretraining = MultiTaskDataset(
        masked_inputs, 
        labels_mlm, 
        torch.tensor(combined_labels_nsp, dtype=torch.long), 
        chest_pain_labels)

    print("6")
    
    # 7. Initialize the Custom Model
    config = BertConfig.from_pretrained('bert-base-uncased', output_hidden_states=True)
    model_pretraining = CustomBertForPreTrainingWithAdversary.from_pretrained('bert-base-uncased', config=config)
    model_pretraining.to(device)

    print("7")

    # 8. Define Training Arguments for Pre-Training
    training_args_pretraining = TrainingArguments(
        output_dir='./results_pretraining',
        num_train_epochs=epochs,
        per_device_train_batch_size=3,
        gradient_accumulation_steps=2,
        save_strategy='epoch',
        logging_dir='./logs_pretraining',
        logging_steps=1000,
        learning_rate=5e-5,
        weight_decay=0.01,
        save_total_limit=2,
        disable_tqdm=False
    )

    print("8")
    
    # 9. Initialize the Multi-Task Trainer
    trainer_pretraining = MultiTaskTrainer(
        model=model_pretraining,
        args=training_args_pretraining,
        train_dataset=dataset_pretraining,
        # eval_dataset=dataset_pretraining,  # Ideally, use a separate validation set
    )

    print("9")
    
    # 10. Train the Multi-Task Model
    print("Starting pre-training with MLM, NSP, and adversarial tasks...")
    trainer_pretraining.train()
    trainer_pretraining.save_model('./models/pretrained_model')

    print("10")
    
    # ------------------------------
    # Learning for Disease Prediction
    # ------------------------------
    
    # 1. Prepare Data for Disease Prediction
    # Tokenize the original sentence pairs without masking
    tokenized_inputs_disease = tokenizer(
        data['X'].tolist(),
        truncation=True,
        padding='max_length',
        max_length=512,
        return_tensors='pt'
    )

    print("11")
    
    # 2. Create Disease Prediction Labels
    # Assuming each sentence pair corresponds to a single clinical note
    # Here, we aggregate disease labels per clinical note
    # Adjust as per your data's actual correspondence
    # For simplicity, we map each positive NSP pair to its corresponding disease labels
    # This might need to be adjusted based on your actual data structure
    disease_labels = data['Y'].apply(parse_y_entry)

    print("12")
    
    # 3. Create Disease Prediction Dataset
    dataset_disease = DiseasePredictionDataset(tokenized_inputs_disease, disease_labels)
    print("13")
    
    # 4. Initialize the Disease Prediction Model
    # Load the pre-trained model
    pretrained_model = BertModel.from_pretrained('./models/pretrained_model')
    model_disease = BertForDiseasePrediction(pretrained_model, num_diseases=49)
    model_disease.to(device)

    print("14")
    
    # 5. Define Training Arguments for Disease Prediction
    training_args_disease = TrainingArguments(
        output_dir='./results_disease',
        num_train_epochs=3,
        per_device_train_batch_size=3,
        save_strategy='epoch',
        logging_dir='./logs_disease',
        logging_steps=50,
        learning_rate=5e-4,  # Higher learning rate since BERT is frozen
        weight_decay=0.01,
        save_total_limit=2,
        disable_tqdm=False
    )

    print("15")
    
    # 6. Initialize the Disease Trainer
    trainer_disease = DiseaseTrainer(
        model=model_disease,
        args=training_args_disease,
        train_dataset=dataset_disease,
        # eval_dataset=dataset_disease,  # Ideally, use a separate validation set
    )

    print("16")
    
    # 7. Train the Disease Prediction Model
    print("Starting fine-tuning for disease prediction...")
    trainer_disease.train()
    trainer_disease.save_model('./models/disease_prediction_model')
    
    print("Training complete. Models saved in './models/' directory.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train BERT-CF Model")
    parser.add_argument('--train_csv', type=str, default="", help="Path to the training CSV file.")
    parser.add_argument('--output_dir', type=str, default="", help="Directory to save the fine-tuned model.")
    parser.add_argument('--epochs', type=int, default=3, help="Number of training epochs.")
    parser.add_argument('--lambda_', type=float, default=3.0, help="Gradient reversal scaling factor.")
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help="Device to train on.")
    
    args = parser.parse_args()
    
    main(args)