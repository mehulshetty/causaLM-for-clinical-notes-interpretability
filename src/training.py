import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertConfig
import argparse
from tqdm import tqdm
from model import BertCFModel

# Define a simple classifier on top of BERT-CF
class BertCFCausalClassifier(nn.Module):
    def __init__(self, bert_cf_model, num_labels):
        super(BertCFCausalClassifier, self).__init__()
        self.bert_cf = bert_cf_model
        self.classifier = nn.Linear(bert_cf_model.config.hidden_size, num_labels)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, input_ids, attention_mask):
        with torch.no_grad():
            outputs = self.bert_cf.bert(input_ids=input_ids, attention_mask=attention_mask)
            pooled_output = outputs.pooler_output  # (batch_size, hidden_size)
        logits = self.classifier(pooled_output)
        probs = self.sigmoid(logits)
        return probs

def train_classifier(args):
    # Initialize tokenizer
    tokenizer = BertTokenizer.from_pretrained(args.bert_model)
    
    # Load the fine-tuned BERT-CF model
    config = BertConfig.from_pretrained(args.bert_model)
    bert_cf_model = BertCFModel(config)
    bert_cf_model.load_state_dict(torch.load(args.bert_cf_path))
    bert_cf_model.to(args.device)
    bert_cf_model.eval()  # Freeze BERT-CF parameters
    
    # Initialize classifier
    classifier = BertCFCausalClassifier(bert_cf_model, num_labels=args.num_labels)
    classifier.to(args.device)
    
    # Prepare dataset and dataloader
    class DiseaseDataset(Dataset):
        def __init__(self, csv_file, tokenizer, max_length):
            self.data = pd.read_csv(csv_file)
            self.tokenizer = tokenizer
            self.max_length = max_length
        
        def __len__(self):
            return len(self.data)
        
        def __getitem__(self, idx):
            row = self.data.iloc[idx]
            note = row['note']
            disease_probs = ast.literal_eval(row['disease_probs'])  # List of [disease, prob]
            
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
            
            # Prepare labels
            labels = torch.tensor([prob for _, prob in disease_probs], dtype=torch.float)
            
            return {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'labels': labels
            }
    
    train_dataset = DiseaseDataset(args.train_csv, tokenizer, args.max_length)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    
    val_dataset = DiseaseDataset(args.val_csv, tokenizer, args.max_length)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Define optimizer and loss function
    optimizer = torch.optim.AdamW(classifier.parameters(), lr=args.learning_rate, eps=1e-8)
    loss_fn = nn.BCELoss()
    
    # Training loop
    for epoch in range(args.epochs):
        classifier.train()
        epoch_loss = 0
        for batch in tqdm(train_dataloader, desc=f"Training Epoch {epoch+1}/{args.epochs}"):
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(args.device)
            attention_mask = batch['attention_mask'].to(args.device)
            labels = batch['labels'].to(args.device)
            
            outputs = classifier(input_ids=input_ids, attention_mask=attention_mask)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_epoch_loss = epoch_loss / len(train_dataloader)
        print(f"Epoch {epoch+1} - Average Training Loss: {avg_epoch_loss:.4f}")
        
        # Validation
        classifier.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_dataloader:
                input_ids = batch['input_ids'].to(args.device)
                attention_mask = batch['attention_mask'].to(args.device)
                labels = batch['labels'].to(args.device)
                
                outputs = classifier(input_ids=input_ids, attention_mask=attention_mask)
                loss = loss_fn(outputs, labels)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_dataloader)
        print(f"Epoch {epoch+1} - Average Validation Loss: {avg_val_loss:.4f}")
        
        # Save checkpoint
        if (epoch + 1) % args.save_steps == 0:
            save_path = f"{args.output_dir}/classifier_epoch_{epoch+1}.pt"
            torch.save(classifier.state_dict(), save_path)
            print(f"Saved classifier checkpoint to {save_path}")
    
    # Save final classifier
    torch.save(classifier.state_dict(), f"{args.output_dir}/classifier_final.pt")
    print(f"Saved final classifier to {args.output_dir}/classifier_final.pt")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Causal Classifier with BERT-CF")
    parser.add_argument('--train_csv', type=str, required=True, help="Path to the training CSV file.")
    parser.add_argument('--val_csv', type=str, required=True, help="Path to the validation CSV file.")
    parser.add_argument('--bert_cf_path', type=str, required=True, help="Path to the fine-tuned BERT-CF model.")
    parser.add_argument('--output_dir', type=str, required=True, help="Directory to save the trained classifier.")
    parser.add_argument('--bert_model', type=str, default='bert-base-uncased', help="Pre-trained BERT model.")
    parser.add_argument('--max_length', type=int, default=512, help="Maximum sequence length.")
    parser.add_argument('--batch_size', type=int, default=16, help="Batch size.")
    parser.add_argument('--epochs', type=int, default=5, help="Number of training epochs.")
    parser.add_argument('--learning_rate', type=float, default=3e-5, help="Learning rate.")
    parser.add_argument('--save_steps', type=int, default=1, help="Save checkpoint every X epochs.")
    parser.add_argument('--num_labels', type=int, required=True, help="Number of disease labels.")
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help="Device to train on.")
    
    args = parser.parse_args()
    
    import os
    os.makedirs(args.output_dir, exist_ok=True)
    
    train_classifier(args)
