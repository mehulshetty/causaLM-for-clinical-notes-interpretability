import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertConfig
from transformers import AdamW, get_linear_schedule_with_warmup
import argparse
from tqdm import tqdm
from causal_dataset import ClinicalNotesCausalDataset
from model import BertCFModel

# Assuming BertCFModel and ClinicalNotesCausalDataset are defined as above

def train_bert_cf(args):
    # Initialize tokenizer and model
    tokenizer = BertTokenizer.from_pretrained(args.bert_model)
    config = BertConfig.from_pretrained(args.bert_model)
    model = BertCFModel(config, lambda_=args.lambda_)
    model.to(args.device)
    
    # Prepare dataset and dataloader
    dataset = ClinicalNotesCausalDataset(
        csv_file=args.train_csv,
        tokenizer=tokenizer,
        max_length=args.max_length,
        controlled_concepts=args.controlled_concepts,
        mask_prob=args.mask_prob
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    # Prepare optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, eps=1e-8)
    total_steps = len(dataloader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=total_steps
    )
    
    # Training loop
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0
        for batch in tqdm(dataloader, desc=f"Training Epoch {epoch+1}/{args.epochs}"):
            optimizer.zero_grad()
            
            input_ids = batch['input_ids'].to(args.device)
            attention_mask = batch['attention_mask'].to(args.device)
            labels = batch['labels'].to(args.device)
            chest_pain = batch['chest_pain'].to(args.device)
            controlled_concept = batch['controlled_concept'].to(args.device) if args.controlled_concepts else None
            
            loss, mlm_loss, cc_loss, tc_loss = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                chest_pain=chest_pain,
                controlled_concept=controlled_concept
            )
            
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            epoch_loss += loss.item()
        
        avg_epoch_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch+1} - Average Loss: {avg_epoch_loss:.4f}")
        
        # Save model checkpoint
        if args.save_steps and (epoch + 1) % args.save_steps == 0:
            save_path = f"{args.output_dir}/bert_cf_epoch_{epoch+1}.pt"
            torch.save(model.state_dict(), save_path)
            print(f"Saved checkpoint to {save_path}")
    
    # Save final model
    torch.save(model.state_dict(), f"{args.output_dir}/bert_cf_final.pt")
    print(f"Saved final model to {args.output_dir}/bert_cf_final.pt")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-Tune BERT-CF Model")
    parser.add_argument('--train_csv', type=str, default='../data/clinical_notes_with_chest_pain.csv', help="Path to the training CSV file.")
    parser.add_argument('--output_dir', type=str, default='../out/finetuned', help="Directory to save the fine-tuned model.")
    parser.add_argument('--bert_model', type=str, default='bert-base-uncased', help="Pre-trained BERT model.")
    parser.add_argument('--max_length', type=int, default=512, help="Maximum sequence length.")
    parser.add_argument('--batch_size', type=int, default=8, help="Batch size.")
    parser.add_argument('--epochs', type=int, default=5, help="Number of training epochs.")
    parser.add_argument('--learning_rate', type=float, default=3e-5, help="Learning rate.")
    parser.add_argument('--warmup_steps', type=int, default=0, help="Number of warmup steps.")
    parser.add_argument('--lambda_', type=float, default=1.0, help="Gradient reversal scaling factor.")
    parser.add_argument('--controlled_concepts', nargs='*', default=None, help="List of controlled concepts.")
    parser.add_argument('--mask_prob', type=float, default=0.15, help="Probability of masking tokens.")
    parser.add_argument('--save_steps', type=int, default=None, help="Save checkpoint every X epochs.")
    parser.add_argument('--device', type=str, default='cpu')
    
    args = parser.parse_args()
    
    import os
    os.makedirs(args.output_dir, exist_ok=True)
    
    train_bert_cf(args)
