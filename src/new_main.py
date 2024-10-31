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

    data_iter = pd.read_csv('data/sampled_input.csv', chunksize=10000)
    scaler = torch.amp.GradScaler(device='cuda')

    # 1. Split Q&A into Sentence Pairs

    sentence_a, sentence_b, labels_nsp = extract_all_positive_qa_pairs(data_iter)
    print("1")


    print("2")

    # 3. Tokenize Sentence Pairs
    # tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    # tokenized_inputs = tokenizer(
    #     combined_sentence_a,
    #     combined_sentence_b,
    #     truncation=True,
    #     padding='max_length',
    #     max_length=512,
    #     return_tensors='pt'
    # )

    # print("3")

    # # 4. Mask Tokens for MLM
    # masked_inputs, labels_mlm = mask_tokens(tokenized_inputs, tokenizer)

    tokenizer = BertTokenizerFast.from_pretrained(
        'bert-base-uncased',
        clean_up_tokenization_spaces=True
    )


    print("4")

    # 5. Prepare Chest Pain Labels for NSP Pairs

    chest_pain_labels = [
        1.0 if ('chest' in a.lower() or 'chest' in b.lower()) else 0.0
        for a, b in zip(sentence_a, sentence_b)
    ]
    chest_pain_labels = torch.tensor(chest_pain_labels, dtype=torch.float32).unsqueeze(1)
    print("5")

    # 6. Create the Multi-Task Dataset
    dataset_pretraining = EfficientMultiTaskDataset(
        sentence_a=sentence_a,
        sentence_b=sentence_b,
        labels_nsp=labels_nsp,
        chest_pain_labels=chest_pain_labels,
        tokenizer=tokenizer,
        max_length=512
    )

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
        per_device_train_batch_size=8,  # Adjust based on GPU memory
        gradient_accumulation_steps=4,
        save_strategy='epoch',
        logging_dir='./logs_pretraining',
        logging_steps=50,
        learning_rate=5e-5,
        weight_decay=0.01,
        save_total_limit=2,
        disable_tqdm=False,
        fp16=True,  # Enable mixed-precision training
        report_to="none"
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
    trainer_pretraining.save_model('/models/pretrained_model')

    print("10")

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
    disease_labels = data_iter['Y'].values.repeat(len(sentence_a)).astype(np.float32)
    disease_labels = torch.tensor(disease_labels)

    print("12")

    # 3. Create Disease Prediction Dataset
    dataset_disease = DiseasePredictionDataset(tokenized_inputs_disease, disease_labels)
    print("13")

    # 4. Initialize the Disease Prediction Model
    # Load the pre-trained model
    pretrained_model = BertModel.from_pretrained('models/pretrained_model')
    model_disease = BertForDiseasePrediction(pretrained_model, num_diseases=50)
    model_disease.to(device)

    print("14")

    # 5. Define Training Arguments for Disease Prediction
    training_args_disease = TrainingArguments(
        output_dir='./results_disease',
        num_train_epochs=3,
        per_device_train_batch_size=16,  # Adjust based on GPU memory
        save_strategy='epoch',
        logging_dir='./logs_disease',
        logging_steps=50,
        learning_rate=1e-3,  # Higher learning rate since BERT is frozen
        weight_decay=0.01,
        save_total_limit=2,
        disable_tqdm=False,
        fp16=True  # Enable mixed-precision training
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
    trainer_disease.save_model('models/disease_prediction_model')

    print("Training complete. Models saved in 'models/' directory.")

class Args:
    train_csv = "/path/to/your/input.csv"  # Update with your actual path
    output_dir = "models/pretrained_model"
    epochs = 5
    lambda_ = 1.0
    device = 'cuda' if torch.cuda.is_available() else 'cpu'


if __name__ == "__main__":
    args = Args()

    main(args)