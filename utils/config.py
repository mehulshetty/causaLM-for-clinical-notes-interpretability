import torch
import os

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TRAIN_SAMPLE_SIZE = 100000

def create_save_files_directories(timestamp, job_id):
    # Create models directory if needed
    os.makedirs("models", exist_ok=True)
    models_dir = f"models/{job_id}_{timestamp}"
    os.makedirs(models_dir, exist_ok=True)

    os.makedirs("results_pretraining", exist_ok=True)
    results_pretraining_dir = f"results_pretraining/{job_id}_{timestamp}"
    os.makedirs(results_pretraining_dir, exist_ok=True)

    os.makedirs("results_disease", exist_ok=True)
    results_disease_dir = f"results_disease/{job_id}_{timestamp}"
    os.makedirs(results_disease_dir, exist_ok=True)

    os.makedirs("logs_pretraining", exist_ok=True)
    logs_pretraining_dir = f"logs_pretraining/{job_id}_{timestamp}"
    os.makedirs(logs_pretraining_dir, exist_ok=True)

    os.makedirs("logs_disease", exist_ok=True)
    logs_disease_dir = f"logs_disease/{job_id}_{timestamp}"
    os.makedirs(logs_disease_dir, exist_ok=True)

    return models_dir, results_pretraining_dir, results_disease_dir, logs_pretraining_dir, logs_disease_dir