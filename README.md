# Using CausaLM to Improve Interpretability for Clinical Notes

## Authors

**Name:** Tina Chandwani, Connor Jordan, Garima Merani,
Alexander Romanus, Mehul Shetty, and Justin Yatco

## Overview
causaLM for Clinical Notes Interpretability is a specialized machine learning framework designed to pre-train and fine-tune BERT-based models on clinical notes. The project focuses on enhancing interpretability in clinical settings by leveraging multi-task learning, including Masked Language Modeling (MLM), Next Sentence Prediction (NSP), and an adversarial task for predicting specific clinical attributes such as chest pain.

This project aims to:

- Fine-tune BERT models on clinical Q&A datasets.
- Incorporate adversarial training to improve model robustness and interpretability.
- Facilitate disease prediction based on clinical notes.
- Optimize resource usage for efficient training on high-performance computing (HPC) clusters like USC's CARC.


## Features

- Multi-Task Learning: Combines MLM, NSP, and adversarial tasks for comprehensive model training.
- Adversarial Chest Pain Prediction: Enhances model robustness by predicting chest pain, encouraging the model to learn more generalized features.
- Custom Dataset Handling: Efficiently processes large clinical datasets with optimized memory usage.
- Scalable Training Scripts: Designed to run on local machines and HPC clusters.
- Flexible Configuration: Easily adjustable hyperparameters and training settings via config.json.

## Dataset Description - ddxplus

The ddxplus training set is a curated dataset specifically designed for training and evaluating language models to interpret natural medical language. This dataset comprises clinical question-and-answer (Q&A) pairs extracted from patient records, along with associated diagnostic labels and metadata. The primary objective of the ddxplus training set is to facilitate the development of models that can understand and interpret clinical narratives, particularly focusing on symptom descriptions and their relationships to specific diagnoses.

### Creating Counterfactual Representations
To enhance the model's ability to understand the influence of specific symptoms on diagnoses, we employed a technique known as counterfactual representation generation. This involves creating modified versions of the original data by systematically masking or altering certain components, thereby allowing the model to learn the causal relationships between symptoms and diseases.

Process Overview:

- Identification of Target Concepts:

    - We focused on symptoms and concepts related to chest pain, as these are critical indicators for a range of cardiovascular conditions.

    - Relevant Q&A pairs that explicitly mention chest pain or related descriptors (e.g., "heavy pain", "sensitive area", "radiating pain") were identified within the dataset.

- Masking Technique:

    - Masking Q&A Pairs: For each identified Q&A pair that describes chest pain or related concepts, we applied a masking strategy to obscure the symptom descriptions. This involves replacing specific words or phrases with a [MASK] token or removing them entirely from the text.

    - Purpose of Masking: By masking these Q&A pairs, we create counterfactual scenarios where the presence of chest pain is either altered or omitted. This helps the model discern the impact of chest pain descriptions on disease predictions by comparing outcomes with and without these critical symptoms.

- Generation of Counterfactual Data:

    - The original dataset is thus expanded to include both factual and counterfactual instances. Factual instances retain all original Q&A pairs, while counterfactual instances have specific Q&A pairs related to chest pain masked or modified.

    - This dual representation ensures that the model is trained to recognize and account for the presence or absence of key symptoms, enhancing its interpretability and robustness in clinical settings.

- Integration with Training Pipeline:

    - The counterfactual representations are seamlessly integrated into the training process. During model training, both factual and counterfactual data are used to optimize the model's ability to predict diseases accurately while understanding the causal relationships between symptoms and diagnoses.

    - This approach not only improves the model's predictive performance but also provides insights into which symptoms are most influential in determining specific diseases.

### Benefits of Counterfactual Representations
- Enhanced Interpretability: By training the model on both factual and counterfactual data, it becomes adept at identifying which symptoms are pivotal for certain diagnoses, thereby offering clearer interpretability of its predictions.

- Robustness to Missing Information: The model learns to make accurate predictions even when specific symptoms are absent or obscured, making it more reliable in real-world scenarios where data may be incomplete or noisy.

- Causal Understanding: This methodology encourages the model to develop a deeper understanding of the causal relationships between symptoms and diseases, moving beyond mere correlations to grasp underlying medical logic.

## Architecture

The project comprises the following key components:

Data Processing:

- CSV Format: Clinical notes are stored in a CSV file with columns X (Q&A pairs), Y (disease labels as arrays), and TC (additional clinical metadata).
- Data Cleaning: Preprocessing steps to handle multiline entries and ensure compatibility with data loaders.

Modeling:

- Custom BERT Model: Extends BertForPreTraining with an adversarial chest pain prediction head.
- Disease Prediction Head: Implements a separate classification layer for disease prediction using Sparsemax activation.

Training:

- Pre-Training: Utilizes MLM and NSP tasks combined with adversarial training.
- Fine-Tuning: Focuses on disease prediction using the pre-trained model.
- Trainer Classes: Custom Trainer classes for multi-task and disease prediction training.

Job Submission:

- SLURM Scripts: Bash scripts to submit and manage training jobs on CARC.
- Resource Optimization: Configurable parameters for memory, GPU usage, and job duration.
