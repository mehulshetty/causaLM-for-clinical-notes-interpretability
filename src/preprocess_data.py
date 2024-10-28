import pandas as pd
import json
import ast
from torch.utils.data import Dataset
from transformers import BertTokenizer
import torch
import random
import numpy as np
import re
from collections import defaultdict

# Load the clinical notes dataset
df = pd.read_csv('../data/release_train_patients.csv')  # Replace with your actual file path

# Load the evidence mapping dataset
with open('../data/release_evidences.json', 'r') as f:
    evidence_map = json.load(f)

def generate_note_for_patient(evidences_str, evidence_map):
    if not evidences_str or pd.isna(evidences_str):
        return ''
    
    try:
        evidences = ast.literal_eval(evidences_str)
        if not isinstance(evidences, list):
            return ''
        
        grouped_evidences = defaultdict(lambda: {'question': '', 'answers': set()})
        
        for evidence in evidences:
            base_code, v_code = evidence.rsplit('@', 1) if '@' in evidence else (evidence, None)
            if v_code:
                base_code = base_code[:-1]
                v_code = v_code[1:]
            
            mapping = evidence_map.get(base_code, {})
            question = mapping.get('question_en', '')
            data_type = mapping.get('data_type')

            if not question:
                continue

            grouped_evidences[base_code]['question'] = question

            if data_type == "B":
                answer = "yes"
            elif data_type == "C":
                value_meaning = mapping.get('value_meaning', {})
                answer = value_meaning.get(v_code, v_code)
                if answer != v_code:
                    answer = answer.get('en', '').lower()
            else:
                value_meaning = mapping.get('value_meaning', {})
                answer = value_meaning.get(v_code, {}).get('en', '').lower()

            if answer:
                grouped_evidences[base_code]['answers'].add(answer)
        
        return ' '.join(f"Q: {details['question']} A: {', '.join(sorted(details['answers'])).capitalize()}."
                        for details in grouped_evidences.values())
    
    except (ValueError, SyntaxError) as e:
        print(f"Error parsing evidences_str: {e}")
        return ''
    
def patient_basic_details(row, evidence_map=evidence_map):
    output = []
    age, sex, pathology, initial_evidence = row.get("AGE"), row.get("SEX"), row.get("PATHOLOGY"), row.get("INITIAL_EVIDENCE")

    if age:
        output.append("Q: What is your age? A: " + str(age) + ".")
    if sex:
        output.append("Q: What is your sex? A: " + ("Female." if sex == "F" else "Male."))
    if pathology:
        output.append("Q: What is your pathology? A: " + pathology + ".")
    if initial_evidence:
        output.append(f'Q: {evidence_map.get(initial_evidence).get("question_en")} A: Yes.')
    
    return " ".join(output)

df["basic_details"] = df.apply(lambda x: patient_basic_details(x), axis=1)
df["evidence_notes"] = df["EVIDENCES"].apply(lambda x: generate_note_for_patient(x, evidence_map))

df["notes"] = df["basic_details"] + " " + df["evidence_notes"]
df.drop(["AGE","DIFFERENTIAL_DIAGNOSIS","SEX","PATHOLOGY","EVIDENCES","INITIAL_EVIDENCE", "basic_details", "evidence_notes"], axis=1, inplace=True)
df.to_csv('../data/clinical_notes_preprocessed.csv', index=False)

with open('../data/release_conditions.json', 'r', encoding='utf-8') as f:
    disease_mapping = json.load(f)

notes = pd.read_csv('../data/clinical_notes_preprocessed.csv')

# Extract all disease names
all_diseases = sorted([details['condition_name'] for details in disease_mapping.values()])

# Create a mapping from disease to index
disease_to_index = {disease: idx for idx, disease in enumerate(all_diseases)}

# Handles Unknown Diseases
disease_to_index['Other'] = 49

#print(disease_to_index)
# {'Acute COPD exacerbation / infection': 0, 'Acute dystonic reactions': 1, 'Acute laryngitis': 2, 'Acute otitis media': 3, 'Acute pulmonary edema': 4, 'Acute rhinosinusitis': 5, 'Allergic sinusitis': 6, 'Anaphylaxis': 7, 'Anemia': 8, 'Atrial fibrillation': 9, 'Boerhaave': 10, 'Bronchiectasis': 11, 'Bronchiolitis': 12, 'Bronchitis': 13, 'Bronchospasm / acute asthma exacerbation': 14, 'Chagas': 15, 'Chronic rhinosinusitis': 16, 'Cluster headache': 17, 'Croup': 18, 'Ebola': 19, 'Epiglottitis': 20, 'GERD': 21, 'Guillain-Barré syndrome': 22, 'HIV (initial infection)': 23, 'Influenza': 24, 'Inguinal hernia': 25, 'Larygospasm': 26, 'Localized edema': 27, 'Myasthenia gravis': 28, 'Myocarditis': 29, 'PSVT': 30, 'Pancreatic neoplasm': 31, 'Panic attack': 32, 'Pericarditis': 33, 'Pneumonia': 34, 'Possible NSTEMI / STEMI': 35, 'Pulmonary embolism': 36, 'Pulmonary neoplasm': 37, 'SLE': 38, 'Sarcoidosis': 39, 'Scombroid food poisoning': 40, 'Spontaneous pneumothorax': 41, 'Spontaneous rib fracture': 42, 'Stable angina': 43, 'Tuberculosis': 44, 'URTI': 45, 'Unstable angina': 46, 'Viral pharyngitis': 47, 'Whooping cough': 48, 'Other': 49}

data = pd.DataFrame({
    'X': notes["notes"]
})

# Function to parse Y and create fixed-length vector
def parse_y(y_str, disease_to_index, num_diseases):
    # Safely evaluate the string to a Python list
    try:
        y_list = ast.literal_eval(y_str)
    except:
        y_list = []

    y_vector = np.zeros(num_diseases, dtype=np.float32)
    for disease, prob in y_list:
        if disease in disease_to_index:
            idx = disease_to_index[disease]
            y_vector[idx] = prob
        else:
            # Optionally handle unknown diseases
            y_vector[-1] += prob
    return y_vector

data['Y'] = df["DIFFERENTIAL_DIAGNOSIS"].apply(lambda y: parse_y(y, disease_to_index, len(all_diseases)))

# Define chest-related terms
chest_related_terms = [
    'chest', 'lower chest', 'posterior chest wall(l)', 'posterior chest wall(r)', 'side of the chest(l)', 'side of the chest(r)'
    # Add more terms as needed
]

# Compile regex patterns for efficiency
pain_location_pattern = re.compile(
    r'Q:\s*Do you feel pain somewhere\?\s*A:\s*([^\.]+)\.',
    re.IGNORECASE
)

def extract_chest_pain_label(x_text, chest_terms):
    """
    Extracts a binary label indicating the presence of chest pain.
    
    Parameters:
    - x_text (str): The clinical note in Q&A format.
    - chest_terms (list): List of chest-related terms to search for.
    
    Returns:
    - int: 1 if chest pain is present, 0 otherwise.
    """
    # Check if the patient reports having pain
    
    pain_location_match = pain_location_pattern.search(x_text)
    if pain_location_match:
        locations = pain_location_match.group(1).lower()
        for term in chest_terms:
            # Use word boundaries to avoid partial matches
            if re.search(r'\b' + re.escape(term) + r'\b', locations):
                return 1
    return 0

# Apply the function to create the chest_pain_label column
data['TC'] = data['X'].apply(lambda x: extract_chest_pain_label(x, chest_related_terms))

data.to_csv('../data/input.csv', index=False)

