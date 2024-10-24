import pandas as pd
import json
import ast
from torch.utils.data import Dataset
from transformers import BertTokenizer
import torch
import random

# Load the clinical notes dataset
df = pd.read_csv('../data/release_train_patients.csv')  # Replace with your actual file path

# Load the evidence mapping dataset
with open('../data/release_evidences.json', 'r') as f:
    evidence_map = json.load(f)

def generate_note_for_patient(evidences_str, evidence_map):
    if not evidences_str or pd.isna(evidences_str):
        return ''
    try:
        # Convert the string representation of the list into an actual list
        evidences = ast.literal_eval(evidences_str)
        if not isinstance(evidences, list):
            return ''
        note_parts = []
        for evidence in evidences:
            # Evidence codes might be in the format 'E_55_@_V_89'
            if '_@_' in evidence:
                e_code, v_code = evidence.split('_@_')
            else:
                e_code = evidence
                v_code = None
            mapping = evidence_map.get(e_code, {})
            question = mapping.get('question_en', '').strip()
            if not question:
                continue  # Skip if question is not available
            if v_code and 'value_meaning' in mapping and v_code in mapping['value_meaning']:
                answer = mapping['value_meaning'][v_code]['en'].strip()
            else:
                # Use default value or indicate missing answer
                default_value = mapping.get('default_value', '')
                if default_value and 'value_meaning' in mapping and default_value in mapping['value_meaning']:
                    answer = mapping['value_meaning'][default_value]['en'].strip()
                else:
                    answer = ''  # Missing answer
            # Construct the question-answer pair
            if answer:
                note_parts.append(f"Q: {question} A: {answer}")

        # Combine all question-answer pairs to form the note
        note = '. '.join(note_parts)
        return note
    except (ValueError, SyntaxError):
        return ''

def extract_chest_pain(evidences_str, chest_pain_evidences):
    if not evidences_str or pd.isna(evidences_str):
        return 0
    try:
        # Convert the string representation of the list into an actual list
        evidences = ast.literal_eval(evidences_str)
        if not isinstance(evidences, list):
            return 0
        # Check if any evidence code indicates chest pain
        for evidence in evidences:
            if evidence in chest_pain_evidences:
                return 1
        return 0
    except (ValueError, SyntaxError):
        return 0


df['note'] = df['EVIDENCES'].apply(lambda x: generate_note_for_patient(x, evidence_map))

df['unique_id'] = df.index

def get_chest_pain_evidences(evidence_map):
    chest_pain_evidences = set()
    # Keywords to identify chest pain
    chest_keywords = ['chest', 'sternum', 'thorax', 'breast', 'pectoral', 'rib', 'precordial']
    pain_keywords = ['pain', 'douleur']
    
    for e_code, mapping in evidence_map.items():
        question_en = mapping.get('question_en', '').lower()
        question_fr = mapping.get('question_fr', '').lower()
        
        # Check if the question is about pain
        if any(pain_kw in question_en for pain_kw in pain_keywords) or \
           any(pain_kw in question_fr for pain_kw in pain_keywords):
            # Check if there are value meanings
            value_meaning = mapping.get('value_meaning', {})
            for v_code, meaning in value_meaning.items():
                meaning_en = meaning.get('en', '').lower()
                # Check if the meaning indicates chest area
                if any(chest_kw in meaning_en for chest_kw in chest_keywords):
                    chest_pain_evidences.add(f"{e_code}_@_{v_code}")
    return chest_pain_evidences

# Get the set of evidence codes that indicate chest pain
chest_pain_evidences = get_chest_pain_evidences(evidence_map)


def extract_chest_pain(evidences_str, chest_pain_evidences):
    if not evidences_str or pd.isna(evidences_str):
        return 0
    try:
        evidences = ast.literal_eval(evidences_str)
        if not isinstance(evidences, list):
            return 0
        for evidence in evidences:
            if evidence in chest_pain_evidences:
                return 1
        return 0
    except (ValueError, SyntaxError):
        return 0

# Apply the function to create the 'chest_pain' column
df['chest_pain'] = df['EVIDENCES'].apply(lambda x: extract_chest_pain(x, chest_pain_evidences))


