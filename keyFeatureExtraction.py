import os
import csv
from keybert import KeyBERT
from typing import List, Dict
from transformers import AutoModel, AutoTokenizer
import torch
from sklearn.base import BaseEstimator, TransformerMixin


model_name = "microsoft/codebert-base"
model = AutoModel.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
#creat embedding function
class CodeBERTEmbedder(BaseEstimator, TransformerMixin):
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def fit(self, X, y=None):
        return self  # Nothing to fit, so just return self

    def transform(self, texts):
        encoded_input = self.tokenizer(texts, padding=True, truncation=True, return_tensors='pt', max_length=512)
        with torch.no_grad():
            model_output = self.model(**encoded_input)
        input_mask_expanded = encoded_input['attention_mask'].unsqueeze(-1).expand(model_output.last_hidden_state.size()).float()
        embeddings = torch.sum(model_output.last_hidden_state * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return embeddings.cpu().numpy()  # Convert embeddings to NumPy array for compatibility with KeyBERT
codebert_embedder = CodeBERTEmbedder(model=model, tokenizer=tokenizer)
def load_file_content(file_path: str) -> str:
    """Load and return the content of a file."""
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def extract_keywords(text: str, num_keywords: int = 512) -> List[str]:
    """Extract keywords from the provided text."""
    kw_model = KeyBERT(model = codebert_embedder)
    keywords = kw_model.extract_keywords(text,keyphrase_ngram_range=(1, 1), stop_words='english', top_n=500)
    return [kw[0] for kw in keywords]

def extract_features_from_files(directory: str) -> Dict[str, List[str]]:
    """Extract features from all .cpp files in the specified directory and save to CSV."""
    features = {}
    for filename in os.listdir(directory):
        if filename.endswith(".cpp"):
            file_path = os.path.join(directory, filename)
            content = load_file_content(file_path)
            keywords = extract_keywords(content)
            base_filename = os.path.splitext(filename)[0]  # Remove the .cpp extension
            features[base_filename] = keywords
    return features

def save_features_to_csv(features: Dict[str, List[str]], csv_path: str):
    """Save the features dictionary to a CSV file."""
    with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Filename', 'Keywords'])
        for filename, keywords in features.items():
            writer.writerow([filename, ','.join(keywords)])

# Directory containing the C++ files
directory = '../DARE/hpc_applications/Benchmarks/'
# Path to save the CSV
csv_path = './keyBERTFeatures.csv'

features = extract_features_from_files(directory)
save_features_to_csv(features, csv_path)

print(f"Data has been successfully saved to {csv_path}")