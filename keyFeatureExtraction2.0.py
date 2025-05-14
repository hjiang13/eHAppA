import os
import csv
from keybert import KeyBERT
from typing import List, Dict
from transformers import AutoModel, AutoTokenizer
import torch
from torch import nn
from sklearn.base import BaseEstimator, TransformerMixin


model_name = "microsoft/codebert-base"
model = AutoModel.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
#creat embedding function
class CodeBERTEmbedder(BaseEstimator, TransformerMixin):
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.lstm = nn.LSTM(input_size=self.model.config.hidden_size,
                            hidden_size=512,  # You can modify the size of the LSTM hidden layer
                            num_layers=1,    # Number of LSTM layers
                            batch_first=True,
                            bidirectional=True)  # Using a bidirectional LSTM

    def fit(self, X, y=None):
        return self  # Nothing to fit, so just return self

    def transform(self, texts):
        self.model.eval()
        self.lstm.eval()
        embeddings = []
        #encoded_input = self.tokenizer(texts, padding=True, truncation=True, return_tensors='pt', max_length=512)
        with torch.no_grad():
            for doc in texts:
                encoded_input = self.tokenizer(doc, return_tensors='pt', padding=True, truncation=False, max_length=512)
                outputs = self.model(**encoded_input)
                sequence_output = outputs.last_hidden_state
                # LSTM processing
                lstm_output, (hidden, cell) = self.lstm(sequence_output)
                # use the last hidden state or apply another pooling method over the LSTM output
                lstm_embedding = lstm_output[:, -1, :]  # Using the last hidden state
                embeddings.append(lstm_embedding.squeeze(0).cpu().numpy())
        return nn.array(embeddings)
codebert_embedder = CodeBERTEmbedder(model=model, tokenizer=tokenizer)
def load_file_content(file_path: str) -> str:
    """Load and return the content of a file."""
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def extract_keywords(text: str, num_keywords: int = 512) -> List[str]:
    """Extract keywords from the provided text."""
    kw_model = KeyBERT(model = codebert_embedder)
    keywords = kw_model.extract_keywords(text,keyphrase_ngram_range=(1, 1), stop_words='english', top_n=512)
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
csv_path = './keyFeature.csv'

features = extract_features_from_files(directory)
save_features_to_csv(features, csv_path)

print(f"Data has been successfully saved to {csv_path}")