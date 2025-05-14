import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from transformers import RobertaTokenizer, RobertaModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from torch import nn
import json
import os
import torch.nn.functional as F
from torch.optim import AdamW
import argparse
import random
import numpy as np
import matplotlib.pyplot as plt

# Set random seed for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# Parse command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--overlap_ratio', type=float, default=0.0, help='Overlap ratio between chunks (0.0 ~ 1.0)')
args = parser.parse_args()

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load data
trainDataPath = "../dataset/SDC_train_resilience_r.jsonl"
evalDataPath = "../dataset/SDC_test_resilience_r.jsonl"
train_data = pd.DataFrame({"code": [], "label": []})
eval_data = pd.DataFrame(columns=['code', 'label'])

with open(trainDataPath, "r") as data_file:
    for line in data_file:
        line = json.loads(line)
        train_data = pd.concat([train_data, pd.DataFrame([[line["code"], line["label"]]], columns=['code', 'label'])])

with open(evalDataPath, "r") as data_file:
    for line in data_file:
        line = json.loads(line)
        eval_data = pd.concat([eval_data, pd.DataFrame([[line["code"], line["label"]]], columns=['code', 'label'])])

# Initialize tokenizer
tokenizer = RobertaTokenizer.from_pretrained("neulab/codebert-cpp")
chunk_size = 512
stride = int(chunk_size * (1 - args.overlap_ratio))
if stride <= 0:
    stride = 1

# Estimate max_chunks based on training data
def estimate_max_chunks(texts):
    chunk_counts = []
    for text in texts:
        tokens = tokenizer.encode(text, add_special_tokens=True)
        total_len = len(tokens)
        if total_len < chunk_size:
            chunk_counts.append(1)
        else:
            num_chunks = max(1, (total_len - chunk_size) // stride + 1)
            chunk_counts.append(num_chunks)
    print("Max chunks:", max(chunk_counts))
    print("95th percentile chunks:", int(np.percentile(chunk_counts, 95)))
    return int(np.percentile(chunk_counts, 95))

max_chunks = estimate_max_chunks(train_data['code'].tolist())

# Dataset class
class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, chunk_size=512, overlap_ratio=0.0, max_chunks=8):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.chunk_size = chunk_size
        self.overlap_ratio = overlap_ratio
        self.max_chunks = max_chunks

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        tokens = self.tokenizer.encode(text, add_special_tokens=True, return_tensors="pt").squeeze()
        attention_mask = torch.ones_like(tokens)

        stride = int(self.chunk_size * (1 - self.overlap_ratio))
        if stride <= 0:
            stride = 1

        total_len = tokens.size(0)
        if total_len < self.chunk_size:
            pad_len = self.chunk_size - total_len
            tokens = F.pad(tokens, (0, pad_len), value=self.tokenizer.pad_token_id)
            attention_mask = F.pad(attention_mask, (0, pad_len), value=0)

        input_ids_chunks = tokens.unfold(0, self.chunk_size, stride)
        attention_mask_chunks = attention_mask.unfold(0, self.chunk_size, stride)

        # Pad or truncate to max_chunks
        num_chunks = input_ids_chunks.size(0)
        if num_chunks < self.max_chunks:
            pad_len = self.max_chunks - num_chunks
            pad_tensor = torch.full((pad_len, self.chunk_size), self.tokenizer.pad_token_id, dtype=torch.long)
            input_ids_chunks = torch.cat([input_ids_chunks, pad_tensor], dim=0)
            attention_mask_chunks = torch.cat([
                attention_mask_chunks,
                torch.zeros((pad_len, self.chunk_size), dtype=torch.long)
            ], dim=0)
        else:
            input_ids_chunks = input_ids_chunks[:self.max_chunks]
            attention_mask_chunks = attention_mask_chunks[:self.max_chunks]

        return {
            'input_ids': input_ids_chunks,
            'attention_mask': attention_mask_chunks,
            'labels': torch.tensor(label, dtype=torch.float)
        }

# Attention Pooling module
class AttentionPooling(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.attn = nn.Linear(input_dim, 1)

    def forward(self, x):  # x: [B, N, H]
        scores = self.attn(x).squeeze(-1)  # [B, N]
        weights = torch.softmax(scores, dim=1)  # [B, N]
        output = torch.sum(x * weights.unsqueeze(-1), dim=1)  # [B, H]
        return output

# Regression model with Attention Pooling
class BertRegressor(nn.Module):
    def __init__(self, bert_model="neulab/codebert-cpp", output_size=1):
        super().__init__()
        self.bert = RobertaModel.from_pretrained(bert_model)
        self.pooling = AttentionPooling(self.bert.config.hidden_size)
        self.regressor = nn.Linear(self.bert.config.hidden_size, output_size)

    def forward(self, input_ids, attention_mask):
        batch_size, seq_len, chunk_size = input_ids.size()
        input_ids = input_ids.view(-1, chunk_size).to(device)
        attention_mask = attention_mask.view(-1, chunk_size).to(device)

        with torch.no_grad():
            bert_output = self.bert(input_ids, attention_mask=attention_mask)

        cls_embeddings = bert_output.last_hidden_state[:, 0, :].view(batch_size, seq_len, -1)  # [B, N, H]
        pooled = self.pooling(cls_embeddings)  # [B, H]
        return torch.sigmoid(self.regressor(pooled))  # [B, 1]

# Build datasets and dataloaders
train_dataset = SentimentDataset(train_data['code'].to_numpy(), train_data['label'].to_numpy(), tokenizer, chunk_size=chunk_size, overlap_ratio=args.overlap_ratio, max_chunks=max_chunks)
eval_dataset = SentimentDataset(eval_data['code'].to_numpy(), eval_data['label'].to_numpy(), tokenizer, chunk_size=chunk_size, overlap_ratio=args.overlap_ratio, max_chunks=max_chunks)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, drop_last=False)
eval_loader = DataLoader(eval_dataset, batch_size=1, shuffle=False)

# Initialize model, optimizer, loss
model = BertRegressor().to(device)
optimizer = AdamW(model.parameters(), lr=5e-5)
loss_fn = nn.MSELoss()

# Training
model.train()
for epoch in range(5):
    for batch in train_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = loss_fn(outputs.squeeze(), labels)
        loss.backward()
        optimizer.step()

    # Save checkpoint
    checkpoint_dir = "BERTRegression/checkpoint-best-acc"
    os.makedirs(checkpoint_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(checkpoint_dir, "model.pt"))
    print(f"Epoch {epoch}, Loss: {loss.item()}")

# Evaluation
model.load_state_dict(torch.load(os.path.join(checkpoint_dir, "model.pt")))
model.eval()
prediction_list, label_list = [], []
total_mse, total_samples = 0, 0

for batch in eval_loader:
    with torch.no_grad():
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

        prediction_list.append(outputs.squeeze().item())
        label_list.append(labels.item())

        batch_mse = loss_fn(outputs, labels)
        print(f"Predicted: {round(outputs.squeeze().item(), 3)}, Actual: {round(labels.item(), 3)}, MSE: {batch_mse.item()}")

        total_mse += batch_mse.item()
        total_samples += 1

print("Final MSE:", total_mse / total_samples)