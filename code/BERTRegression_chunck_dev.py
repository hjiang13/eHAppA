import os
import torch
import time
import json
import random
import numpy as np
import pandas as pd
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import RobertaTokenizer, RobertaModel
from peft import get_peft_model, LoraConfig, TaskType

# Fix seed for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# Configuration
EPOCHS = 3
BATCH_SIZE = 8
USE_LORA_OPTIONS = [False, True]
POOLING_TYPE = "lstm"
OVERLAP_RATIO = 0.5
CHUNK_SIZE = 512
TRAIN_PATH = "../dataset/SDC_train_resilience_r.jsonl"
EVAL_PATH = "../dataset/SDC_test_resilience_r.jsonl"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load data
def load_data(path):
    data = []
    with open(path, "r") as f:
        for line in f:
            entry = json.loads(line)
            data.append([entry["code"], entry["label"]])
    return pd.DataFrame(data, columns=["code", "label"])

train_data = load_data(TRAIN_PATH)
eval_data = load_data(EVAL_PATH)

# Tokenizer
tokenizer = RobertaTokenizer.from_pretrained("neulab/codebert-cpp")

# Estimate max_chunks
def estimate_max_chunks(texts, chunk_size=512, overlap_ratio=0.0):
    stride = int(chunk_size * (1 - overlap_ratio))
    chunk_counts = []
    for text in texts:
        tokens = tokenizer.encode(text, add_special_tokens=True)
        total_len = len(tokens)
        if total_len < chunk_size:
            chunk_counts.append(1)
        else:
            num_chunks = max(1, (total_len - chunk_size) // stride + 1)
            chunk_counts.append(num_chunks)
    return int(np.percentile(chunk_counts, 95))

MAX_CHUNKS = estimate_max_chunks(train_data["code"], CHUNK_SIZE, OVERLAP_RATIO)

# Dataset
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
        total_len = tokens.size(0)
        if total_len < self.chunk_size:
            pad_len = self.chunk_size - total_len
            tokens = F.pad(tokens, (0, pad_len), value=self.tokenizer.pad_token_id)
            attention_mask = F.pad(attention_mask, (0, pad_len), value=0)
        input_ids_chunks = tokens.unfold(0, self.chunk_size, stride)
        attention_mask_chunks = attention_mask.unfold(0, self.chunk_size, stride)
        num_chunks = input_ids_chunks.size(0)
        if num_chunks < self.max_chunks:
            pad_len = self.max_chunks - num_chunks
            pad_tensor = torch.full((pad_len, self.chunk_size), self.tokenizer.pad_token_id, dtype=torch.long)
            input_ids_chunks = torch.cat([input_ids_chunks, pad_tensor], dim=0)
            attention_mask_chunks = torch.cat([attention_mask_chunks, torch.zeros((pad_len, self.chunk_size), dtype=torch.long)], dim=0)
        else:
            input_ids_chunks = input_ids_chunks[:self.max_chunks]
            attention_mask_chunks = attention_mask_chunks[:self.max_chunks]
        return {
            'input_ids': input_ids_chunks,
            'attention_mask': attention_mask_chunks,
            'labels': torch.tensor(label, dtype=torch.float)
        }

# Model
class BertRegressor(nn.Module):
    def __init__(self, bert_model="neulab/codebert-cpp", output_size=1, use_lora=False, pooling_type="mean", lstm_hidden_size=512):
        super().__init__()
        base_model = RobertaModel.from_pretrained(bert_model)
        if use_lora:
            lora_config = LoraConfig(
                task_type=TaskType.FEATURE_EXTRACTION,
                inference_mode=False,
                r=8,
                lora_alpha=32,
                lora_dropout=0.1,
                target_modules=["query", "value"]
            )
            self.bert = get_peft_model(base_model, lora_config)
        else:
            self.bert = base_model

        self.pooling_type = pooling_type
        hidden_dim = self.bert.config.hidden_size
        if pooling_type == "lstm":
            self.lstm = nn.LSTM(hidden_dim, lstm_hidden_size, batch_first=True)
            self.regressor = nn.Linear(lstm_hidden_size, output_size)
        else:
            self.regressor = nn.Linear(hidden_dim, output_size)

    def forward(self, input_ids, attention_mask):
        batch_size, seq_len, chunk_size = input_ids.size()
        input_ids = input_ids.view(-1, chunk_size).to(DEVICE)
        attention_mask = attention_mask.view(-1, chunk_size).to(DEVICE)
        with torch.no_grad():
            bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_embeddings = bert_output.last_hidden_state[:, 0, :].view(batch_size, seq_len, -1)
        if self.pooling_type == "mean":
            pooled = torch.mean(cls_embeddings, dim=1)
        elif self.pooling_type == "max":
            pooled, _ = torch.max(cls_embeddings, dim=1)
        elif self.pooling_type == "lstm":
            _, (hidden, _) = self.lstm(cls_embeddings)
            pooled = hidden.squeeze(0)
        return torch.sigmoid(self.regressor(pooled))

# Run comparison
results = {}

for use_lora in USE_LORA_OPTIONS:
    tag = "LoRA" if use_lora else "Full"
    print(f"\n==== Training with {tag} Fine-Tuning ====")

    train_dataset = SentimentDataset(train_data["code"].to_numpy(), train_data["label"].to_numpy(), tokenizer,
                                     chunk_size=CHUNK_SIZE, overlap_ratio=OVERLAP_RATIO, max_chunks=MAX_CHUNKS)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = BertRegressor(use_lora=use_lora, pooling_type=POOLING_TYPE).to(DEVICE)
    optimizer = AdamW(model.parameters(), lr=5e-5)
    loss_fn = nn.MSELoss()

    def count_trainable_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    total = sum(p.numel() for p in model.parameters())
    trainable = count_trainable_parameters(model)
    print(f"Total Params: {total:,}, Trainable: {trainable:,} ({trainable/total:.2%})")

    train_losses = []
    start_time = time.time()
    for epoch in range(EPOCHS):
        epoch_loss = 0
        for batch in train_loader:
            model.train()
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            labels = batch['labels'].to(DEVICE)

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = loss_fn(outputs.squeeze(), labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch + 1}: Loss = {avg_loss:.4f}")
        train_losses.append(avg_loss)
    total_time = time.time() - start_time
    results[tag] = {
        "train_time": total_time,
        "losses": train_losses,
        "params": trainable,
    }

# Plot results
plt.figure()
for tag in results:
    plt.plot(results[tag]["losses"], label=f"{tag} (Time: {results[tag]['train_time']:.1f}s)")
plt.xlabel("Epoch")
plt.ylabel("Training Loss")
plt.title(f"Training Loss Comparison ({POOLING_TYPE} pooling)")
plt.legend()
plt.tight_layout()
plt.savefig("lora_vs_full_finetune_loss.png")
plt.show()