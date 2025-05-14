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

# Dataset class
class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_token_len=512, chunk_size=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_token_len = max_token_len
        self.chunk_size = chunk_size

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        tokens = self.tokenizer.encode_plus(
            text,
            max_length=self.max_token_len,
            truncation=True,
            padding='max_length',
            add_special_tokens=True,
            return_tensors="pt"
        )
        input_ids = tokens['input_ids'].squeeze()
        attention_mask = tokens['attention_mask'].squeeze()

        total_chunks = len(input_ids) // self.chunk_size
        input_ids_chunks = input_ids.unfold(0, self.chunk_size, self.chunk_size)[:total_chunks]
        attention_mask_chunks = attention_mask.unfold(0, self.chunk_size, self.chunk_size)[:total_chunks]

        return {
            'input_ids': input_ids_chunks,
            'attention_mask': attention_mask_chunks,
            'labels': torch.tensor(label, dtype=torch.float)
        }

# Load tokenizer
tokenizer = RobertaTokenizer.from_pretrained("neulab/codebert-cpp")

# Build datasets and dataloaders
train_dataset = SentimentDataset(train_data['code'].to_numpy(), train_data['label'].to_numpy(), tokenizer)
eval_dataset = SentimentDataset(eval_data['code'].to_numpy(), eval_data['label'].to_numpy(), tokenizer)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
eval_loader = DataLoader(eval_dataset, batch_size=1)

# Regression model with LSTM
class BertRegressor(nn.Module):
    def __init__(self, bert_model="neulab/codebert-cpp", lstm_hidden_size=512, output_size=1):
        super().__init__()
        self.bert = RobertaModel.from_pretrained(bert_model)
        self.lstm = nn.LSTM(self.bert.config.hidden_size, lstm_hidden_size, batch_first=True)
        self.regressor = nn.Linear(lstm_hidden_size, output_size)

    def forward(self, input_ids, attention_mask):
        batch_size, seq_len, chunk_size = input_ids.size()
        input_ids = input_ids.view(-1, chunk_size).to(device)
        attention_mask = attention_mask.view(-1, chunk_size).to(device)

        with torch.no_grad():
            bert_output = self.bert(input_ids, attention_mask=attention_mask)

        cls_embeddings = bert_output.last_hidden_state[:, 0, :].view(batch_size, seq_len, -1)
        _, (hidden, _) = self.lstm(cls_embeddings)

        return torch.sigmoid(self.regressor(hidden.squeeze(0)))

# Initialize model, optimizer, loss
model = BertRegressor().to(device)
optimizer = AdamW(model.parameters(), lr=5e-5)
loss_fn = nn.MSELoss()

# Training
model.train()
for epoch in range(1):
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