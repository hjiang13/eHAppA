import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from transformers import BertTokenizer, BertModel, AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from torch import nn
import json
import os
import torch.nn.functional as F

from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
                          RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer)
from transformers import LongformerConfig, LongformerModel

# Load the data
#data = pd.DataFrame({
#    "code": ["I love this movie!", "This book is amazing.", "The weather today is terrible."],
#    "label": [0.8, 0.9, 0.2]
#})
trainDataPath = "../dataset/SDC_train_resilience_r.jsonl"
evalDataPath = "../dataset/SDC_test_resilience_r.jsonl"
train_data = pd.DataFrame( {"code": [], "label": []}) 
val_data = pd.DataFrame( columns=['code', 'label'])

with open(trainDataPath, "r") as data_file:
    i = 0
    for line in data_file:
        line = json.loads(line)
        lineList= [[line["code"], line["label"]]]
        df_line = pd.DataFrame(lineList, columns=['code', 'label'])
        train_data = pd.concat([train_data, df_line])
        i += 1
        #if i > 5:
        #    break

with open(evalDataPath, "r") as data_file:
    i = 0
    for line in data_file:
        line = json.loads(line)
        lineList= [[line["code"], line["label"]]]
        df_line = pd.DataFrame(lineList, columns=['code', 'label'])
        val_data = pd.concat([val_data, df_line])
        i += 1
        #if i > 5:
        #    break


#data = pd.read_json("../dataset/benign_train_resilience_r.jsonl")

# define a datasets
class SentimentDataset(Dataset):
    def __init__(self, codes, labels, tokenizer, max_len=512):
        self.codes = codes
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __len__(self):
        return len(self.codes)
    
    def __getitem__(self, item):
        code = str(self.codes[item])
        label = self.labels[item]

        encoding = self.tokenizer.encode_plus(
            code,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True
        )

        return {
            'code': code,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.float)
        }

# Initialize tokenizer
tokenizer = RobertaTokenizer.from_pretrained("neulab/codebert-cpp")

# Prepare the dataset
#train_data, val_data = train_test_split(data, test_size=0.1)
train_dataset = SentimentDataset(train_data['code'].to_numpy(), train_data['label'].to_numpy(), tokenizer)
val_dataset = SentimentDataset(val_data['code'].to_numpy(), val_data['label'].to_numpy(), tokenizer)

train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=1)

# Define a regression model on BERT
# Define a regression model on BERT
class BertRegressor(nn.Module):
    def __init__(self):
        super(BertRegressor, self).__init__()
        self.bert = BertModel.from_pretrained('neulab/codebert-cpp', num_labels=1)
        self.regressor = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, 512), 
            nn.ReLU(), # Changed to ReLU for intermediate layers
            nn.Dropout(0.1), # Added dropout for regularization
            nn.Linear(512, 128),
            nn.ReLU(), # Using ReLU again
            nn.Linear(128, 1) # No activation function here to allow any range of output values
        )
    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        # Apply sigmoid activation function to squash the output to the range [0, 1]
        return F.sigmoid(self.regressor(pooled_output))

model = BertRegressor()

# Def optimizer and loss function
optimizer = AdamW(model.parameters(), lr=2e-5)
loss_fn = nn.MSELoss()


# Train the model
model.train()
best_acc = -10
for epoch in range(10):  # To be changed
    total_accuracy = 0
    total_samples = 0
    prediction_list = []
    label_list = []
    for batch in train_loader:
        optimizer.zero_grad()
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = loss_fn(outputs.squeeze(), labels)
        prediction_list.append(outputs.squeeze())
        label_list.append(batch['labels'])
        loss.backward()
        optimizer.step()
        # Calculate accuracy for this batch
        batch_accuracy = torch.abs(outputs.squeeze() - labels)
        mask = labels != 0  # Create a mask where label value is not equal to 0
        batch_accuracy[mask] = 1 - batch_accuracy[mask] / labels[mask]  # Calculate accuracy for non-zero labels
        batch_accuracy[~mask] = 0  # Set accuracy to 0 where label value is 0
        batch_samples = labels.size(0)
        
        batch_accuracy = batch_accuracy.mean().item()  # Average accuracy per sample
        batch_samples = labels.size(0)

        total_accuracy += batch_accuracy * batch_samples

        total_samples += batch_samples

    # Calculate overall accuracy
    accuracy = total_accuracy / total_samples
    print("Accuracy:", accuracy)
    
    
    if (accuracy > best_acc):
        best_acc = accuracy
        checkpoint_prefix = 'checkpoint-best-acc'
        output_dir = os.path.join("BERTRegression", '{}'.format(checkpoint_prefix)) 
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)      
        model_to_save = model.module if hasattr(model,'module') else model
        output_dir = os.path.join(output_dir, '{}'.format('model.bin')) 
        torch.save(model_to_save.state_dict(), output_dir)
    print(f"Epoch {epoch}, Loss: {loss.item()} \n")
    print(f"Best acc {best_acc}, Epoch: {epoch} \n")

    
    
    if (accuracy > best_acc):
        best_acc = accuracy
        checkpoint_prefix = 'checkpoint-best-acc'
        output_dir = os.path.join("BERTRegression", '{}'.format(checkpoint_prefix)) 
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)      
        model_to_save = model.module if hasattr(model,'module') else model
        output_dir = os.path.join(output_dir, '{}'.format('model.bin')) 
        torch.save(model_to_save.state_dict(), output_dir)
    print(f"Epoch {epoch}, Loss: {loss.item()} \n")
    print(f"Best acc {best_acc}, Epoch: {epoch} \n")


# Evaluation
#model.load_state_dict(torch.load(output_dir))
model.eval()
prediction_list = []
label_list = []
total_accuracy = 0
total_samples = 0
for batch in val_loader:
    with torch.no_grad():
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']  # Get labels from batch

        # Forward pass
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        
        # Print predicted and actual labels for debugging
        print(f"Predicted label: {outputs.squeeze().item()}, Actual label: {labels.item()}")

        # Append predictions and labels
        prediction_list.append(outputs.squeeze().item())
        label_list.append(labels.item())

        # Calculate accuracy for this batch
        batch_accuracy = torch.abs(outputs.squeeze() - labels)
        mask = labels != 0  # Create a mask where label value is not equal to 0
        batch_accuracy[mask] = 1 - batch_accuracy[mask] / labels[mask]
        batch_accuracy[~mask] = 0  # Set accuracy to 0 where label value is 0
        batch_accuracy = batch_accuracy.sum().item() / labels.size(0)  # Average accuracy per sample
        batch_samples = labels.size(0)
    
        total_accuracy += batch_accuracy * batch_samples
        total_samples += batch_samples

# Calculate overall accuracy
accuracy = total_accuracy / total_samples
print("Accuracy:", accuracy)
    
