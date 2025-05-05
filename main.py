import opendatasets as od
import json
import os
import torch
import torch.nn as nn
from torch.optim import Adam
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

od.download('https://www.kaggle.com/datasets/rmisra/news-headlines-dataset-for-sarcasm-detection')

data = pd.read_json('/content/news-headlines-dataset-for-sarcasm-detection/Sarcasm_Headlines_Dataset.json', lines=True)
data.dropna(inplace=True)
data.drop_duplicates(inplace=True)
data.drop('article_link', axis=1, inplace=True)

print(data.shape)
print(data.head())

X_train, X_test, y_train, y_test = train_test_split(data['headline'].values, data['is_sarcastic'].values, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=42)

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
bert_model = AutoModel.from_pretrained("bert-base-uncased")

class SarcasmDataset(Dataset):
    def __init__(self, X, Y):
        self.X = [tokenizer(x, max_length=100, truncation=True, padding="max_length", return_tensors='pt') for x in X]
        self.Y = torch.tensor(Y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        item = {key: val.squeeze(0) for key, val in self.X[index].items()}
        item['labels'] = self.Y[index]
        return item

training_data = SarcasmDataset(X_train, y_train)
validation_data = SarcasmDataset(X_val, y_val)
testing_data = SarcasmDataset(X_test, y_test)

BATCH_SIZE = 32
EPOCHS = 10
LR = 1e-4

train_loader = DataLoader(training_data, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(validation_data, batch_size=BATCH_SIZE)
test_loader = DataLoader(testing_data, batch_size=BATCH_SIZE)

class MyModel(nn.Module):
    def __init__(self, bert):
        super(MyModel, self).__init__()
        self.bert = bert
        self.dropout = nn.Dropout(0.25)
        self.linear1 = nn.Linear(768, 384)
        self.linear2 = nn.Linear(384, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids, attention_mask):
        pooled_output = self.bert(input_ids, attention_mask=attention_mask, return_dict=False)[0][:, 0]
        output = self.linear1(pooled_output)
        output = self.dropout(output)
        output = self.linear2(output)
        return self.sigmoid(output).squeeze(1)

for param in bert_model.parameters():
    param.requires_grad = False

model = MyModel(bert_model).to('cuda')
criterion = nn.BCELoss()
optimizer = Adam(model.parameters(), lr=LR)

train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []

for epoch in range(EPOCHS):
    model.train()
    total_loss_train = 0
    correct_train = 0

    for batch in train_loader:
        input_ids = batch['input_ids'].to('cuda')
        attention_mask = batch['attention_mask'].to('cuda')
        labels = batch['labels'].to('cuda')

        outputs = model(input_ids, attention_mask)
        loss = criterion(outputs, labels)
        total_loss_train += loss.item()

        preds = outputs.round()
        correct_train += (preds == labels).sum().item()

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    model.eval()
    total_loss_val = 0
    correct_val = 0

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to('cuda')
            attention_mask = batch['attention_mask'].to('cuda')
            labels = batch['labels'].to('cuda')

            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            total_loss_val += loss.item()

            preds = outputs.round()
            correct_val += (preds == labels).sum().item()

    train_accuracy = correct_train / len(training_data) * 100
    val_accuracy = correct_val / len(validation_data) * 100

    train_losses.append(total_loss_train / len(train_loader))
    val_losses.append(total_loss_val / len(val_loader))
    train_accuracies.append(train_accuracy)
    val_accuracies.append(val_accuracy)

    print(f"Epoch {epoch+1}: Train Loss = {train_losses[-1]:.4f}, Train Acc = {train_accuracy:.2f}% | "
          f"Val Loss = {val_losses[-1]:.4f}, Val Acc = {val_accuracy:.2f}%")

model.eval()
correct_test = 0
total_loss_test = 0

with torch.no_grad():
    for batch in test_loader:
        input_ids = batch['input_ids'].to('cuda')
        attention_mask = batch['attention_mask'].to('cuda')
        labels = batch['labels'].to('cuda')

        outputs = model(input_ids, attention_mask)
        loss = criterion(outputs, labels)
        total_loss_test += loss.item()

        preds = outputs.round()
        correct_test += (preds == labels).sum().item()

test_accuracy = correct_test / len(testing_data) * 100
print(f"\nTest Accuracy: {test_accuracy:.2f}%")

fig, axs = plt.subplots(1, 2, figsize=(14, 5))

axs[0].plot(train_losses, label='Training Loss')
axs[0].plot(val_losses, label='Validation Loss')
axs[0].set_title('Loss over Epochs')
axs[0].set_xlabel('Epoch')
axs[0].set_ylabel('Loss')
axs[0].legend()

axs[1].plot(train_accuracies, label='Training Accuracy')
axs[1].plot(val_accuracies, label='Validation Accuracy')
axs[1].set_title('Accuracy over Epochs')
axs[1].set_xlabel('Epoch')
axs[1].set_ylabel('Accuracy (%)')
axs[1].legend()

plt.tight_layout()
plt.show()

def predict_sarcasm(headlines):
    model.eval()
    inputs = tokenizer(headlines, max_length=100, truncation=True, padding="max_length", return_tensors='pt')
    inputs = {key: val.to('cuda') for key, val in inputs.items()}

    with torch.no_grad():
        outputs = model(inputs['input_ids'], inputs['attention_mask'])
        predictions = outputs.cpu().numpy()
        predicted_labels = np.round(predictions).astype(int)

    for headline, pred, score in zip(headlines, predicted_labels, predictions):
        label = "Sarkastyczne" if pred == 1 else "Niesarkastyczne"
        print(f"{headline}\n → {label} (score: {score:.4f})\n")

sample_headlines = [
    "10 reasons why cats are secretly planning to take over the world",
    "Government announces new policy to improve education system",
    "Wow, another Monday. Just what I needed!",
    "Scientists discover cure for common cold"
]

predict_sarcasm(sample_headlines)
