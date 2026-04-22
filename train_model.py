# train_model.py
import kagglehub
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.optim import AdamW
import os

class SentimentDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

    def __len__(self):
        return len(self.labels)

def deploy_model():
    print("Downloading dataset...")
    path = kagglehub.dataset_download("sbhatti/financial-sentiment-analysis")
    print(f"Path to dataset files: {path}")
    
    data = pd.read_csv(os.path.join(path, "data.csv"))
    
    # Assign integer labels: negative=0, neutral=1, positive=2
    label_map = {"negative": 0, "neutral": 1, "positive": 2}
    data["label"] = data["Sentiment"].str.lower().map(label_map)
    data = data.dropna(subset=["label"])
    
    # "Mega Ultra Turbo Lite Fast" 
    data = data.sample(800, random_state=42)
    
    sentences = data["Sentence"].tolist()
    labels = data["label"].tolist()
    
    model_name = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)
    
    print("Tokenizing data...")
    encodings = tokenizer(sentences, truncation=True, padding=True, max_length=64)
    dataset = SentimentDataset(encodings, labels)
    loader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()
    
    optimizer = AdamW(model.parameters(), lr=5e-5)
    
    print("Fine-tuning Transformer model...")
    for epoch in range(1): # 1 epoca porque entrenar muchas tarda mucho, pero se puede aumentar
        for batch in loader:
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            target = batch['labels'].to(device)
            
            outputs = model(input_ids, attention_mask=attention_mask, labels=target)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            
    print("Saving fine-tuned transformer...")
    
    model.save_pretrained("./transformer_model")
    tokenizer.save_pretrained("./transformer_model")
    print("Training complete.")

if __name__ == "__main__":
    deploy_model()
