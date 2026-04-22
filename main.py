from fastapi import FastAPI
from pydantic import BaseModel
import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_PATH = "./transformer_model"

if not os.path.exists(MODEL_PATH):
    print("Model not found. Initiating training...")
    from train_model import deploy_model
    deploy_model()


app = FastAPI()

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()


class InputData(BaseModel):
    text: str

@app.post("/predict")
def predict(data: InputData): ## Equivalente a su score.py
    
    encoded = tokenizer(
        data.text,
        padding=True, 
        truncation=True, 
        return_tensors="pt", 
        max_length=64
    )
    
    with torch.no_grad():
        outputs = model(**encoded)
        logits = outputs.logits
        probs = torch.nn.functional.softmax(logits, dim=-1)[0]
        pred = torch.argmax(logits, dim=-1).item()

    return {
        "prediction": pred,
        "probability": float(probs[pred])
    }

