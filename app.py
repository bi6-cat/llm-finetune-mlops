from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

app = FastAPI()

model_path = "./outputs"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

class InputText(BaseModel):
    text: str

@app.post("/predict/")
async def predict(input: InputText):
    inputs = tokenizer(input.text, return_tensors="pt", truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    probs = torch.softmax(logits, dim=1)
    pred_label = torch.argmax(probs).item()
    confidence = probs[0][pred_label].item()
    return {"label": pred_label, "confidence": confidence}

@app.get("/")
async def root():
    return {"message": "API is running. Use /docs to see endpoints."}