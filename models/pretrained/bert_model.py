from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import AdamW
import torch
from models.base_model import BaseModel

class BERTModel(BaseModel):
    def __init__(self, model_name="bert-base-uncased", num_labels=3):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

    def train(self, data, labels, epochs=3, learning_rate=5e-5):
        optimizer = AdamW(self.model.parameters(), lr=learning_rate)
        self.model.train()
        
        for epoch in range(epochs):
            for text, label in zip(data, labels):
                inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
                labels_tensor = torch.tensor([label]).unsqueeze(0)
                outputs = self.model(**inputs, labels=labels_tensor)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

    def predict(self, data):
        self.model.eval()
        predictions = []
        with torch.no_grad():
            for text in data:
                inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
                outputs = self.model(**inputs)
                predictions.append(torch.argmax(outputs.logits, dim=-1).item())
        return predictions
