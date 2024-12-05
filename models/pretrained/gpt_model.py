from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch
from models.base_model import BaseModel

class GPTModel(BaseModel):
    def __init__(self, model_name="gpt2"):
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_name)

    def train(self, data, labels=None):
        raise NotImplementedError("GPT is typically used for generation and fine-tuning.")

def predict(self, data):
    """
    Generates predictions (text) for the given input data.

    :param data: List of input text strings.
    :return: List of generated text strings.
    """
    self.model.eval()
    predictions = []
    with torch.no_grad():
        for text in data:
            # Tokenize the input
            inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)

            # Generate output sequences
            outputs = self.model.generate(
                inputs["input_ids"],
                max_length=50,
                pad_token_id=self.tokenizer.eos_token_id,  # Use EOS token as PAD
                attention_mask=inputs.get("attention_mask")
            )
            
            # Decode outputs to text
            decoded_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            predictions.append(decoded_output)
    return predictions

