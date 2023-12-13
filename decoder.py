import torch
from transformers import DistilGPT2LMHeadModel, GPT2Tokenizer

class ImageCaptioningModel:
    def __init__(self, model_name="distilgpt2"):
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = DistilGPT2LMHeadModel.from_pretrained(model_name)

    def combine_features(self, image_features, text_features):

        return text_features

    def generate_caption(self, combined_features):
        inputs = self.tokenizer.encode(combined_features, return_tensors="pt", add_special_tokens=True)
        outputs = self.model.generate(inputs, max_length=50, num_beams=5, early_stopping=True)
        caption = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return caption

    def forward(self, image_features, text_features):
        combined_features = self.combine_features(image_features, text_features)
        caption = self.generate_caption(combined_features)
        return caption

