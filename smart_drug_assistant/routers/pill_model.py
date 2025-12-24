import torch
import torch.nn as nn
from transformers import VisionEncoderDecoderModel

class PillMultiTaskModel(nn.Module):
    def __init__(self, model_name: str, num_labels: int):
        super().__init__()
        self.trocr = VisionEncoderDecoderModel.from_pretrained(model_name)
        self.color_head = nn.Sequential(
            nn.Linear(self.trocr.config.encoder.hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_labels)
        )

    def forward(self, pixel_values, labels=None):
        ocr_loss = None
        if labels is not None:
            outputs = self.trocr(pixel_values=pixel_values, labels=labels)
            ocr_loss = outputs.loss

        encoder_outputs = self.trocr.get_encoder()(pixel_values=pixel_values)
        pooled_features = encoder_outputs.last_hidden_state.mean(dim=1)
        color_logits = self.color_head(pooled_features)
        return ocr_loss, color_logits
