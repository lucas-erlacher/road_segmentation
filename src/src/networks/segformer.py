import torch
from transformers import (
    SegformerImageProcessor,
    SegformerForSemanticSegmentation,
    SegformerConfig,
)
import torch.nn as nn


class Segformer(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.model = SegformerForSemanticSegmentation.from_pretrained(
            "nvidia/segformer-b0-finetuned-ade-512-512"
        )

        def weight_reset(m):
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                m.reset_parameters()

        self.model.decode_head.classifier = nn.Conv2d(
            256, 2, kernel_size=(1, 1), stride=(1, 1)
        )
        self.model.decode_head.apply(weight_reset)

    def forward(self, x):
        return self.model(x).logits
