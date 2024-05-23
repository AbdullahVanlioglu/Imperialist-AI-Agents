import torch
import pickle
import logging
import numpy as np
import gym

from typing import Dict, List

from config.config import q_transformer_config
from transformer.q_transformer import Transformer, ModelArgs
from transformer.maxvit_transformer import MaxViT, max_vit_base_224


class QTransformer:
    def __init__(self, model: Transformer, vit: MaxViT, config: Dict, args: ModelArgs):
        self.model = model
        self.vit = vit
        self.config: dict = config
        self.vit: MaxViT



    @staticmethod
    def build(config: Dict, args: ModelArgs):
        vision_transformer = max_vit_base_224(in_channels=4)
        model = Transformer
        args = ModelArgs()
        

        return QTransformer(model = model,
                            vit = vision_transformer,
                            config = config, 
                            args = args
                            )

        




if __name__ == '__main__':
    torch.manual_seed(0)

    logging.info("CUDA is available:", torch.cuda.is_available())
    allow_cuda = False
    device = 'cuda' if torch.cuda.is_available() and allow_cuda else 'cpu'

    # Inputs
    img = torch.randn(1, 3, 224, 224)
    text = torch.randint(0, 20000, (1, 1024))

    config = q_transformer_config
    # model = QTransformer.build(config = config, 
    #                           args = ModelArgs())

    # Test Vision Transformer
    vision_transformer = max_vit_base_224(in_channels=3)
    output = vision_transformer(img)
    print(output.shape)
