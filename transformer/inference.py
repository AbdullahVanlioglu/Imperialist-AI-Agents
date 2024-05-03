import torch
import time
import json

from pathlib import Path
from typing import Optional
from sentencepiece import SentencePieceProcessor
from tqdm import tqdm

from llama_transformer import ModelArgs, Transformer

class LLaMA:

    def __init__(self, model: Transformer, tokenizer: SentencePieceProcessor, model_args: ModelArgs)
        self.model = model
        self.tokenizer = tokenizer
        self.args = model_args

    @staticmethod
    def build():
        
