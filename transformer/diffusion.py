import torch
from torch.nn import nn
from torch.nn.functional import functional as F
from attentions.attention import SelfAttention, CrossAttention

class Diffusion(nn.Module):
    def __init__(self):
        self.time_embedding = TimeEmbedding(320)
        self.unet = UNET()
        self.final = UNET_OutputLayer(320, 4)

    def forward(self, latent: torch.Tensor, context: torch.Tensor, time: torch.Tensor):
        # latent: (Batch_Size, 4, Height/8, Width/8)
        # context: (Batch_Size, Seq_Len, Dim)
        # time: (1, 320)
        