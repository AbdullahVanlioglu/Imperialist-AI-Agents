import torch
import math
import torch.nn as nn

class TransformerBlock(nn.Module):
    def __init__(self,
                 height:int,
                 width:int,
                 patch_size:int,
                 layer_norm:bool=False,
                 flatten_output:bool=True,
                 ):
        super().__init__()


class PatchEmbed(nn.Module):
    def __init__(self,
                 height:int,
                 width:int,
                 patch_size:int,
                 layer_norm:bool=False,
                 flatten_output:bool=True,
                 ):
        super().__init__()

        self.flatten = flatten_output
        self.layer_norm = layer_norm

        num_patches = (height // patch_size) * (width // patch_size)

    def forward(self, latent):

        


class DiffusionTransformer2D(nn.Module):
    def __init__(self,
                 height:int,
                 width:int,
                 patch_size:int,
                 layer_norm:bool=False,
                 flatten_output:bool=True,
                 ):
        super().__init__()
        
        self.patch_embed = PatchEmbed(
            height=height: int,
            width=width: int,
            patch_size=patch_size: int,
            layer_norm=layer_norm: bool,
            flatten_output=flatten_output: bool,
        )

    def forward(self, latent):
        
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(
                height=height: int,
                width=width: int,
                patch_size=patch_size: int,
                layer_norm=layer_norm: bool,
                flatten_output=flatten_output: bool,
            ) for _ in range(num_layers)
        ])

