import torch
import math
import torch.nn as nn

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
        


class DiT(nn.Module):
    def __init__(self):
        
        self.patch_embed = self.PatchEmbed()