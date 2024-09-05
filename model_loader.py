import model_converter

from encoders.clip import CLIP
from encoders.vae import VAE_Encoder, VAE_Decoder
from transformer.diffusion import Diffusion

def preload_models_from_standard_weights(ckpt_path, device):
    state_dict = model_converter.load_from_standard_weights(ckpt_path, device)

    encoder = VAE_Encoder().to(device)
    encoder.load_state_dict(state_dict["encoder"], strict=True)

    decoder = VAE_Decoder().to(device)
    decoder.load_state_dict(state_dict["encoder"], strict=True)
