import torch
import wandb

from trl import PPOTrainer, PPOConfig
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLMWithValueHead


wandb.init()

dataset = load_dataset("imdb", split="train")

tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
model = AutoModel.from_pretrained("google-bert/bert-base-uncased")


config = PPOConfig(
        model_name="lvwerra/gpt2-imdb",
        learning_rate=1.41e-5,
        log_with="wandb")

model = AutoModelForCausalLMWithValueHead.from_pretrained(config.model_name)
ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(config.model_name)


tokenizer = AutoTokenizer.from_pretrained(config.model_name)
tokenizer.pad_token = tokenizer.eos_token

ppo_trainer = PPOTrainer(config, model, ref_model, tokenizer, dataset=dataset, data_collator=collator)
