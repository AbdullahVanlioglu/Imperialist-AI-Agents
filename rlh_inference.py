import torch
import wandb
import tqdm

from trl import PPOTrainer, PPOConfig
from datasets import load_dataset
from transformers import pipeline, AutoTokenizer, AutoModel, AutoModelForCausalLMWithValueHead
from trl.core import LengthSampler


wandb.init()

dataset = load_dataset("imdb", split="train")

tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
model = AutoModel.from_pretrained("google-bert/bert-base-uncased")


def collator(data):
    return dict((key, [d[key] for d in data]) for key in data[0])


if __name__ == '__main__':
	config = PPOConfig(
		model_name="lvwerra/gpt2-imdb",
		learning_rate=1.41e-5,
		log_with="wandb")

	model = AutoModelForCausalLMWithValueHead.from_pretrained(config.model_name)
	ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(config.model_name)


	tokenizer = AutoTokenizer.from_pretrained(config.model_name)
	tokenizer.pad_token = tokenizer.eos_token

	ppo_trainer = PPOTrainer(config, model, ref_model, tokenizer, dataset=dataset, data_collator=collator)

	sentiment_pipe = pipeline("sentiment-analysis", model="lvwerra/distilbert-imdb", device=device)

	sent_kwargs = {"return_all_scores": True, "function_to_apply": "none", "batch_size": 16}
	text = "this movie was really bad!!"
	print(sentiment_pipe(text, **sent_kwargs))

	text = "this movie was really good!!"
	print(sentiment_pipe(text, **sent_kwargs)) # [{'label': 'NEGATIVE', 'score': -2.335047960281372}, {'label': 'POSITIVE', 'score': 2.557039737701416}]
	
	output_min_length = 4
	output_max_length = 16
	output_length_sampler = LengthSampler(output_min_length, output_max_length)

	response_generation_kwargs = {
		"min_length": -1,
		"top_k": 0.0,
		"top_p": 1.0,
		"do_sample": True,
		"pad_token_id": tokenizer.eos_token_id
		}
	
	for epoch, batch in tqdm(enumerate(ppo_trainer.dataloader)):
		query_tensors = batch["input_ids"]

		response_tensors = []
		for query in query_tensors:
			gen_len = output_length_sampler()
			response_generation_kwargs["max_new_tokens"] = gen_len
			response = ppo_trainer.generate(query, **response_generation_kwargs)
			response_tensors.append(response.squeeze()[-gen_len:])
		batch["response"] = [tokenizer.decode(r.squeeze()) for r in response_tensors]

		texts = [q + r for q,r in zip(batch["query"], batch["response"])]
		pipe_outputs = sentiment_pipe(texts, **sent_kwargs)

		rewards = [torch.tensor(output[1]["score"]) for output in pipe_outputs]

		stats = ppo_trainer.step(query_tensors, response_tensors, rewards)