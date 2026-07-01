""" Training the Refiner Agent with GRPO"""

from datasets import Dataset
from trl import GRPOConfig, GRPOTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer

import sys
from pathlib import Path


from utils import *
from summary_rewards import *

def reward_function(model_summaries, reference_summaries):
    rewards = []

    for model_summary, reference_summary in zip(model_summaries, reference_summaries):
        reward = 0.80 * compute_rouge_score(model_summary, reference_summary) + 0.2 * length_penalty_score(model_summary, min_length, max_length)

        rewards.append(float(reward))

    return rewards


tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    
model = AutoModelForCausalLM.from_pretrained(MODEL_ID,
                                             device_map="auto",
                                             )

training_args = GRPOConfig(output_dir="./rafts-refiner-grpo",
                           per_device_train_batch_size=1,
                           gradient_accumulation_steps=4,
                           num_generations=4,
                           max_prompt_length=4096,
                           max_completion_length=512,
                           learning_rate=1e-6,
                           logging_steps=10,
                           save_steps=100,
                           )

trainer = GRPOTrainer(model=model,
                      args=training_args,
                      train_dataset='summarie',
                      reward_funcs=reward_function,
                      )

trainer.train()
trainer.save_model("./rafts-refiner-grpo")

# refiner_model_name = "./rafts-refiner-grpo"

# refiner_tokenizer = AutoTokenizer.from_pretrained(refiner_model_name)
# refiner_model = AutoModelForCausalLM.from_pretrained(refiner_model_name,
#                                                      device_map="auto",
#                                                      )

# from datasets import load_dataset, Dataset
# import json

# dataset = load_dataset(
#     "ccdv/govreport-summarization",
#     split="train[:100]"
# )

# rows = []

