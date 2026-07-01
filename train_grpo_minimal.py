"""
train_grpo_minimal.py

The simplest possible GRPO training script for your RAFTS refiner.

Input CSV needs at least:

    model_summary,reference_summary

Example:

    python train_grpo_minimal.py \
      --data_path summaries/summaries.csv \
      --draft_col model_summary \
      --reference_col reference_summary \
      --model_id Qwen/Qwen3-0.6B \
      --max_steps 5

For your original model:

    python train_grpo_minimal.py \
      --data_path summaries/summaries.csv \
      --draft_col model_summary \
      --reference_col reference_summary \
      --model_id Qwen/Qwen3-1.7B \
      --max_steps 5
"""

from __future__ import annotations

import argparse
from typing import List

from datasets import load_dataset
from rouge_score import rouge_scorer
from transformers import AutoTokenizer
from trl import GRPOConfig, GRPOTrainer


def build_prompt(draft_summary: str) -> str:
    return f"""You are the Refiner Agent in a multi-agent summarization system.

Rewrite the draft summary to make it more accurate, clear, concise, and complete.

Rules:
- Preserve only factual claims.
- Remove repetition.
- Keep the summary coherent.
- Output only the revised summary.

Draft summary:
{draft_summary}

Revised summary:
"""


def make_reward_function(reference_col: str):
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

    def reward_func(completions: List[str], **kwargs) -> List[float]:
        references = kwargs[reference_col]
        rewards = []

        for completion, reference in zip(completions, references):
            completion = completion.strip()
            reference = str(reference).strip()

            if not completion:
                rewards.append(-1.0)
                continue

            rouge_l = scorer.score(reference, completion)["rougeL"].fmeasure

            words = completion.split()
            if 75 <= len(words) <= 350:
                length_score = 1.0
            else:
                length_score = 0.0

            reward = 0.85 * rouge_l + 0.15 * length_score
            rewards.append(float(reward))

        return rewards

    return reward_func


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_path", required=True)
    parser.add_argument("--draft_col", default="model_summary")
    parser.add_argument("--reference_col", default="reference_summary")
    parser.add_argument("--model_id", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--output_dir", default="./rafts-refiner-grpo-minimal")

    parser.add_argument("--max_steps", type=int, default=5)
    parser.add_argument("--num_generations", type=int, default=2)
    parser.add_argument("--max_prompt_length", type=int, default=1024)
    parser.add_argument("--max_completion_length", type=int, default=256)

    return parser.parse_args()


def main():
    args = parse_args()

    dataset = load_dataset("csv", data_files=args.data_path, split="train")

    dataset = dataset.filter(
        lambda row: row.get(args.draft_col) is not None
        and row.get(args.reference_col) is not None
        and str(row.get(args.draft_col)).strip() != ""
        and str(row.get(args.reference_col)).strip() != ""
    )

    def add_prompt(row):
        row["prompt"] = build_prompt(str(row[args.draft_col]))
        return row

    dataset = dataset.map(add_prompt)

    print(dataset)
    print("Example prompt:\n")
    print(dataset[0]["prompt"][:1500])

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    training_args = GRPOConfig(
        output_dir=args.output_dir,
        max_steps=args.max_steps,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=2,
        num_generations=args.num_generations,
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
        learning_rate=1e-6,
        logging_steps=1,
        save_steps=args.max_steps,
        report_to="none",
        remove_unused_columns=False,
    )

    trainer = GRPOTrainer(
        model=args.model_id,
        args=training_args,
        train_dataset=dataset,
        reward_funcs=make_reward_function(args.reference_col),
        processing_class=tokenizer,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    print(f"Saved trained model to {args.output_dir}")


if __name__ == "__main__":
    main()
