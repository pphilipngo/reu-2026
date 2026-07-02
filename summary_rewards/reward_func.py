from .rouge_reward_score import *
from .length_penalty_score import *


def reward_function(model_summaries, reference_summaries):
    rewards = []

    for model_summary, reference_summary in zip(model_summaries, reference_summaries):
        reward = 0.80 * compute_rouge_score(model_summary, reference_summary) + 0.2 * compute_length_penalty_score(model_summary, 400, 600)

        rewards.append(float(reward))

    return rewards

def reward_function_single(model_summary, reference_summary):
    reward = 0.80 * compute_rouge_score(model_summary, reference_summary) + 0.2 * compute_length_penalty_score(model_summary, 400, 600)

    return reward


# def main():
#     for i in range(1, 4):
#         text = next(ds)
#         orig_text, reference = text["report"], text["summary"]

#         reference_file_name = f"summaries/reference_summaries/ref_gov_doc_{i}"
#         with open(reference_file_name, "w", encoding="utf-8") as file:
#             file.write(reference)

# if __name__ == "__main__":
#     main()