from rouge_score import rouge_scorer
from statistics import geometric_mean

# score_dict = {}
# for i in range(1, 4):
#     model_file = f'summaries/model_summaries/model_gov_doc_{i}.txt'
#     with open(model_file, 'r', encoding='utf-8') as f:
#         model_summary = f.read().strip()

#     reference_file = f'summaries/system_summaries/sys_gov_doc_{i}.txt'
#     with open(reference_file, 'r', encoding='utf-8') as f:
#         reference_summary = f.read().strip()

#     scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
#     scores = scorer.score(model_summary, reference_summary)

#     single_score_dict = {"rouge1": scores["rouge1"][2],
#                          "rouge2": scores["rouge2"][2],
#                          "rougeL": scores["rougeL"][2]}
    
#     model_file_name = f'GovReport {i}'
#     score_dict[model_file_name] = single_score_dict


def compute_rouge_score(model_summary, reference_summary):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(model_summary, reference_summary)

    single_score_dict = {"rouge1": scores["rouge1"][2],
                         "rouge2": scores["rouge2"][2],
                         "rougeL": scores["rougeL"][2]}

    single_score_list = [scores["rouge1"][2], scores["rouge2"][2], scores["rougeL"][2]]

    score = geometric_mean(single_score_list)
    return score