def compute_length_penalty_score(model_summary, min_length, max_length):
    summary_length = len(model_summary.split())
    return min(summary_length, max_length) / max(summary_length, min_length)