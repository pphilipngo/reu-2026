from bert_score import score, BERTScorer

_scorer = None

def get_bertscore_scorer():
    global _scorer
    if _scorer is None:
        _scorer = BERTScorer(lang="en", rescale_with_baseline=True)
    return _scorer

def compute_bert_score(model_summary, reference_summary):
    scorer = get_bertscore_scorer()

    P, R, F1 = scorer.score(
        [model_summary],
        [reference_summary]
    )

    return F1.item()
