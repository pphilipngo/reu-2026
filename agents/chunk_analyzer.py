from __future__ import annotations
from langchain_core.prompts import PromptTemplate
from utils import *

def parse_score(text: str) -> int:
    match = re.search(r"\b([1-4])\b", text.strip())

    if match:
        return int(match.group(1))

    return 1

def chunk_analyzer_answer(state: GraphState, call_qwen_batch) -> GraphState:
    chunks = state["chunks"]
    summaries = {}

    # chunks = state["chunker_answer"].split("~")
    # summaries = []
    user_prompts = []
    for i, chunk in enumerate(chunks):
        chunker_analyzer_prompt = f"""
        You are a chunk analyzer. Score this chunk from 1 to 4 based on how important it is to the overall document.

        1 = irrelevant
        2 = somewhat relevant
        3 = important
        4 = essential

        Return only one number: 1, 2, 3, or 4.

        Chunk {i + 1} of {len(chunks)}:
        {chunk}

        Score:
        """.strip()

        user_prompts.append(prompt)

    outputs = call_qwen_batch(system_prompt=("You evaluate document chunks. Return only a numeric score from 1 to 4."),
                                user_prompts=user_prompts,
                                batch_size=3,
                                max_new_tokens=4,
                                do_sample=False,)

    scores = [parse_score(output) for output in outputs]

    filtered_chunks = [chunk for chunk, score in zip(chunks, scores) if score >= 3]

    return {**state, "chunk_scores": scores, "filtered_chunks": filtered_chunks}

    #     score = call_qwen(
    #         system_prompt="You are a chunker analyzer. Only choose the most important chunks.",
    #         user_prompt=chunker_analyzer_prompt)
        
        
    #     summaries[i] = int(score)

    # filtered_index = {key: value for key, value in summaries.items() if value >= 3}

    # filtered_chunks = [chunks[key] for key in filtered_index.keys()]

    # return {**state, "filtered_chunks": filtered_chunks}
    # return {**state, }