from __future__ import annotations
from langchain_core.prompts import PromptTemplate
from utils import *


def chunk_analyzer_answer(state: GraphState, call_qwen=call_qwen) -> GraphState:
    # user_prompt = summarizer_prompt.format(user_request=state["user_request"],
    #                                        chunker_answer=state["chunker_answer"])

    # draft = call_qwen(
    #     system_prompt="You are a helpful assistant that helps summarize text.",
    #     user_prompt=user_prompt,
    # )

    
    chunks = state["chunks"]
    summaries = {}

    # chunks = state["chunker_answer"].split("~")
    # summaries = []
    for i, chunk in enumerate(chunks):
        chunker_analyzer_prompt = f"""
        You are a chunker analyzer. Given the chunk, score it on a scale of 1-4 on how important this chunk is to the overall document.
        Determine whether this chunk is necessary for the final summary. Reference the original document. Only output the numeric score, nothing else.
        Original document: 
        {state["document"]}
        
        Chunk {i + 1} of {len(chunks)}:
        {chunk}
        
        Score:
        """

        score = call_qwen(
            system_prompt="You are a chunker analyzer. Only choose the most important chunks.",
            user_prompt=chunker_analyzer_prompt)
        
        
        summaries[i - 1] = int(score)

    filtered_index = {key: value for key, value in summaries.items() if value >= 3}

    filtered_chunks = [chunks[key] for key in filtered_index.keys()]

    return {**state, "filtered_chunks": filtered_chunks}
    # return {**state, }


