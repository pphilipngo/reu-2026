from __future__ import annotations
from langchain_core.prompts import PromptTemplate
from utils import *


def summarizer_answer(state: GraphState, call_qwen=call_qwen) -> GraphState:
    # user_prompt = summarizer_prompt.format(user_request=state["user_request"],
    #                                        chunker_answer=state["chunker_answer"])

    # draft = call_qwen(
    #     system_prompt="You are a helpful assistant that helps summarize text.",
    #     user_prompt=user_prompt,
    # )

    
    chunks = state["filtered_chunks"]
    summaries = []
    for i, chunk in enumerate(chunks):
        summarizer_prompt = f"""
        You are the Section Summarizer Agent in a multi-agent long-document summarization system.

        Your job is to summarize one document chunk accurately and concisely.

        You must follow these rules:
        - Use only information from the provided chunk.
        - Do not introduce outside information.
        - Do not guess missing facts.
        - Preserve important names, dates, organizations, findings, claims, evidence, and conclusions.
        - Keep the summary faithful to the original text.
        - If the chunk contains uncertainty, disagreement, or limitations, include them.
        - If the chunk is mostly background or setup, say so clearly.
        - Avoid vague phrases such as "the text discusses" or "Section X states".
        - Do not over-compress important technical details.
        
        Chunk {i + 1} of {len(chunks)}:
        {chunk}
        
        Section summary:
        """

        summary = call_qwen(
            system_prompt="You are a section summarizer. Summarize the following document chunk clearly and faithfully.",
            user_prompt=summarizer_prompt)
        
        summaries.append(summary)

    return {**state, "chunk_summaries": summaries}





