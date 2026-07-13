from __future__ import annotations
from langchain_core.prompts import PromptTemplate
from utils import *

refiner_prompt = PromptTemplate.from_template(
        """
        Revise the summary using the critique.

        Rules:
        - Use only information supported by the original document.
        - Fix inaccuracies.
        - Add missing important points.
        - Remove repetition.
        - Make the summary clear and concise.
        - Make sure the summary is within the word limit.
        - Make sure the summary sound human-like.
        - Avoid vague phrases such as "the text discusses" or "Section X states".


        Draft summary:
        {draft_summary}

        Critique:
        {critique}

        Minimum word length: {min_length}
        Maximum word length: {max_length}

        Revised summary:
        """
    )

def refiner_answer(state: GraphState, call_qwen) -> GraphState:
    user_prompt = refiner_prompt.format(draft_summary=state["draft_summary"], 
                                       critique=state["critique"],
                                       min_length=state["min_length"],
                                       max_length=state["max_length"])
    final = call_qwen(
        system_prompt="You are a careful editor that improves draft summaries using feedback.",
        user_prompt=user_prompt,
    )

    doc_num = state["doc_num"]
    model_file_name = f"summaries/model_summaries/model_gov_doc_{doc_num}.txt"
    with open(model_file_name, "w", encoding="utf-8") as file:
        file.write(final)

    return {**state, "final_summary": final}