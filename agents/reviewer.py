from __future__ import annotations
from langchain_core.prompts import PromptTemplate
from utils import *

# reviewer_prompt = PromptTemplate.from_template(
#         """
#         You are a critic agent.

#         Evaluate the draft summary for:
#         1. factual consistency
#         2. missing important information
#         3. unsupported claims
#         4. clarity

#         Double check if the summary fits within the word length limits.
#         Minimum word length: {min_length}
#         Maximum word length: {max_length}

#         Make it sound human-like. Do more "tell" than "show", i.e. stating XYZ policies instead of stating there are policies pertaining to XYZ.

#         Draft summary: {draft_summary}

#         Return a concise critique.
#         """
#     )

reviewer_prompt = PromptTemplate.from_template(
    """
    You are the Reviewer Agent in a factuality-aware multi-agent summarization system.

    Your job is to evaluate a draft summary by comparing it against the original document.

    You are not rewriting the summary. You are only identifying strengths, weaknesses, factual errors, missing information, unsupported claims, and clarity issues.

    Evaluate the summary using these four criteria:

    1. Factual Consistency
    - Does the summary only contain claims supported by the original document?
    - Are any statements exaggerated, distorted, or hallucinated?

    2. Coverage
    - Does the summary include the most important points from the original document?
    - Are key methods, findings, arguments, limitations, or conclusions missing?

    3. Coherence
    - Is the summary logically organized?
    - Does it flow clearly from one idea to another?

    4. Conciseness
    - Does the summary avoid unnecessary repetition?
    - Is it too vague, too long, or too compressed?

    Important rules:
    - Use the original document as the source of truth.
    - Do not penalize the summary for omitting minor details.
    - Do penalize unsupported claims.
    - Be specific.
    - Quote or paraphrase the relevant original evidence when useful.
    - If a claim is unsupported, explain why.
    - If information is missing, explain what should be added.

    Draft summary:
    {draft_summary}


    """
)

    # Original document:
    # {document}

def reviewer_answer(state: GraphState, call_qwen=call_qwen) -> GraphState:
    user_prompt = reviewer_prompt.format(draft_summary=state["draft_summary"],
                                         document=state["document"]
                                         )
    
    draft = call_qwen(
        system_prompt="You are a helpful assistant that provides a critique on a draft summary given the original text.",
        user_prompt=user_prompt,
    )
    return {**state, "critique": draft}