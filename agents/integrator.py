from __future__ import annotations
from langchain_core.prompts import PromptTemplate
from utils import *

# You are an integrator agent. Combine the following section summaries into one coherent overall summary. Avoid adding facts that are not supported by the section summaries.
#         Keep joined summary within the minimum and maximum word length. Do not be strict about being right in the middle. Make sure that the summary is grammatically correct, and it does not end without a complete sentence.
#         Make it sound human-like. Do more "tell" than "show", i.e. stating XYZ policies instead of stating there are policies pertaining to XYZ.


integrator_prompt = PromptTemplate.from_template(
        """
        Your job is to combine multiple section-level summaries into one coherent draft summary of the full document.
        You will combine summaries sequentially, creating a globla rolling summary. 

        You must follow these rules:
        - Use only the provided section summaries.
        - Do not add outside information.
        - Do not invent missing context.
        - Preserve the document's main argument, structure, findings, and conclusions.
        - Remove redundancy between section summaries.
        - Keep the final draft logically organized.
        - Maintain a neutral and objective tone.
        - If the document includes methods, results, limitations, or implications, preserve those distinctions.
        - If there are disagreements, caveats, or uncertainty, include them.
        - Do not mention that the summary was produced from chunks unless explicitly necessary.
        - Keep the draft within the word length. 
        - Make the summary sound human-like.
        - Avoid vague phrases such as "the text discusses" or "Section X states".


        Section summaries: {joined_summaries}
        Minimum word length: {min_length}
        Maximum word length: {max_length}

        Integrated draft:
        """
    )

test_prompt = PromptTemplate.from_template(
        """
        Your job is to combine multiple section-level summaries into one coherent draft summary of the full document.
        You will combine summaries sequentially, creating a globla rolling summary. 
        You are at summary number {next_sum_num} out of out of {num_summaries}.

        You must follow these rules:
        - Use only the provided section summaries.
        - Do not add outside information.
        - Do not invent missing context.
        - Preserve the document's main argument, structure, findings, and conclusions.
        - Remove redundancy between section summaries.
        - Keep the final draft logically organized.
        - Maintain a neutral and objective tone.
        - If the document includes methods, results, limitations, or implications, preserve those distinctions.
        - If there are disagreements, caveats, or uncertainty, include them.
        - Do not mention that the summary was produced from chunks unless explicitly necessary.
        - Keep the draft within the word length. 
        - Make the summary sound human-like.
        - Avoid vague phrases such as "the text discusses" or "Section X states".


        Global rolling summary: {global_summary}
        Next summary: {next_summary}
        Minimum word length: {min_length}
        Maximum word length: {max_length}

        Integrated draft:
        """
    )

def integrator_answer(state: GraphState, call_qwen=call_qwen) -> GraphState:
    # user_prompt = integrator_prompt.format(user_request=state["user_request"],
    #                                        summarizer_answer=state["summarizer_answer"],
    #                                        min_length=state["min_length"],
    #                                        max_length=state["max_length"])
    
    # draft = call_qwen(
    #     system_prompt="You are a helpful assistant that helps summarize text.",
    #     user_prompt=user_prompt,
    # )

    joined_summaries = "\n\n".join(
        f"Section {i + 1} Summary:\n{summary}"
        for i, summary in enumerate(state["chunk_summaries"])
    )

    user_prompt = integrator_prompt.format(joined_summaries=joined_summaries,
                                           min_length=state["min_length"],
                                           max_length=state["max_length"])

    draft = call_qwen(
        system_prompt="You are a helpful assistant that combines summaries together to form one cohesive summary.",
        user_prompt=user_prompt,
        )
    
    return {**state, "draft_summary": draft}


def test_answer(state: GraphState, call_qwen=call_qwen) -> GraphState:
    chunk_summaries = state["chunk_summaries"]
    global_summary, num_summaries = chunk_sums[0], len(chunk_sums)
    for i in range(1, num_summaries):
        user_prompt = test_prompt.format(next_sum_num=i + 1,
                                        num_summaries=num_summaries,
                                       global_summary=global_summary,
                                       next_summary=chunk_summaries[i],
                                       min_length=state["min_length"],
                                       max_length=state["max_length"])
        
        global_summary = call_qwen(
            system_prompt="You are a helpful assistant that combines summaries together to form one cohesive summary.",
            user_prompt=user_prompt,
            )
        
    return {**state, "draft_summary": global_summary}