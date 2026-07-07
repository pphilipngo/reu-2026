from __future__ import annotations
from langchain_core.prompts import PromptTemplate
from utils import *

# chunker_prompt = PromptTemplate.from_template(
#         """
#         Split the given user request text into chunks to send to the Summarizer Agent to summarize each chunk separately.
#         Have each chunk be a paragraph.

#         Document: {document}
#         """
#     )

# chunker_prompt = PromptTemplate.from_template(
#         """
#         Split the given document into chunks to send to the Summarizer Agent to summarize each chunk separately.
#         Have each chunk contain paragraphs that are semantically similar. Separate each chunk with "~".

#         Document: {document}
#         """
#     )

def chunk_text(text, chunk_size):
    chunked_text, overlap = [], (n / 10) * 2
    split_text = text.split()
    len_text = len(split_text)

    for i in range(0, len_text, n - overlap):
        chunk_tokens = split_text[i:i + chunk_size]
        chunks.append(chunk_tokens)

    # while len_text > 0:
    #     if len_text < n:
    #         chunked_text.append(text)
    #         break
    #     n_split = text.split(maxsplit=n)
    #     chunk, text = " ".join(n_split[:n]), n_split[n]
    #     chunked_text.append(chunk)
    #     len_text = len(text.split())
    return chunked_text

def chunker_answer(state: GraphState, call_qwen=call_qwen) -> GraphState:
    # user_prompt = chunker_prompt.format(document=state["document"])
    # draft = call_qwen(
    #     system_prompt="You are a helpful assistant that separates text to help summarize text.",
    #     user_prompt=user_prompt,
    # )

    draft = chunk_text(state["document"], state["chunk_size"])
    return {**state, "chunks": draft}
    # return {**state, "chunker_answer": draft}


