from __future__ import annotations

from typing import TypedDict

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_huggingface import HuggingFacePipeline

from agents import * 

MODEL_ID = "Qwen/Qwen3-1.7B"
min_length = 400
max_length = 600

class GraphState(TypedDict):

    user_request: str
    chunker_answer: str
    # summarizer_answer: str
    draft_summary: str
    critique: str
    final_summary: str
    min_length: str
    max_length: str
    chunk_size: int
    chunks: list[str]
    filtered_chunks: list[str]
    chunk_summaries: list[str]
    document: str
    doc_num: str


def strip_qwen_thinking(text: str) -> str:
    """
    Qwen3 can emit <think>...</think> blocks.
    For this beginner pipeline, remove them from the visible output.
    """
    # text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    return text


def build_llm() -> tuple[HuggingFacePipeline, AutoTokenizer]:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype="auto",
        device_map="auto",
    )

    text_summarization_pipeline = pipeline(
        task="text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=1024,
        temperature=0.3,
        top_p=0.9,
        do_sample=True,
        return_full_text=False,
        pad_token_id=tokenizer.eos_token_id,
    )

    llm = HuggingFacePipeline(pipeline=text_summarization_pipeline)
    return llm, tokenizer


def format_qwen_prompt(tokenizer: AutoTokenizer, system_prompt: str, user_prompt: str) -> str:
    """
    Qwen3 supports enable_thinking=False in its chat template. The try/except keeps
    this file usable even if your installed Transformers/tokenizer version does not
    expose that argument yet.
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )


def make_qwen_caller(llm: HuggingFacePipeline, tokenizer: AutoTokenizer):

    def qwen_model(system_prompt: str, user_prompt: str):
        prompt = format_qwen_prompt(tokenizer, system_prompt, user_prompt)
        output = llm.invoke(prompt)
        return strip_qwen_thinking(str(output))

    return qwen_model

llm, tokenizer = build_llm()
call_qwen = make_qwen_caller(llm, tokenizer)

def chunk_text(text, n):
    chunked_text = []
    len_text = len(text.split())
    while len_text > 0:
        if len_text < n:
            chunked_text.append(text)
            break
        n_split = text.split(maxsplit=n)
        chunk, text = " ".join(n_split[:n]), n_split[n]
        chunked_text.append(chunk)
        len_text = len(text.split())
    return chunked_text