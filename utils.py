from __future__ import annotations

from typing import TypedDict

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_huggingface import HuggingFacePipeline

from agents import * 
from data import *

MODEL_ID = "Qwen/Qwen3-1.7B"
min_length = 400
max_length = 600

ds = load_govreport()

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
        max_new_tokens=512,
        temperature=0.3,
        top_p=0.9,
        do_sample=True,
        return_full_text=False,
        pad_token_id=tokenizer.eos_token_id,
    )

    llm = HuggingFacePipeline(pipeline=text_summarization_pipeline)
    return llm, tokenizer, text_summarization_pipeline


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
        return str(output)

    return qwen_model

def make_qwen_batch_caller(text_generation_pipeline, tokenizer):
    def call_qwen_batch(system_prompt: str,
                        user_prompts: list[str],
                        batch_size: int = 2,
                        max_new_tokens: int = 8,
                        do_sample: bool = False,) -> list[str]:
        prompts = [format_qwen_prompt(tokenizer, system_prompt, user_prompt,) for user_prompt in user_prompts]

        outputs = text_generation_pipeline(prompts,
                                            batch_size=batch_size,
                                            max_new_tokens=max_new_tokens,
                                            do_sample=do_sample,
                                            return_full_text=False,)

        return [output[0]["generated_text"] for output in outputs]

    return call_qwen_batch