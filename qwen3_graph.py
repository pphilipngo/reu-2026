"""
pip install -U langchain langgraph transformers accelerate torch sentencepiece python qwen3_langchain_langgraph_no_pipeline.py
python3 qwen3_graph.py
Optional model cache location: export HF_HOME=/path/to/your/model/cache
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph
from agents import * 
from utils import *
from data import *

import os
import csv
import regex as re
from pathlib import Path
import sys
from summary_rewards import *



MODEL_ID = "Qwen/Qwen3-1.7B"

llm, tokenizer, text_summarization_pipeline = build_llm()

call_qwen = make_qwen_caller(
    llm,
    tokenizer,
)

call_qwen_batch = make_qwen_batch_caller(
    text_summarization_pipeline,
    tokenizer,
)

def build_graph():

    graph = StateGraph(GraphState)

    graph.add_node("chunker_answer", chunker_answer)
    graph.add_node("chunk_analyzer_answer", lambda state: chunk_analyzer_answer(state, call_qwen_batch))
    graph.add_node("summarizer_answer", lambda state: summarizer_answer(state, call_qwen_batch))
    graph.add_node("integrator_answer", lambda state: integrator_answer(state, call_qwen))
    graph.add_node("reviewer_answer", lambda state: reviewer_answer(state, call_qwen))
    graph.add_node("refiner_answer", lambda state: refiner_answer(state, call_qwen))

    graph.add_edge(START, "chunker_answer")
    graph.add_edge("chunker_answer", 'chunk_analyzer_answer')
    graph.add_edge("chunk_analyzer_answer", 'summarizer_answer')
    graph.add_edge("summarizer_answer", "integrator_answer")
    graph.add_edge("integrator_answer", "reviewer_answer")
    graph.add_edge("reviewer_answer", "refiner_answer")
    graph.add_edge("refiner_answer", END)

    return graph.compile()


def main():
    print(f"Loading {MODEL_ID}...")
    app = build_graph()
    ds = load_govreport()
    output_rewards = "summaries/output_rewards.csv"    

    with open(output_rewards, 'w', newline='', encoding='utf-8') as output_file:
        output_writer = csv.writer(output_file)

        for i in range(1, 11):
            gov_doc, ref_sum, ds = output_text_and_ref(ds)
            # write_ref(ref_sum, i)


            result = app.invoke({"user_request": "Summarize this document.", 
                                "document": gov_doc,
                                "doc_num": str(i),
                                "draft_summary": "", 
                                "critique": "", 
                                "final_summary": "",
                                "min_length": str(min_length),
                                "max_length": str(max_length),
                                "chunk_size": 300,
                                "chunks": [],
                                "filtered_chunks": [],
                                "chunk_summaries":[],
                                "chunker_answer": "",
                                })
            
            
            output_writer.writerow(["num", "reward"])

            reward_score = reward_function_single(result["final_summary"], ref_sum)
                    
            output_writer.writerow([i, reward_score])

            print(f"======== Finished Document {i} ========")

        # print("\n=== CHUNKER ANSWER ===\n")
        # print(result["chunks"])

        # print("\n=== CHUNKER ANSWER ===\n")
        # print(result["chunker_answer"])

        # print("\n=== CHUNKER ANALYZER ANSWER ===\n")
        # print(result["filtered_chunks"])

        # print("\n=== SUMMARIZER ANSWER ===\n")
        # print(result["chunk_summaries"])

        # print("\n=== DRAFT SUMMARY ===\n")
        # print(result["draft_summary"])

        # print("\n=== CRITIQUE ===\n")
        # print(result["critique"])

        # print("\n=== FINAL SUMMARY ===\n")
        # print(result["final_summary"])


if __name__ == "__main__":
    main()
