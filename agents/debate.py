"""STAGE 8 - Multi-Agent Debate (Visual / History / Metadata agents).

Real algorithm: each agent builds a prompt and calls the LOCAL LLM
(Qwythos) at http://localhost:11434/v1 (OpenAI-compatible). Keep
temperature low (0.1-0.3) for stable JSON. Parse {cause, conf}.

Contract:
  run_debate(visual_ctx, history_ctx, metadata, chains)
      -> list[{"agent", "cause", "conf"}]
"""
from __future__ import annotations


def run_debate(visual_ctx, history_ctx, metadata, chains):
    # ---- SCAFFOLD DUMMY (replace with 3 Qwythos calls) ----
    return [
        {"agent": "visual",   "cause": "overheating",       "conf": 0.70},
        {"agent": "history",  "cause": "poor cooling",      "conf": 0.60},
        {"agent": "metadata", "cause": "material fatigue",  "conf": 0.55},
    ]
    # TODO: for each specialist, build prompt + call:
    #   from openai import OpenAI
    #   llm = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
    #   r = llm.chat.completions.create(model="qwythos", messages=[...])
    #   votes.append(parse_json(r.choices[0].message.content))
