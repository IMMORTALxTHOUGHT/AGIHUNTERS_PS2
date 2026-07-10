"""STAGE 8 - Multi-Agent Debate for Root-Cause Analysis.

Three specialist agents (Process / Materials / Reliability) each propose a
root-cause hypothesis grounded in the defect, factory metadata, similar past
cases, and the Knowledge Graph. Returns structured votes.

Contract:
  run_debate(defect, metadata, similar_cases, kg_info, use_llm=True)
      -> list[{"agent", "role", "cause", "reasoning", "conf"}]
"""
from __future__ import annotations

from agents.llm import chat, parse_json, available
from agents import prompts

ROLES = {
    "process": "Process Engineer",
    "materials": "Materials Scientist",
    "reliability": "Reliability & Quality Engineer",
}


def _fallback(agent: str, defect: str, kg_info: dict) -> tuple:
    fix = (kg_info or {}).get("fix", "Review process parameters.")
    causes = (kg_info or {}).get("causes", [])
    cond = causes[0]["condition"] if causes else "unknown condition"
    reasons = {
        "process": f"Process drift suspected (KG links {defect} to {cond}). {fix}",
        "materials": f"Material/supplier factor suspected (KG links {defect} to {cond}). {fix}",
        "reliability": f"Systemic/equipment factor suspected (KG links {defect} to {cond}). {fix}",
    }
    return (defect + " — " + cond, reasons.get(agent, fix), 0.5)


def run_debate(defect, metadata, similar_cases, kg_info, use_llm: bool = True) -> list:
    similar_labels = [c.get("label") for c in (similar_cases or [])[:3]]
    votes = []
    for agent, system, user in prompts.build_debate_prompts(
        defect, metadata, similar_labels, kg_info
    ):
        if use_llm and available():
            raw = chat(system, user)
            data = parse_json(raw, default={})
            cause = data.get("cause") or f"{defect} (undetermined)"
            reasoning = data.get("reasoning") or (raw or "")[:280]
            try:
                conf = float(data.get("conf", 0.5))
            except Exception:
                conf = 0.5
        else:
            cause, reasoning, conf = _fallback(agent, defect, kg_info)

        votes.append({
            "agent": agent,
            "role": ROLES[agent],
            "cause": cause,
            "reasoning": reasoning,
            "conf": round(conf, 2),
        })
    return votes


def run_group_debate(defect_type, summary, members, kg_info, use_llm: bool = True) -> list:
    """Multi-agent debate for a BATCH GROUP (many parts, one shared defect type).
    Reuses the same three specialists but feeds them the aggregated group
    context so they synthesise a common root cause instead of analysing a
    single part. Returns the same vote structure as run_debate()."""
    votes = []
    for agent, system, user in prompts.build_group_debate_prompts(
        defect_type, summary, members, kg_info
    ):
        if use_llm and available():
            raw = chat(system, user)
            data = parse_json(raw, default={})
            cause = data.get("cause") or f"{defect_type} (undetermined)"
            reasoning = data.get("reasoning") or (raw or "")[:280]
            try:
                conf = float(data.get("conf", 0.5))
            except Exception:
                conf = 0.5
        else:
            cause, reasoning, conf = _fallback(agent, defect_type, kg_info)
        votes.append({
            "agent": agent,
            "role": ROLES[agent],
            "cause": cause,
            "reasoning": reasoning,
            "conf": round(conf, 2),
        })
    return votes
