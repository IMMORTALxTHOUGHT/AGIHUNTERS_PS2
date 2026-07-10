"""STAGE 8 - Multi-Agent Debate for Root-Cause Analysis.

Three specialist agents (Process / Materials / Reliability) each propose a
root-cause hypothesis grounded in the defect, factory metadata, similar past
cases, and the Knowledge Graph. Returns structured votes.

Contract:
  run_debate(defect, metadata, similar_cases, kg_info, use_llm=True)
      -> list[{"agent", "role", "cause", "reasoning", "conf"}]
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

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


def _vote_for(agent, system, user, defect_label, kg_info, use_llm):
    """One specialist's vote. Stateless (single HTTP call), so it is safe to
    run concurrently across agents/groups."""
    if use_llm and available():
        raw = chat(system, user)
        data = parse_json(raw, default={})
        cause = data.get("cause") or f"{defect_label} (undetermined)"
        reasoning = data.get("reasoning") or (raw or "")[:280]
        try:
            conf = float(data.get("conf", 0.5))
        except Exception:
            conf = 0.5
    else:
        cause, reasoning, conf = _fallback(agent, defect_label, kg_info)
    return {
        "agent": agent,
        "role": ROLES[agent],
        "cause": cause,
        "reasoning": reasoning,
        "conf": round(conf, 2),
    }


def run_debate(defect, metadata, similar_cases, kg_info, use_llm: bool = True) -> list:
    prompts_list = prompts.build_debate_prompts(
        defect, metadata,
        [c.get("label") for c in (similar_cases or [])[:3]], kg_info)
    # the 3 specialists are independent -> run them in parallel
    with ThreadPoolExecutor(max_workers=3) as ex:
        votes = list(ex.map(
            lambda p: _vote_for(p[0], p[1], p[2], defect, kg_info, use_llm),
            prompts_list))
    return votes


def run_group_debate(defect_type, summary, members, kg_info, use_llm: bool = True) -> list:
    """Multi-agent debate for a BATCH GROUP (many parts, one shared defect type).
    Reuses the same three specialists but feeds them the aggregated group
    context so they synthesise a common root cause instead of analysing a
    single part. The 3 agents run in parallel. Returns vote structure like
    run_debate()."""
    prompts_list = prompts.build_group_debate_prompts(
        defect_type, summary, members, kg_info)
    with ThreadPoolExecutor(max_workers=3) as ex:
        votes = list(ex.map(
            lambda p: _vote_for(p[0], p[1], p[2], defect_type, kg_info, use_llm),
            prompts_list))
    return votes
