"""STAGE 8 (moderator) + STAGE 13 - Recommendation Generator.

Feeds all debate votes to the LOCAL LLM (Qwythos) with a moderator prompt
and returns the winning root cause, a rationale, and concrete actions.

Contract:
  moderate(votes, defect=None, metadata=None, kg_info=None, use_llm=True)
      -> {"winning_cause", "rationale", "actions":[str], "conf", "votes"}
"""
from __future__ import annotations

from agents.llm import chat, parse_json, available
from agents import prompts


def _fallback_moderate(votes, kg_info):
    best = max(votes, key=lambda v: v["conf"])
    fix = (kg_info or {}).get("fix", "")
    return {
        "winning_cause": best["cause"],
        "rationale": (f"Selected the highest-confidence specialist hypothesis "
                      f"({best['role']}, conf {best['conf']}). LLM unavailable — "
                      f"using knowledge-graph fallback."),
        "actions": [fix] if fix else ["Review process parameters with the line owner."],
        "conf": best["conf"],
        "votes": votes,
    }


def moderate(votes: list, defect=None, metadata=None, kg_info=None,
             use_llm: bool = True) -> dict:
    if use_llm and available():
        system, user = prompts.moderator_prompt(votes, defect, metadata, kg_info)
        raw = chat(system, user)
        data = parse_json(raw, default={})
        if data:
            winning = data.get("winning_cause") or max(votes, key=lambda v: v["conf"])["cause"]
            rationale = data.get("rationale") or "Moderator synthesized the debate."
            actions = data.get("actions") or [(kg_info or {}).get("fix", "")]
            try:
                conf = float(data.get("conf", max(v["conf"] for v in votes)))
            except Exception:
                conf = max(v["conf"] for v in votes)
            return {
                "winning_cause": winning,
                "rationale": rationale,
                "actions": [a for a in actions if a],
                "conf": round(conf, 2),
                "votes": votes,
            }
    return _fallback_moderate(votes, kg_info)
