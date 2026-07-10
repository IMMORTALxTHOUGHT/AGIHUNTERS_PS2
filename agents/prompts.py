"""LLM prompt templates for the multi-agent Root-Cause Analysis debate.

Each specialist agent gets the defect, factory metadata, top similar past
cases, and the Knowledge Graph's learned associations + curated fix, so its
hypothesis is grounded in data rather than free-form guesses.

All agent prompts ask for JSON:
    {"cause": str, "reasoning": str, "conf": float 0..1}
The moderator prompt asks for:
    {"winning_cause": str, "rationale": str, "actions": [str], "conf": float}
"""
from __future__ import annotations

import json

SYSTEMS = {
    "process": (
        "You are a senior process engineer in a heavy-industry forge. You "
        "diagnose defects from process parameters (temperature, pressure, "
        "cooling, line speed). Be concise, mechanistic, and cite the numbers."
    ),
    "materials": (
        "You are a materials scientist in a heavy-industry forge. You diagnose "
        "defects from material properties, supplier quality, lubrication, and "
        "equipment wear. Be concise and mechanistic."
    ),
    "reliability": (
        "You are a reliability & quality engineer. You diagnose defects from "
        "systemic patterns: machine, operator, shift, and historical recurrence. "
        "Think about what tends to recur and why. Be concise."
    ),
}

_MODERATOR_SYSTEM = (
    "You are the lead engineer adjudicating competing root-cause hypotheses "
    "from three specialists. Pick the most defensible cause, explain why it "
    "beats the others, and give 2-3 concrete maintenance/process actions. "
    "Reference the factory conditions and the learned knowledge graph."
)


def _fmt(v) -> str:
    return json.dumps(v, ensure_ascii=False)


def build_debate_prompts(defect, metadata, similar_labels, kg_info) -> list:
    meta = metadata or {}
    causes = kg_info.get("causes", []) if kg_info else []
    fix = kg_info.get("fix", "") if kg_info else ""
    causes_str = "; ".join(f"{c['condition']} (seen {c['count']}x)" for c in causes) or "none yet"
    similar_str = ", ".join(str(s) for s in (similar_labels or [])) or "none"

    process_ctx = {
        "defect": defect,
        "Temperature": meta.get("Temperature"), "Pressure": meta.get("Pressure"),
        "Humidity": meta.get("Humidity"), "Shift": meta.get("Shift"),
        "similar_cases": similar_labels, "kg_associated": causes_str,
        "kg_fix": fix,
    }
    materials_ctx = {
        "defect": defect,
        "Material": meta.get("Material"), "Supplier": meta.get("Supplier"),
        "LubricationHours": meta.get("LubricationHours"),
        "MachineAge": meta.get("MachineAge"), "similar_cases": similar_labels,
        "kg_associated": causes_str, "kg_fix": fix,
    }
    reliability_ctx = {
        "defect": defect,
        "Machine": meta.get("Machine"), "Operator": meta.get("Operator"),
        "Shift": meta.get("Shift"), "similar_cases": similar_labels,
        "kg_associated": causes_str, "kg_fix": fix,
    }

    base = (
        f"\nDefect under analysis: {defect}. Respond ONLY with JSON: "
        '{"cause": str, "reasoning": str, "conf": float 0..1}.'
    )

    return [
        ("process", SYSTEMS["process"],
         "Process context: " + _fmt(process_ctx) + base),
        ("materials", SYSTEMS["materials"],
         "Materials context: " + _fmt(materials_ctx) + base),
        ("reliability", SYSTEMS["reliability"],
         "Reliability context: " + _fmt(reliability_ctx) + base),
    ]


def moderator_prompt(votes, defect, metadata, kg_info) -> tuple:
    ctx = {
        "defect": defect,
        "metadata": metadata or {},
        "kg_associated": [c["condition"] for c in (kg_info or {}).get("causes", [])],
        "kg_fix": (kg_info or {}).get("fix", ""),
    }
    user = (
        "Adjudicate these specialist votes:\n"
        + _fmt(votes)
        + "\nContext: " + _fmt(ctx)
        + '\nRespond ONLY with JSON: {"winning_cause": str, "rationale": str, '
          '"actions": [str], "conf": float 0..1}.'
    )
    return _MODERATOR_SYSTEM, user
