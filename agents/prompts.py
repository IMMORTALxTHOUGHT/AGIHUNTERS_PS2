"""LLM prompt templates for debate agents and moderator.

Each function returns a system + user prompt pair.
Scaffold — customize wording on the box.
"""
from __future__ import annotations


def visual_agent_prompt(defect: str, heat_summary: str) -> str:
    return (
        "You are a senior computer-vision engineer inspecting a manufacturing defect. "
        f"Defect type: {defect}. Heatmap analysis: {heat_summary}. "
        "Propose the most likely root cause as JSON: {\"cause\": str, \"conf\": float 0-1}."
    )


def history_agent_prompt(past_cases: list[dict]) -> str:
    return (
        "You are a quality historian reviewing similar past failures. "
        f"Top similar cases: {past_cases}. "
        "Propose root cause as JSON: {\"cause\": str, \"conf\": float 0-1}."
    )


def metadata_agent_prompt(metadata: dict) -> str:
    return (
        "You are a process engineer analyzing factory-floor telemetry. "
        f"Metadata: {metadata}. "
        "Propose root cause as JSON: {\"cause\": str, \"conf\": float 0-1}."
    )


def moderator_prompt(votes: list[dict]) -> str:
    return (
        "You are the lead engineer adjudicating root-cause proposals. "
        f"Received votes: {votes}. "
        "Output JSON: {\"winning_cause\": str, \"rationale\": str, \"actions\": [str]}."
    )
