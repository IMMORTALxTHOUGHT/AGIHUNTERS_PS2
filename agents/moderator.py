"""STAGE 8 (moderator) + STAGE 13 - Recommendation Generator.

Real algorithm: feed all debate votes to the LOCAL LLM (Qwythos) with a
moderator prompt; ask for the winning cause, a rationale, and 1-3 concrete
maintenance actions referencing the failure chain + machine/supplier.

Contract:
  moderate(votes, ctx=None)
      -> {"winning_cause", "rationale", "votes", "actions":[str]}
"""
from __future__ import annotations


def moderate(votes: list, ctx=None) -> dict:
    # ---- SCAFFOLD DUMMY (replace with Qwythos moderator call) ----
    winning = max(votes, key=lambda v: v["conf"])
    return {
        "winning_cause": winning["cause"],
        "rationale": "Moderator selected the highest-confidence agent vote (scaffold).",
        "votes": votes,
        "actions": ["Inspect Heat Furnace 4", "Check Cooling Line B"],
    }
    # TODO: call Qwythos with all votes + failure chains + metadata,
    #       return parsed {winning_cause, rationale, actions}.
