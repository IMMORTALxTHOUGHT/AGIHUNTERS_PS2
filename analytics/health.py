"""STAGE 11 - Factory Health Score (aggregate risk).

Contract: health(records: list[{"meta":dict,"is_defect":bool}]) -> dict
"""
from __future__ import annotations
from collections import defaultdict


def health(records: list) -> dict:
    g = defaultdict(list)
    for r in records:
        g[r["meta"]["Machine"]].append(1 if r["is_defect"] else 0)
    by_machine = {m: 1 - sum(v) / len(v) for m, v in g.items()}
    factory = 100.0 * sum(by_machine.values()) / len(by_machine) if by_machine else 100.0
    return {"factory_pct": round(factory, 2), "by_machine": by_machine}
