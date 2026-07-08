"""STAGE 2 - Synthetic Factory Metadata.

Every image gets a believable factory context (temp, supplier, shift, ...)
generated DETERMINISTICALLY from its filename, so the same file always
produces the same metadata (reproducible demo).

This is a SIMULATED data layer - label it as such in the writeup. It is
NOT real sensor telemetry.

Contract: make_metadata(filename: str) -> dict
"""
from __future__ import annotations
import random
import hashlib


def make_metadata(filename: str) -> dict:
    seed = int(hashlib.md5(filename.encode()).hexdigest(), 16) % (2 ** 32)
    r = random.Random(seed)
    return {
        "BatchID": f"B{seed % 1000000:06d}",
        "Machine": r.choice(["M1", "M2", "M3", "M4"]),
        "Operator": r.choice(["Op-A", "Op-B", "Op-C"]),
        "Temperature": round(r.uniform(20, 80), 1),        # deg C
        "Humidity": round(r.uniform(30, 90), 1),           # %
        "Shift": r.choice(["A", "B", "C"]),
        "Material": r.choice(["Steel-58HRC", "Al-40HRC", "Ti-55HRC"]),
        "Pressure": round(r.uniform(1.0, 6.0), 2),         # bar
        "Supplier": r.choice(["Sup-A", "Sup-B", "Sup-C"]),
        "MachineAge": r.randint(1, 15),                    # years
        "LubricationHours": r.randint(0, 500),
    }
