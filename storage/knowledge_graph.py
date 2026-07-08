"""STAGE 7 - Knowledge Graph of failure chains.

Nodes: defect -> cause -> component -> failure. Edges seeded by domain
knowledge and enriched on every inspection (self-learning, Stage 12).

Contract:
  KnowledgeGraph.add_edge(a, b)
  KnowledgeGraph.chains_from(defect) -> list[list[str]]
  KnowledgeGraph.render() -> graph object (pyvis for dashboard)
"""
from __future__ import annotations
import networkx as nx


class KnowledgeGraph:
    def __init__(self):
        self.G = nx.DiGraph()
        # seed edges (extend with your domain rules)
        self.G.add_edge("Scratch", "Vibration")
        self.G.add_edge("Vibration", "Bearing Wear")
        self.G.add_edge("Bearing Wear", "Motor Stress")
        self.G.add_edge("Motor Stress", "Heat")
        self.G.add_edge("Heat", "Shutdown")

    def add_edge(self, a: str, b: str) -> None:
        self.G.add_edge(a, b)

    def chains_from(self, defect: str) -> list:
        try:
            return [nx.shortest_path(self.G, defect, t)
                    for t in nx.descendants(self.G, defect)]
        except nx.NetworkXError:
            return []

    def render(self):
        # TODO: build a pyvis network from self.G and return HTML
        return self.G
