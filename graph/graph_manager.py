import os
from dotenv import load_dotenv

load_dotenv()                           # reads .env into process-env
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")


# graph/graph_manager.py

class GraphManager:
    def __init__(self):
        self._next_id = 1
        self.nodes = {}   # node_id → {"label": ..., "properties": {...}}
        self.edges = []   # list of (src_id, tgt_id, relation)

    def _gen_id(self) -> str:
        nid = str(self._next_id)
        self._next_id += 1
        return nid

    def add_claim(self, claim: str) -> str:
        node_id = self._gen_id()
        self.nodes[node_id] = {
            "label": "Claim",
            "properties": {"text": claim}
        }
        return node_id

    def add_document(self, url: str, text: str) -> str:
        node_id = self._gen_id()
        self.nodes[node_id] = {
            "label": "Document",
            "properties": {"url": url, "text": text}
        }
        return node_id

    def add_snippet(self, summary: str, label: str, score: float) -> str:
        node_id = self._gen_id()
        self.nodes[node_id] = {
            "label": "Snippet",
            "properties": {"summary": summary, "label": label, "score": score}
        }
        return node_id

    def add_edge(self, src_id: str, tgt_id: str, relation: str):
        self.edges.append((src_id, tgt_id, relation))

    def stats(self) -> dict:
        return {"nodes": len(self.nodes), "edges": len(self.edges)}

    def get_top_snippets(self, claim_id: str, relation: str, k: int = 3):
        # find all snippet-IDs linked by the given relation
        snippet_ids = [
            tgt for src, tgt, rel in self.edges
            if src == claim_id and rel == relation
        ]
        # collect (summary, score)
        snippets = []
        for sid in snippet_ids:
            node = self.nodes.get(sid)
            if node and node["label"] == "Snippet":
                props = node["properties"]
                snippets.append((props["summary"], props["score"]))
        snippets.sort(key=lambda x: x[1], reverse=True)
        return snippets[:k]
