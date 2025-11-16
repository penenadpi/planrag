"""
Merged RAG pipeline for retrieving RDF subjects (entities/classes)
AND predicate/property URIs (attributes) depending on relevance.

FAISS-free version – uses sklearn NearestNeighbors only.
"""

from typing import List, Dict, Tuple
import rdflib
from rdflib import Graph, URIRef, Literal
import numpy as np

from sklearn.neighbors import NearestNeighbors
from sentence_transformers import SentenceTransformer, CrossEncoder


# ==========================================================
# Load RDF
# ==========================================================
def load_rdf_graph(paths: List[str]) -> Graph:
    g = Graph()
    for p in paths:
        fmt = None
        if p.endswith(".ttl"):
            fmt = "turtle"
        elif p.endswith(".rdf") or p.endswith(".xml"):
            fmt = "xml"
        elif p.endswith(".nt"):
            fmt = "nt"
        g.parse(p, format=fmt)
        print(f"Loaded {p}, triples so far: {len(g)}")
    return g


# ==========================================================
# Utilities
# ==========================================================
def short_uri(node):
    s = str(node)
    return s.rsplit("#", 1)[-1].rsplit("/", 1)[-1]


def get_label(g, uri):
    preds = [
        rdflib.RDFS.label,
        rdflib.URIRef("http://www.w3.org/2004/02/skos/core#prefLabel"),
        rdflib.URIRef("http://xmlns.com/foaf/0.1/name"),
        rdflib.URIRef("http://purl.org/dc/terms/title"),
    ]
    for p in preds:
        for o in g.objects(uri, p):
            if isinstance(o, Literal):
                return str(o)
    return None


# ==========================================================
# SUBJECT-based chunks (entities)
# ==========================================================
def subject_chunks(g: Graph) -> List[Dict]:
    chunks = []
    subjects = set(g.subjects())

    for s in subjects:
        triples = list(g.triples((s, None, None)))
        if not triples:
            continue

        lines = []
        for (sub, pred, obj) in triples:
            p_lbl = get_label(g, pred) or short_uri(pred)
            if isinstance(obj, URIRef):
                o_lbl = get_label(g, obj) or short_uri(obj)
            else:
                o_lbl = str(obj)
            lines.append(f"{p_lbl}: {o_lbl}")

        chunks.append({
            "chunk_type": "subject",
            "uri": str(s),
            "text": " ; ".join(lines),
            "triples": triples
        })

    print(f"Created {len(chunks)} SUBJECT chunks.")
    return chunks


# ==========================================================
# PREDICATE-based chunks (attributes)
# ==========================================================
def predicate_chunks(g: Graph) -> List[Dict]:
    chunks = []
    predicates = set(g.predicates())

    for p in predicates:
        label = get_label(g, p)
        fallback = short_uri(p)
        p_text = label if label else fallback

        # Example text: "predicate title describes properties of resources"
        text = f"Attribute/Property: {p_text}. URI: {p}"

        chunks.append({
            "chunk_type": "predicate",
            "uri": str(p),
            "text": text,
            "triples": []
        })

    print(f"Created {len(chunks)} PREDICATE chunks.")
    return chunks


# ==========================================================
# Combined chunking
# ==========================================================
def build_all_chunks(g: Graph):
    s_chunks = subject_chunks(g)
    p_chunks = predicate_chunks(g)
    all_chunks = s_chunks + p_chunks
    print(f"Total chunks: {len(all_chunks)}")
    return all_chunks


# ==========================================================
# Retriever (bi-encoder + NearestNeighbors)
# ==========================================================
class RetrieverIndex:
    def __init__(self, model="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model)
        self.texts = []
        self.uris = []
        self.types = []
        self.emb = None
        self.nn = None
        self.chunks = []

    def add_chunks(self, chunks):
        self.chunks = chunks
        self.texts = [c["text"] for c in chunks]
        self.uris = [c["uri"] for c in chunks]
        self.types = [c["chunk_type"] for c in chunks]

        print("Encoding chunk embeddings...")
        self.emb = np.array(
            self.model.encode(self.texts, convert_to_numpy=True, show_progress_bar=True)
        )

        print("Building sklearn NearestNeighbors index…")
        self.nn = NearestNeighbors(n_neighbors=10, metric="cosine")
        self.nn.fit(self.emb)

    def query(self, query_text, k=10):
        q_emb = np.array(self.model.encode([query_text], convert_to_numpy=True))
        dist, idxs = self.nn.kneighbors(q_emb, n_neighbors=k)

        dist = dist[0]
        idxs = idxs[0]
        scores = 1 - dist

        return [(idx, float(scores[i])) for i, idx in enumerate(idxs)]


# ==========================================================
# Cross-Encoder reranking
# ==========================================================
class Reranker:
    def __init__(self, model="cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.cross = CrossEncoder(model)

    def rerank(self, query, chunks_idx, chunk_list, top_m=5):
        pairs = [(query, chunk_list[idx]["text"]) for idx in chunks_idx]
        scores = self.cross.predict(pairs)

        results = list(zip(chunks_idx, scores))
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_m]


# ==========================================================
# Extract final results
# ==========================================================
def collect_subject_properties(g: Graph, uri: str):
    out = []
    for p, o in g.predicate_objects(URIRef(uri)):
        pred = get_label(g, p) or short_uri(p)
        if isinstance(o, URIRef):
            obj = get_label(g, o) or short_uri(o)
        else:
            obj = str(o)
        out.append((pred, obj))
    return out


def extract_final(g, reranked, chunks):
    results = []

    for idx, score in reranked:
        chunk = chunks[idx]
        ctype = chunk["chunk_type"]
        uri = chunk["uri"]

        if ctype == "subject":
            props = collect_subject_properties(g, uri)
        else:
            # predicate chunks have no triples: return label + uri
            lbl = get_label(g, URIRef(uri)) or short_uri(uri)
            props = [("label", lbl)]

        results.append({
            "uri": uri,
            "type": ctype,
            "score": float(score),
            "properties": props
        })

    return results


# ==========================================================
# Pipeline
# ==========================================================
def pipeline_run(paths, query, top_k=15, top_m=5):
    g = load_rdf_graph(paths)
    chunks = build_all_chunks(g)

    retr = RetrieverIndex()
    retr.add_chunks(chunks)

    retrieved = retr.query(query, k=top_k)
    cand_idxs = [idx for idx, _ in retrieved]

    reranker = Reranker()
    reranked = reranker.rerank(query, cand_idxs, chunks, top_m=top_m)

    final = extract_final(g, reranked, chunks)
    return final, reranked, chunks


# ==========================================================
# Example run
# ==========================================================
if __name__ == "__main__":
    files = ["test2.rdf"]
    query1 = "What attributes does a paper have?"
    print("Experiment1: "+query1 + "\n")
    final, reranked, chunks = pipeline_run(files, query1)

    print("\n=== Combined Results (subjects + attributes) ===")
    for r in final:
        print(f"\nURI: {r['uri']}  [type={r['type']}]  score={r['score']:.4f}")
        for p, o in r["properties"]:
            print(f"  - {p}: {o}")


    query2 = "What attributes does author have"
    print("Experiment2: "+query2 + "\n")
    final, reranked, chunks = pipeline_run(files, query2)

    print("\n=== Combined Results (subjects + attributes) ===")
    for r in final:
        print(f"\nURI: {r['uri']}  [type={r['type']}]  score={r['score']:.4f}")
        for p, o in r["properties"]:
            print(f"  - {p}: {o}")



    query3 = "Which scientific areas are available?"
    print("Experiment3: "+query3 + "\n")
    final, reranked, chunks = pipeline_run(files, query3)

    print("\n=== Combined Results (subjects + attributes) ===")
    for r in final:
        print(f"\nURI: {r['uri']}  [type={r['type']}]  score={r['score']:.4f}")
        for p, o in r["properties"]:
            print(f"  - {p}: {o}")
