"""BM25 (Okapi) retrieval in pure Python.

Implemented directly instead of depending on a library: the knowledge base has fewer than
a few hundred items, the result is deterministic, and the repository installs with fewer
dependencies.
"""
import math
import re
from collections import Counter
from typing import List, Sequence, Tuple

from .schema import KnowledgeItem, Scenario

_TOKEN = re.compile(r"[a-z0-9]+")

STOPWORDS = frozenset(
    "a an and are as at be been but by for from has have in is it its of on or that the "
    "their them they this to was were will with we our you your not no".split()
)


def tokenize(text: str) -> List[str]:
    return [t for t in _TOKEN.findall(text.lower()) if len(t) > 1 and t not in STOPWORDS]


def scenario_query(s: Scenario) -> str:
    return " ".join([s.narrative] + list(s.pain_points) + list(s.objections))


class BM25Retriever:
    def __init__(self, items: Sequence[KnowledgeItem], k1: float = 1.5, b: float = 0.75):
        self.items = list(items)
        self.k1 = k1
        self.b = b
        self._docs = [tokenize(i.text) for i in self.items]
        self._tf = [Counter(d) for d in self._docs]
        self._n = len(self._docs)
        self._avgdl = (sum(len(d) for d in self._docs) / self._n) if self._n else 0.0
        df: Counter = Counter()
        for d in self._docs:
            df.update(set(d))
        self._idf = {t: math.log(1.0 + (self._n - n + 0.5) / (n + 0.5)) for t, n in df.items()}

    def __len__(self) -> int:
        return self._n

    def scores(self, query: str) -> List[float]:
        terms = sorted(set(tokenize(query)))
        out = []
        for i in range(self._n):
            tf = self._tf[i]
            dl = len(self._docs[i])
            norm = self.k1 * (1.0 - self.b + self.b * (dl / self._avgdl if self._avgdl else 0.0))
            s = 0.0
            for t in terms:
                f = tf.get(t, 0)
                if f:
                    s += self._idf[t] * f * (self.k1 + 1.0) / (f + norm)
            out.append(s)
        return out

    def search(self, query: str, k: int = 5) -> List[Tuple[KnowledgeItem, float]]:
        """Top-k items with a positive score. Ties are broken by id so the order is stable."""
        if not self._n:
            return []
        scores = self.scores(query)
        order = sorted(range(self._n), key=lambda i: (-scores[i], self.items[i].id))
        return [(self.items[i], scores[i]) for i in order[:k] if scores[i] > 0.0]
