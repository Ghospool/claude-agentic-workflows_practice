"""Pluggable knowledge sources. The agent only sees SearchResult/Document, never the backend."""
from __future__ import annotations

import html.parser
import ipaddress
import math
import re
import socket
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Protocol

_STOP = set("a an the of to in on for and or is are was were be by with as at from that this it its what which how "
            "should we our their do does need needs".split())


def tokens(text: str) -> list[str]:
    return [w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in _STOP and len(w) > 1]


@dataclass
class SearchResult:
    doc_id: str
    title: str
    snippet: str
    score: float = 0.0


@dataclass
class Document:
    doc_id: str
    title: str
    text: str
    url: str = ""


class SourceProvider(Protocol):
    async def search(self, query: str, limit: int) -> list[SearchResult]: ...
    async def fetch(self, doc_id: str) -> Document: ...


class LocalCorpusProvider:
    """Keyword (BM25-style) search over a folder of .md/.txt files. Zero external dependencies."""

    def __init__(self, folder: str | Path) -> None:
        self.root = Path(folder)
        self.docs: dict[str, Document] = {}
        for p in sorted(self.root.rglob("*")):
            if p.suffix.lower() in {".md", ".txt"} and p.is_file():
                text = p.read_text(encoding="utf-8")
                first = text.strip().splitlines()[0] if text.strip() else p.stem
                rid = p.relative_to(self.root).as_posix()
                self.docs[rid] = Document(rid, first.lstrip("# ").strip() or p.stem, text)
        self._tf = {k: _count(tokens(d.title + " " + d.text)) for k, d in self.docs.items()}
        n = max(1, len(self.docs))
        df: dict[str, int] = {}
        for tf in self._tf.values():
            for t in tf:
                df[t] = df.get(t, 0) + 1
        self._idf = {t: math.log(1 + (n - c + 0.5) / (c + 0.5)) for t, c in df.items()}

    async def search(self, query: str, limit: int = 5) -> list[SearchResult]:
        q = tokens(query)
        out = []
        for k, tf in self._tf.items():
            score = 0.0
            for t in q:
                f = tf.get(t, 0)
                if f:
                    score += self._idf.get(t, 0) * (f * 2.2) / (f + 1.2)
            if score > 0:
                d = self.docs[k]
                out.append(SearchResult(k, d.title, _best_sentence(d.text, q), round(score, 3)))
        return sorted(out, key=lambda r: -r.score)[:limit]

    async def fetch(self, doc_id: str) -> Document:
        if doc_id not in self.docs:                      # no path traversal: only indexed ids are readable
            raise KeyError(f"unknown document id: {doc_id}")
        return self.docs[doc_id]


def _count(words: list[str]) -> dict[str, int]:
    c: dict[str, int] = {}
    for w in words:
        c[w] = c.get(w, 0) + 1
    return c


def _best_sentence(text: str, q: list[str]) -> str:
    sents = [s for s in re.split(r"(?<=[.!?])\s+|\n+", text) if s.strip() and not s.startswith("#")]
    best = max(sents, key=lambda s: len(set(tokens(s)) & set(q)), default="")
    return best.strip()[:220]


class CompositeProvider:
    """Fan out to several providers; doc ids are namespaced 'name::id' so sources stay attributable."""

    def __init__(self, providers: dict[str, SourceProvider]) -> None:
        self.providers = providers

    async def search(self, query: str, limit: int = 5) -> list[SearchResult]:
        import asyncio
        res = await asyncio.gather(*(p.search(query, limit) for p in self.providers.values()))
        merged = [SearchResult(f"{n}::{r.doc_id}", r.title, r.snippet, r.score)
                  for n, rs in zip(self.providers, res) for r in rs]
        return sorted(merged, key=lambda r: -r.score)[:limit]

    async def fetch(self, doc_id: str) -> Document:
        name, _, rid = doc_id.partition("::")
        if name not in self.providers:
            raise KeyError(f"unknown source: {name}")
        d = await self.providers[name].fetch(rid)
        return Document(doc_id, d.title, d.text, d.url)


# ---------------- web access (network code; guard is unit-tested, fetching is not exercised offline) -------------
def is_safe_url(url: str) -> bool:
    """SSRF guard: http(s) only, and the host must not resolve to a private, loopback or link-local address."""
    try:
        u = urllib.parse.urlparse(url)
        if u.scheme not in ("http", "https") or not u.hostname:
            return False
        for info in socket.getaddrinfo(u.hostname, None):
            ip = ipaddress.ip_address(info[4][0])
            if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast:
                return False
        return True
    except (ValueError, OSError):
        return False


class _Text(html.parser.HTMLParser):
    def __init__(self) -> None:
        super().__init__(); self.out: list[str] = []; self._skip = 0

    def handle_starttag(self, tag, attrs):
        self._skip += tag in ("script", "style", "noscript")

    def handle_endtag(self, tag):
        self._skip -= tag in ("script", "style", "noscript") and self._skip > 0

    def handle_data(self, data):
        if not self._skip and data.strip():
            self.out.append(data.strip())


class _SafeRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if not is_safe_url(newurl):
            raise urllib.error.URLError("redirect to disallowed address blocked")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


class WebProvider:
    """Web research. Supply `search_fn(query, limit) -> [{'url','title','snippet'}]` backed by your search API."""

    def __init__(self, search_fn: Callable[[str, int], list[dict]], allowed_domains: set[str] | None = None,
                 max_bytes: int = 1_000_000, timeout_s: float = 15.0) -> None:
        self.search_fn, self.allowed, self.max_bytes, self.timeout = search_fn, allowed_domains, max_bytes, timeout_s

    def _allowed(self, url: str) -> bool:
        host = urllib.parse.urlparse(url).hostname or ""
        return not self.allowed or any(host == d or host.endswith("." + d) for d in self.allowed)

    async def search(self, query: str, limit: int = 5) -> list[SearchResult]:
        import asyncio
        rows = await asyncio.to_thread(self.search_fn, query, limit)
        return [SearchResult(r["url"], r.get("title", r["url"]), r.get("snippet", "")[:220])
                for r in rows if self._allowed(r["url"]) and is_safe_url(r["url"])][:limit]

    async def fetch(self, doc_id: str) -> Document:
        import asyncio
        if not (self._allowed(doc_id) and is_safe_url(doc_id)):
            raise PermissionError("url not permitted")

        def get() -> str:
            op = urllib.request.build_opener(_SafeRedirect)
            with op.open(urllib.request.Request(doc_id, headers={"User-Agent": "research-agent/0.1"}),
                         timeout=self.timeout) as r:
                raw = r.read(self.max_bytes)
            p = _Text(); p.feed(raw.decode("utf-8", "replace"))
            return "\n".join(p.out)

        text = await asyncio.to_thread(get)
        return Document(doc_id, doc_id, text, url=doc_id)
