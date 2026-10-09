"""Fetch Wikipedia articles and turn them into citable passages for quizzes.

Uses the public MediaWiki API (no key needed). Wikimedia asks clients to send a descriptive
User-Agent, which this module does. Article text is licensed CC BY-SA 4.0; every passage
keeps its article title, section, and URL so the app can attribute and link it.
"""

from __future__ import annotations

import http.client
import json
import re
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from backofthebook.chunking import chunk_pages
from backofthebook.documents import Page
from backofthebook.errors import BackOfTheBookError, ErrorCode

API_URL = "https://en.wikipedia.org/w/api.php"
USER_AGENT = "BackOfTheBook/0.1 (https://github.com/RettWilson22/back-of-the-book)"
# Sections that list references or links rather than explain the topic. Their subsections
# are left out too.
SKIP_SECTIONS = {
    "notes",
    "references",
    "citations",
    "bibliography",
    "sources",
    "external links",
    "see also",
    "further reading",
    "footnotes",
    "works cited",
    "notes and references",
    "references and notes",
    "general references",
    "general and cited references",
    "cited sources",
    "explanatory notes",
}
# How many passages to take from the introduction, at most, before the other sections.
LEAD_PASSAGES = 3

FetchJSON = Callable[[dict[str, str]], dict[str, Any]]


@dataclass(frozen=True)
class Passage:
    """A piece of source text a quiz question can cite."""

    citation: str  # e.g. "Wikipedia: Super Mario Galaxy § Gameplay"
    text: str
    url: str | None = None
    source: str | None = None  # the file it came from, for passages from course materials


@dataclass(frozen=True)
class Article:
    title: str
    url: str
    text: str  # plain text with "== Section ==" headings

    def sections(self) -> list[tuple[str, str]]:
        """(section name, body) pairs that explain the topic; the introduction comes first with
        an empty name. Reference and link sections are left out with all their subsections."""
        parts = re.split(r"^(==+) ?(.+?) ?\1\s*$", self.text, flags=re.M)
        sections = [("", parts[0])]
        skipping: int | None = None  # heading level of the skipped section we're inside
        for i in range(1, len(parts) - 2, 3):
            level, name = len(parts[i]), parts[i + 1].strip()
            if skipping is not None and level > skipping:
                continue  # a subsection of a skipped section
            skipping = level if name.lower() in SKIP_SECTIONS else None
            if skipping is None:
                sections.append((name, parts[i + 2]))
        return [(name, body.strip()) for name, body in sections if body.strip()]


def _fetch_json(params: dict[str, str]) -> dict[str, Any]:
    query = urllib.parse.urlencode({**params, "format": "json", "formatversion": "2"})
    request = urllib.request.Request(f"{API_URL}?{query}", headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=15) as response:
            data: dict[str, Any] = json.load(response)
            return data
    # OSError covers URLError, timeouts, and dropped connections; HTTPException covers a
    # response cut short; ValueError covers a body that isn't JSON (or isn't UTF-8).
    except (OSError, http.client.HTTPException, ValueError) as e:
        raise BackOfTheBookError(
            ErrorCode.SOURCE_UNAVAILABLE,
            "Couldn't reach Wikipedia to look up the topic; try again in a moment.",
            details={"reason": str(e)},
        ) from e


class WikipediaSource:
    def __init__(self, fetch_json: FetchJSON | None = None) -> None:
        self._fetch_override = fetch_json

    def _fetch(self, params: dict[str, str]) -> dict[str, Any]:
        # Looked up at call time so tests can swap out the network call module-wide.
        return (self._fetch_override or _fetch_json)(params)

    def search(self, topic: str, limit: int = 5) -> list[str]:
        data = self._fetch(
            {"action": "query", "list": "search", "srsearch": topic, "srlimit": str(limit)}
        )
        return [hit["title"] for hit in data.get("query", {}).get("search", [])]

    def article(self, title: str) -> Article | None:
        """The article, or None if it's missing or a disambiguation ("may refer to") page."""
        data = self._fetch(
            {
                "action": "query",
                "prop": "extracts|info|pageprops",
                "explaintext": "1",
                "exsectionformat": "wiki",
                "inprop": "url",
                "ppprop": "disambiguation",
                "redirects": "1",
                "titles": title,
            }
        )
        pages = data.get("query", {}).get("pages", [])
        if not pages:
            return None
        page = pages[0]
        if page.get("missing") or "disambiguation" in page.get("pageprops", {}):
            return None
        text = page.get("extract") or ""
        if not text.strip():
            return None
        return Article(page["title"], page["fullurl"], text)

    def find_article(self, topic: str) -> Article:
        """Best-matching real article for the topic, skipping disambiguation pages."""
        for title in self.search(topic):
            found = self.article(title)
            if found is not None:
                return found
        raise BackOfTheBookError(
            ErrorCode.SOURCE_NOT_FOUND,
            f'Couldn\'t find a Wikipedia article about "{topic}". Check the spelling or try a '
            "more specific topic.",
            details={"topic": topic},
        )


def article_passages(article: Article, limit: int = 10, max_words: int = 180) -> list[Passage]:
    """Up to `limit` passages spread over the whole article, so a quiz isn't drawn only from
    its opening sections.

    A few passages come from the introduction. The rest are spread evenly over the other
    sections, from first to last: one from each, then a second from each, and so on. If the
    article has more sections than that, evenly spaced ones are used. A short article fills
    the remaining places from the rest of its introduction.
    """
    lead: list[Passage] = []
    sections: list[list[Passage]] = []
    for number, (name, body) in enumerate(article.sections(), start=1):
        citation = f"Wikipedia: {article.title}" + (f" § {name}" if name else "")
        url = article.url + ("#" + urllib.parse.quote(name.replace(" ", "_")) if name else "")
        chunks = chunk_pages([Page(article.title, number, body)], max_words=max_words)
        passages = [Passage(citation, c.text, url) for c in chunks]
        if not name:
            lead = passages
        elif passages:
            sections.append(passages)

    from_lead = max(1, min(LEAD_PASSAGES, limit // 3))
    picked = lead[:from_lead]
    room = limit - len(picked)
    if len(sections) > room > 0:  # evenly spaced sections, centered in each stretch
        sections = [sections[(2 * i + 1) * len(sections) // (2 * room)] for i in range(room)]
    depth = 0
    while len(picked) < limit and any(depth < len(s) for s in sections):
        for section in sections:
            if depth < len(section) and len(picked) < limit:
                picked.append(section[depth])
        depth += 1
    return picked + lead[from_lead : from_lead + limit - len(picked)]
