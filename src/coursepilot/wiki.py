"""Fetch Wikipedia articles and turn them into citable passages for quizzes.

Uses the public MediaWiki API (no key needed). Wikimedia asks clients to send a descriptive
User-Agent, which this module does. Article text is licensed CC BY-SA 4.0; every passage
keeps its article title, section, and URL so the app can attribute and link it.
"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from coursepilot.chunking import chunk_pages
from coursepilot.documents import Page
from coursepilot.errors import CoursePilotError, ErrorCode

API_URL = "https://en.wikipedia.org/w/api.php"
USER_AGENT = "CoursePilot/0.1 (https://github.com/RettWilson22/coursepilot)"
# Sections that list references or links rather than explain the topic.
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
}

FetchJSON = Callable[[dict[str, str]], dict[str, Any]]


@dataclass(frozen=True)
class Passage:
    """A piece of source text a quiz question can cite."""

    citation: str  # e.g. "Wikipedia: Super Mario Galaxy § Gameplay"
    text: str
    url: str | None = None


@dataclass(frozen=True)
class Article:
    title: str
    url: str
    text: str  # plain text with "== Section ==" headings

    def sections(self) -> list[tuple[str, str]]:
        """(section name, body) pairs; the introduction comes first with an empty name."""
        parts = re.split(r"^(==+) ?(.+?) ?\1\s*$", self.text, flags=re.M)
        sections = [("", parts[0])]
        for i in range(1, len(parts) - 2, 3):
            sections.append((parts[i + 1].strip(), parts[i + 2]))
        return [(name, body.strip()) for name, body in sections if body.strip()]


def _fetch_json(params: dict[str, str]) -> dict[str, Any]:
    query = urllib.parse.urlencode({**params, "format": "json", "formatversion": "2"})
    request = urllib.request.Request(f"{API_URL}?{query}", headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=15) as response:
            data: dict[str, Any] = json.load(response)
            return data
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
        raise CoursePilotError(
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
        raise CoursePilotError(
            ErrorCode.SOURCE_NOT_FOUND,
            f'Couldn\'t find a Wikipedia article about "{topic}". Check the spelling or try a '
            "more specific topic.",
            details={"topic": topic},
        )


def article_passages(article: Article, limit: int = 10, max_words: int = 180) -> list[Passage]:
    """Passages covering the whole article: the introduction first, then one per section in
    turn, so a quiz isn't drawn only from the opening paragraphs."""
    per_section: list[list[Passage]] = []
    for number, (name, body) in enumerate(article.sections(), start=1):
        if name.lower() in SKIP_SECTIONS:
            continue
        citation = f"Wikipedia: {article.title}" + (f" § {name}" if name else "")
        url = article.url + ("#" + urllib.parse.quote(name.replace(" ", "_")) if name else "")
        chunks = chunk_pages([Page(article.title, number, body)], max_words=max_words)
        per_section.append([Passage(citation, c.text, url) for c in chunks])

    picked: list[Passage] = []
    depth = 0
    while len(picked) < limit and any(depth < len(s) for s in per_section):
        for section in per_section:
            if depth < len(section) and len(picked) < limit:
                picked.append(section[depth])
        depth += 1
    return picked
