"""
Shared HTML-cleaning and extraction helpers.
"""
import re
import sys

from bs4 import BeautifulSoup, Tag

CONTENT_SELECTOR = ("div", {"class": "mw-parser-output"})

# Elements that are never part of the article's actual content.
DROP_SELECTORS = [
    "style",
    "script",
    ".sidebar",
    "table.sidebar",
    ".navbox",
    "table.navbox",
    ".infobox",
    "table.infobox",
    ".metadata",
    ".mw-editsection",
    ".reference",
    ".reflist",
    ".zim-footer",
    "div.zim-footer",
    ".toc",
]


def get_content_root(html: str) -> Tag | None:
    """Parse `html` and return the cleaned `mw-parser-output` div, or None if absent."""
    soup = BeautifulSoup(html, "html.parser")
    root = soup.find(*CONTENT_SELECTOR)
    if root is None:
        return None

    for selector in DROP_SELECTORS:
        for el in root.select(selector):
            el.decompose()

    return root


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def clean_text(text: str) -> str:
    """Normalize whitespace and tidy spacing around punctuation/parentheses."""
    text = normalize_whitespace(text)
    text = re.sub(r"\s+([.,;:!?])", r"\1", text)
    text = re.sub(r"\(\s+", "(", text)
    text = re.sub(r"\s+\)", ")", text)
    return text


def extract_categories(html: str) -> list[str]:
    """Return category names from the page's `#catlinks` box.

    Kiwix/MediaWiki renders categories as links such as
    ``<a href="../Category:Climate_change">Climate change</a>`` in a
    `div#catlinks`, which sits outside `mw-parser-output`, so this parses
    the full document rather than the cleaned content root.
    """
    soup = BeautifulSoup(html, "html.parser")
    catlinks = soup.find(id="catlinks") or soup.find(class_="catlinks")
    if catlinks is None:
        return []

    categories = []
    for anchor in catlinks.find_all("a", href=True):
        if "Category:" not in anchor["href"]:
            continue
        name = normalize_whitespace(anchor.get_text(" ", strip=True))
        if name:
            # Interned: the same category name recurs across huge numbers of
            # articles, so sharing one string object instead of a fresh copy
            # per occurrence matters a lot at full-corpus scale (see extract_links).
            categories.append(sys.intern(name))
    return categories


def extract_links(root: Tag | None) -> list[str]:
    """Return internal (intra-wiki) link targets, skipping external and self links.

    Link targets are interned (`sys.intern`): a small set of popular pages
    (country names, common concepts, dates, ...) gets linked to from a huge
    fraction of all articles, so most of the millions of link strings across
    the full corpus are exact duplicates of each other. Interning makes
    duplicates share one string object in memory instead of each holding its
    own copy -- on the full en.wikipedia corpus this is the difference
    between the corpus's link data fitting in RAM and not, independent of
    (and in addition to) not keeping full article body text around
    (`analysis.metadata.extract_metadata`'s `include_sections=False`).
    """
    if root is None:
        return []
    links = []
    for anchor in root.find_all("a", href=True):
        href = anchor["href"]
        if href.startswith("http") or href.startswith("./"):
            continue
        # "Hydrogen" and "Hydrogen#Applications" point at the same target
        # article. Without stripping the fragment they'd count as two
        # distinct link targets, splitting both interning and the frequency
        # map for no semantic reason
        target = href.split("#", 1)[0]
        if target:
            links.append(sys.intern(target))
    return links


def _iter_content_nodes(tag: Tag):
    """Yield heading/paragraph/list children, transparently unwrapping <section>.

    Parsoid-rendered ZIM HTML (the format modern Kiwix dumps use) wraps each
    heading's content in a `<section data-mw-section-id="...">`, nested per
    subsection, instead of putting headings and paragraphs side by side as
    direct children of `mw-parser-output`. Unwrapping `<section>` here lets the
    rest of `split_into_sections` keep using simple direct-child iteration for
    everything else, so nested lists still can't cause double-counted text.
    """
    for child in tag.find_all(recursive=False):
        if not isinstance(child, Tag):
            continue
        if child.name == "section":
            yield from _iter_content_nodes(child)
        else:
            yield child


def split_into_sections(root: Tag | None) -> list[tuple[str, str]]:
    """Split cleaned content into (section_title, section_text) pairs.

    Text before the first heading is grouped under "Lead". Only direct
    children (after unwrapping `<section>` wrappers, see `_iter_content_nodes`)
    are inspected, so nested headings inside e.g. infoboxes (already stripped)
    can't create spurious sections.
    """
    if root is None:
        return []

    sections: list[tuple[str, str]] = []
    current_title = "Lead"
    buffer: list[str] = []

    def flush() -> None:
        text = clean_text(" ".join(buffer))
        if text:
            sections.append((current_title, text))

    for child in _iter_content_nodes(root):
        if child.name in ("h2", "h3"):
            flush()
            buffer.clear()
            heading = normalize_whitespace(child.get_text(" ", strip=True))
            current_title = heading or "Untitled"
        elif child.name in ("p", "ul", "ol"):
            text = child.get_text(" ", strip=True)
            if text:
                buffer.append(text)

    flush()
    return sections
