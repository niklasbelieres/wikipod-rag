from bs4 import BeautifulSoup, NavigableString, Tag
from src.wikipod.chunking.models import Section
import re


def _get_article_root(html: str):
    soup = BeautifulSoup(html, "html.parser")
    return soup.find("div", class_="mw-parser-output")


def _clean_article_root(html: str):
    root = _get_article_root(html)

    if root is None:
        return None

    # Remove irrelevant elements
    for tag in root.find_all(["style", "script"]):
        tag.decompose()

    for selector in [
        "table.sidebar",
        "div.zim-footer",
        "table.infobox",
        "table.navbox",
    ]:
        for element in root.select(selector):
            element.decompose()

    return root


def _clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\s+([.,;:!?])", r"\1", text)
    text = re.sub(r"\(\s+", "(", text)
    text = re.sub(r"\s+\)", ")", text)
    return text


def extract_article_text(html: str) -> str:
    root = _clean_article_root(html)

    if root is None:
        return ""

    text = root.get_text(separator=" ", strip=True)
    return _clean_text(text)


def extract_sections(html: str) -> list[Section]:
    root = _clean_article_root(html)

    if root is None:
        return []

    sections: list[Section] = []

    current_title = "Introduction"
    current_content: list[str] = []

    # only iterate over direct children of the article
    for element in root.find_all(recursive=False):

        # new section starts
        if (
            isinstance(element, Tag)
            and element.name in {"h2", "h3"}
        ):
            text = _clean_text(" ".join(current_content))

            if text:
                sections.append(
                    Section(
                        title=current_title,
                        text=text,
                    )
                )

            current_title = element.get_text(" ", strip=True)
            current_content = []

        # paragraphs
        elif (
            isinstance(element, Tag)
            and element.name == "p"
        ):
            paragraph = element.get_text(" ", strip=True)

            if paragraph:
                current_content.append(paragraph)

        # lists (optional)
        elif (
            isinstance(element, Tag)
            and element.name in {"ul", "ol"}
        ):
            txt = element.get_text(" ", strip=True)

            if txt:
                current_content.append(txt)

    # last section
    text = _clean_text(" ".join(current_content))

    if text:
        sections.append(
            Section(
                title=current_title,
                text=text,
            )
        )

    return sections