from libzim.reader import Archive
from bs4 import BeautifulSoup
import time
import json


def extract_chunks(entry_title, soup):
    content_div = soup.find("div", {"id": "mw-content-text"})
    if not content_div:
        return []

    # remove noice
    for tag in content_div(["style", "script", "table"]):
        tag.decompose()

    chunks = []
    current_section = "Introduction"
    current_text = ""
    current_links = set()

    for tag in content_div.find_all(["h2", "h3", "p"]):

        if tag.name in ["h2", "h3"]:
            if current_text.strip():
                chunks.append({
                    "title": entry_title,
                    "section": current_section,
                    "content": current_text.strip(),
                    "links": list(current_links)
                })

            current_section = tag.get_text(strip=True)
            current_text = ""
            current_links = set()

        elif tag.name == "p":
            text = tag.get_text(" ", strip=True)
            current_text += " " + text

            # extract links to other articles
            for a in tag.find_all("a"):
                href = a.get("href")
                if href and ":" not in href and not href.startswith("#"):
                    current_links.add(href)

    # save last chunk
    if current_text.strip():
        chunks.append({
            "title": entry_title,
            "section": current_section,
            "content": current_text.strip(),
            "links": list(current_links)
        })

    return chunks


# Demo script for mathematics-wiki
ZIM_FILE = "./data/wikipedia_en_mathematics_nopic_2026-03.zim"

zim = Archive(ZIM_FILE)

all_chunks = []

start_time = time.time()
total_entries = zim.article_count
for i in range(zim.article_count):
    try:
        entry = zim._get_entry_by_id(i)

        html = bytes(entry.get_item().content).decode("utf-8", errors="ignore")
        soup = BeautifulSoup(html, "html.parser")

        # skip redirects
        if soup.find("meta", {"http-equiv": "refresh"}):
            continue

        chunks = extract_chunks(entry.title, soup)

        for idx, c in enumerate(chunks):
            all_chunks.append({
                "id": f"{i}_{idx}",
                "title": c["title"],
                "section": c["section"],
                "content": c["content"],
                "links": c["links"]
            })

        if i % 10 == 0:
            print(f"Processed {i:6d} entries of {total_entries:6d}, total chunks: {len(all_chunks):4d}, time: {(time.time() - start_time):5.2f}s")

    except Exception as e:
        print(f"Error at entry {i}: {e}")
        continue


# Save to JSON

with open("chunks.json", "w", encoding="utf-8") as f:
    json.dump(all_chunks, f, ensure_ascii=False, indent=2)

print(f"\nDone. Saved {len(all_chunks)} chunks to chunks.json in {(time.time() - start_time):.2f}s")
