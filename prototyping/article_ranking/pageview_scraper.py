import gzip
from bs4 import BeautifulSoup
import requests

"""
Base-URL to directory with all pageview files (one for each hour)
in April 2026 --> can later be extended to scrape data for multiple
month
"""
base = "https://dumps.wikimedia.org/other/pageviews/2026/2026-04/"
pageview_dict = {}

response = requests.get(base)
soup = BeautifulSoup(response.text, "html.parser")

for a in soup.find_all('a', href=True):
    ref = a['href']

    if not ref.endswith(".gz"):
        continue
    file_url = base + ref
    print(f"Processing: {ref}")

    try:
        with requests.get(file_url, stream=True) as req:
            req.raw.decode_content = True

            with gzip.open(req.raw, "rt") as f:
                for line in f:
                    project, title, views, _ = line.strip().split()

                    if not project.startswith("en"):
                        continue

                    views = int(views)

                    if title not in pageview_dict:
                        pageview_dict[title] = views
                    else:
                        pageview_dict[title] += views

    except Exception as e:
        print(f"Error with {ref}: {e}")