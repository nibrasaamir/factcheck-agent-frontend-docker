import os
from dotenv import load_dotenv
load_dotenv()

import requests
from newspaper import Article
from bs4 import BeautifulSoup
from langchain.tools import Tool

# Browser header to avoid 403 error.
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/605.1.15 (KHTML, like Gecko) Version/15.1 Safari/605.1.15"
    )
}

def _limit_paragraphs(text, max_paras = 6):

    # using only first paragraph from the link.
    # assumption: Paragraphs are assumed to be separated by blank lines.

    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    return "\n\n".join(paras[:max_paras])

def fetch(url):

    # using : 
    # 1. newspaper3k 
    # if newspaper3k doesnt work or returns an error or small paragraph or less then 1 para, we will do to option 2.
    # 2. beautifulSoup
    
    text = ""

    # 1) Newspaper3k
    try:
        art = Article(url)
        art.download()
        art.parse()
        full = art.text.strip()
        if full:
            text = full
    except Exception:
        pass
# fallback: 

    # 2) BeautifulSoup 
    if not text or len(text.split("\n\n")) < 2:
        try:
            resp = requests.get(url, headers=HEADERS, timeout=10)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")
            paras = [p.get_text(strip=True)for p in soup.find_all("p") if len(p.get_text(strip=True)) > 30]
            text = "\n\n".join(paras)
        except Exception:
            return ""

    # 3) Limit to 6 paragraphs for concise summaries
    return _limit_paragraphs(text, max_paras=6)


# --------------------------------------------------------------------------------------

document_fetcher_tool = Tool(
    name="document_fetcher",
    func=fetch,
    description=(
        "Fetch the main article body from all URLs (newspaper3k → BS4 fallback), "
        "then trim to the first 6 paragraphs only. Returns empty string on error."
    )
)
