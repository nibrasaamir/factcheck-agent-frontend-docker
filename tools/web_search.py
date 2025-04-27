import os
from dotenv import load_dotenv
import requests
from langchain.tools import Tool

load_dotenv()
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")

def search_news(query , num_results = 4):
    # will use google news api here to fetch articles
    # num_results = number of articles scrapped.

    parameters = { "engine": "google_news", "q": query, "api_key": SERPAPI_API_KEY,}

    resp = requests.get("https://serpapi.com/search", params=parameters, timeout=10)
    resp.raise_for_status()
    data = resp.json()

    urls = []
    for section in ("news_results", "organic_results"):
        for item in data.get(section, []):
            link = item.get("link")
            if link:
                urls.append(link)
            if len(urls) >= num_results:
                break
        if urls:
            break

    # Ensure we never return more than num_results
    urls = urls[:num_results]
    return "\n".join(urls)

web_search_tool = Tool(
    name="web_search",
    func=search_news,
    description=(
        "Fetch up to 4 news-article URLs for a query using SerpAPI (Google News). "
        "Input: a search string. Output: newline-separated valid URLs, max 4."
    )
)

# we are using max 4 because my API context length is less and it will give errors if tokens exceed the api limit.
