from langchain_core.tools import tool


@tool
def web_search(query: str) -> str:
    """Search the web for information. Returns search results."""
    return f"[web_search] Results for: '{query}' — web search requires external API integration. Use this as a placeholder for duckduckgo-search or tavily-python."


@tool
def web_fetch(url: str) -> str:
    """Fetch and read content from a URL."""
    return f"[web_fetch] Fetching: {url} — web fetch requires external API integration. Use this as a placeholder for requests + BeautifulSoup."
