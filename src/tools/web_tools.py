import json
import os
import re
import warnings
from urllib.parse import quote_plus

from langchain_core.tools import tool
import requests
from bs4 import BeautifulSoup

# Suppress deprecation warning from duckduckgo_search rename
warnings.filterwarnings("ignore", message=".*duckduckgo_search.*renamed.*")

# ---------------------------------------------------------------------------
# Configurable search backends
# ---------------------------------------------------------------------------
# Priority order:
#   1. TAVILY_API_KEY env var → Tavily AI search (recommended, works in CN)
#   2. SEARXNG_URL    env var → self-hosted SearXNG instance
#   3. SEARCH_API     env var → custom JSON search API (see protocol below)
#   4. DuckDuckGo fallback     (likely blocked in CN)
# ---------------------------------------------------------------------------
# Custom Search API protocol (SEARCH_API):
#   POST {url}  Content-Type: application/json
#   Request:  {"q": "...", "max_results": N}
#   Response: {"results": [{"title": "...", "url": "...", "snippet": "..."}]}


def _search_searxng(query: str, max_results: int = 5) -> str | None:
    """Search via a self-hosted SearXNG instance."""
    instance = os.environ.get("SEARXNG_URL", "")
    if not instance:
        return None
    try:
        resp = requests.post(
            f"{instance.rstrip('/')}/search",
            data={"q": query, "format": "json", "language": "zh-CN"},
            timeout=15,
        )
        data = resp.json()
        out = []
        for r in data.get("results", [])[:max_results]:
            out.append(
                f"[{len(out)+1}] {r.get('title', '')}\n"
                f"    链接: {r.get('url', '')}\n"
                f"    摘要: {r.get('content', '')}"
            )
        return f"搜索 '{query}' 的结果:\n\n" + "\n\n".join(out) if out else None
    except Exception as e:
        return None


def _search_custom_api(query: str, max_results: int = 5) -> str | None:
    """Search via a custom JSON API (configured via SEARCH_API env var)."""
    url = os.environ.get("SEARCH_API", "")
    if not url:
        return None
    try:
        resp = requests.post(
            url,
            json={"q": query, "max_results": max_results},
            headers={"Content-Type": "application/json"},
            timeout=15,
        )
        data = resp.json()
        out = []
        for r in data.get("results", [])[:max_results]:
            out.append(
                f"[{len(out)+1}] {r.get('title', '')}\n"
                f"    链接: {r.get('url', '')}\n"
                f"    摘要: {r.get('snippet', '')}"
            )
        return f"搜索 '{query}' 的结果:\n\n" + "\n\n".join(out) if out else None
    except Exception as e:
        return None


def _search_duckduckgo(query: str, max_results: int = 5) -> str | None:
    """Search via DuckDuckGo."""
    try:
        from duckduckgo_search import DDGS

        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
        if not results:
            return None
        out = []
        for i, r in enumerate(results, 1):
            out.append(
                f"[{i}] {r.get('title', '')}\n"
                f"    链接: {r.get('href', '')}\n"
                f"    摘要: {r.get('body', '')}"
            )
        return f"搜索 '{query}' 的结果:\n\n" + "\n".join(out)
    except Exception:
        return None


def _search_tavily(query: str, max_results: int = 5) -> str | None:
    """Search via Tavily AI search API."""
    api_key = os.environ.get("TAVILY_API_KEY")
    if not api_key:
        return None
    try:
        from tavily import TavilyClient

        client = TavilyClient(api_key=api_key)
        resp = client.search(query=query, max_results=max_results, include_answer=False)
        results = resp.get("results", [])
        if not results:
            return None
        out = []
        for i, r in enumerate(results, 1):
            out.append(
                f"[{i}] {r.get('title', '')}\n"
                f"    链接: {r.get('url', '')}\n"
                f"    摘要: {r.get('content', '')}"
            )
        answer = resp.get("answer", "")
        if answer:
            return f"搜索 '{query}' 的结果:\n\n" + "\n\n".join(out) + f"\n\n📌 摘要: {answer}"
        return f"搜索 '{query}' 的结果:\n\n" + "\n\n".join(out)
    except Exception:
        return None


@tool
def web_search(query: str, max_results: int = 5) -> str:
    """
    Search the web for information.

    Tries backends in order (Tavily → SearXNG → custom API → DuckDuckGo).
    Configure via environment variables:
      TAVILY_API_KEY  - Tavily search API key (recommended, works in China)
      SEARXNG_URL     - Self-hosted SearXNG instance URL
      SEARCH_API      - Custom JSON search API endpoint

    Args:
        query: The search query string
        max_results: Maximum number of results to return (default: 5)
    """
    backends = [
        ("Tavily", _search_tavily),
        ("SearXNG", _search_searxng),
        ("Custom API", _search_custom_api),
        ("DuckDuckGo", _search_duckduckgo),
    ]
    errors = []
    for name, fn in backends:
        result = fn(query, max_results)
        if result:
            return result
        errors.append(name)
    return (
        f"搜索暂不可用（已尝试: {', '.join(errors)}）。\n"
        f"提示：可设置环境变量 SEARXNG_URL 或 SEARCH_API 来配置搜索后端。"
    )


@tool
def web_fetch(url: str) -> str:
    """
    Fetch and read content from a URL.
    
    Args:
        url: The URL to fetch content from
    """
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        
        # 解析HTML
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # 移除脚本和样式标签
        for script in soup(["script", "style", "nav", "footer", "header"]):
            script.decompose()
        
        # 获取文本
        text = soup.get_text(separator='\n', strip=True)
        
        # 清理空行
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        text = '\n'.join(lines)
        
        # 限制返回长度
        max_length = 5000
        if len(text) > max_length:
            text = text[:max_length] + "\n\n[内容已截断，原始内容过长]"
        
        return f"从 {url} 获取的内容:\n\n{text}"
    
    except requests.RequestException as e:
        return f"获取网页时出错: {str(e)}"
    except Exception as e:
        return f"处理网页内容时出错: {str(e)}"
