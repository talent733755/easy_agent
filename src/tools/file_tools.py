from pathlib import Path
from langchain_core.tools import tool


@tool
def read_file(path: str) -> str:
    """Read contents of a file at the given path."""
    p = Path(path).expanduser()
    if not p.exists():
        return f"Error: file not found: {path}"
    try:
        content = p.read_text(encoding="utf-8")
        if len(content) > 4000:
            content = content[:4000] + "\n... (truncated)"
        return content
    except Exception as e:
        return f"Error reading file: {e}"


@tool
def write_file(path: str, content: str) -> str:
    """Write content to a file at the given path. Creates parent directories if needed."""
    p = Path(path).expanduser()
    p.parent.mkdir(parents=True, exist_ok=True)
    try:
        p.write_text(content, encoding="utf-8")
        return f"Successfully wrote {len(content)} chars to {path}"
    except Exception as e:
        return f"Error writing file: {e}"


@tool
def list_dir(path: str) -> str:
    """List files and directories at the given path."""
    p = Path(path).expanduser()
    if not p.exists():
        return f"Error: directory not found: {path}"
    if not p.is_dir():
        return f"Error: not a directory: {path}"
    items = []
    for item in sorted(p.iterdir()):
        suffix = "/" if item.is_dir() else ""
        items.append(f"  {item.name}{suffix}")
    return "\n".join(items) if items else "(empty directory)"
