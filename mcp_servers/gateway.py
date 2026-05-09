"""MCP 统一网关：自动发现并挂载所有 MCP 服务。"""
import importlib
import pkgutil
import pathlib
import sys

from fastapi import FastAPI

# 确保项目根目录在 sys.path 中（用于 uvicorn 启动）
_project_root = str(pathlib.Path(__file__).parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

app = FastAPI(title="MCP Gateway", version="1.0.0")


def discover_and_mount(fastapi_app: FastAPI) -> list[str]:
    """扫描 mcp_servers/ 子目录，挂载所有包含 router.py 的服务。

    Returns:
        已挂载的服务名列表
    """
    servers_dir = pathlib.Path(__file__).parent
    mounted = []

    for importer, name, is_pkg in pkgutil.iter_modules([str(servers_dir)]):
        if name in ("gateway", "__pycache__"):
            continue
        try:
            module = importlib.import_module(f"mcp_servers.{name}.router")
            if hasattr(module, "router"):
                fastapi_app.include_router(module.router, prefix=f"/{name}")
                mounted.append(name)
        except ImportError:
            pass

    return mounted


@app.get("/health")
async def health():
    return {"status": "ok"}


# 启动时自动发现并挂载
mounted_services = discover_and_mount(app)
