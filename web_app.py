#!/usr/bin/env python3
"""Easy Agent Web Application Entry Point."""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

import uvicorn

from web.app import create_app


def parse_args():
    parser = argparse.ArgumentParser(description="Easy Agent Web Interface")
    parser.add_argument("--host", "-H", default="0.0.0.0", help="Host to bind")
    parser.add_argument("--port", "-p", type=int, default=8080, help="Port to bind")
    parser.add_argument("--config", "-c", help="Path to config.yaml")
    parser.add_argument("--reload", "-r", action="store_true", help="Enable auto-reload")
    parser.add_argument("--ws-ping-interval", type=int, default=30, help="WebSocket ping interval in seconds")
    parser.add_argument("--ws-ping-timeout", type=int, default=300, help="WebSocket ping timeout in seconds")
    return parser.parse_args()


def main():
    args = parse_args()

    app = create_app(args.config)

    print(f"""
╔══════════════════════════════════════════╗
║         Easy Agent Web v0.1.0            ║
║                                          ║
║  Open http://{args.host}:{args.port} in your browser  ║
╚══════════════════════════════════════════╝
""")

    uvicorn.run(
        "web_app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        ws_ping_interval=args.ws_ping_interval,
        ws_ping_timeout=args.ws_ping_timeout,
    )


if __name__ == "__main__":
    app = create_app()
    main()
else:
    # For uvicorn import
    app = create_app()