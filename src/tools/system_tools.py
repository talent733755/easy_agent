import subprocess
from datetime import datetime
from langchain_core.tools import tool


@tool
def get_time() -> str:
    """Get the current date and time."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


@tool
def run_shell(command: str) -> str:
    """Execute a shell command and return the output. Use with caution."""
    try:
        result = subprocess.run(
            command, shell=True, capture_output=True, text=True, timeout=30
        )
        output = result.stdout.strip() or result.stderr.strip()
        if len(output) > 2000:
            output = output[:2000] + "\n... (truncated)"
        return output if output else "(no output)"
    except subprocess.TimeoutExpired:
        return "Error: command timed out after 30s"
    except Exception as e:
        return f"Error executing command: {e}"
