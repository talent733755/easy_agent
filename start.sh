#!/bin/bash
set -e

cd "$(dirname "$0")"

# ---- 颜色 ----
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# ---- .env 文件 ----
ENV_FILE=".env"
if [ ! -f "$ENV_FILE" ]; then
    echo -e "${YELLOW}[WARN]${NC} 未找到 .env 文件，正在从 .env.example 创建..."
    cp .env.example "$ENV_FILE"
    echo -e "${YELLOW}请先编辑 .env 填入 API Key，然后重新运行此脚本${NC}"
    exit 1
fi

# 加载环境变量
set -a
source "$ENV_FILE"
set +a

# ---- 虚拟环境 ----
VENV_DIR=".venv"
if [ ! -d "$VENV_DIR" ]; then
    echo -e "${GREEN}[INFO]${NC} 创建虚拟环境..."
    python3 -m venv "$VENV_DIR"
fi

echo -e "${GREEN}[INFO]${NC} 激活虚拟环境..."
source "$VENV_DIR/bin/activate"

# ---- 安装依赖 ----
echo -e "${GREEN}[INFO]${NC} 检查依赖..."
pip install -r requirements.txt -q

# ---- API Keys ----
export TAVILY_API_KEY="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"

# ---- PID 文件 ----
PID_DIR=".run"
mkdir -p "$PID_DIR"

start_process() {
    local name=$1
    local pid_file="$PID_DIR/${name}.pid"
    local cmd=$2

    if [ -f "$pid_file" ]; then
        local old_pid=$(cat "$pid_file")
        if kill -0 "$old_pid" 2>/dev/null; then
            echo -e "${YELLOW}[WARN]${NC} $name 已在运行 (PID: $old_pid)"
            return 0
        fi
        rm -f "$pid_file"
    fi

    echo -e "${GREEN}[INFO]${NC} 启动 $name ..."
    $cmd > "$PID_DIR/${name}.log" 2>&1 &
    local pid=$!
    echo "$pid" > "$pid_file"
    echo -e "${GREEN}[OK]${NC} $name 已启动 (PID: $pid, 日志: $PID_DIR/${name}.log)"
}

# ---- 启动服务 ----
start_process "web" "python web_app.py --host 0.0.0.0 --port 8080 --ws-ping-interval 30 --ws-ping-timeout 300"
start_process "mcp_gateway" "python -m uvicorn mcp_servers.gateway:app --host 0.0.0.0 --port 3001"

echo ""
echo -e "${GREEN}服务已启动！${NC}"
echo "  Web 界面: http://localhost:8080"
echo "  MCP 网关: http://localhost:3001"
echo "  停止服务: ./stop.sh"
echo "  查看日志: tail -f .run/web.log"
