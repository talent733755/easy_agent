#!/bin/bash
set -e

cd "$(dirname "$0")"

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

PID_DIR=".run"

stop_process() {
    local name=$1
    local pid_file="$PID_DIR/${name}.pid"

    if [ ! -f "$pid_file" ]; then
        echo -e "${YELLOW}[WARN]${NC} $name 未在运行"
        return 0
    fi

    local pid=$(cat "$pid_file")
    if kill -0 "$pid" 2>/dev/null; then
        echo -e "${GREEN}[INFO]${NC} 停止 $name (PID: $pid)..."
        kill "$pid"
        # 等待进程退出
        local wait=0
        while kill -0 "$pid" 2>/dev/null && [ $wait -lt 5 ]; do
            sleep 1
            wait=$((wait + 1))
        done
        if kill -0 "$pid" 2>/dev/null; then
            echo -e "${YELLOW}[WARN]${NC} $name 未响应，强制终止..."
            kill -9 "$pid" 2>/dev/null || true
        fi
        echo -e "${GREEN}[OK]${NC} $name 已停止"
    else
        echo -e "${YELLOW}[WARN]${NC} $name 进程不存在 (PID: $pid)"
    fi

    rm -f "$pid_file"
}

stop_process "web"
stop_process "mcp_gateway"

# 清理空目录
[ -d "$PID_DIR" ] && rmdir "$PID_DIR" 2>/dev/null || true

echo ""
echo -e "${GREEN}所有服务已停止${NC}"
