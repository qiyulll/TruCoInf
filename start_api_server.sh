#!/bin/bash
# 启动 GaC API 服务器脚本

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# 获取脚本所在目录
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# 默认配置
CONFIG_FILE="config.yaml"
PORT=8000
HOST="0.0.0.0"

# 打印帮助信息
print_help() {
    cat << EOF
用法: ./start_api_server.sh [选项]

选项：
  --config FILE      配置文件路径（默认：config.yaml）
  --port PORT        监听端口（默认：8000）
  --host HOST        监听地址（默认：0.0.0.0）
  --help             显示此帮助信息

示例：
  ./start_api_server.sh                              # 使用默认配置启动
  ./start_api_server.sh --config config.yaml         # 使用指定配置
  ./start_api_server.sh --port 8080                  # 在 8080 端口启动

EOF
}

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        --host)
            HOST="$2"
            shift 2
            ;;
        --help)
            print_help
            exit 0
            ;;
        *)
            echo "未知选项: $1"
            print_help
            exit 1
            ;;
    esac
done

# 检查配置文件
if [ ! -f "$CONFIG_FILE" ]; then
    echo -e "${RED}✗ 配置文件不存在: $CONFIG_FILE${NC}"
    echo ""
    echo -e "${YELLOW}可用的配置文件：${NC}"
    ls -1 *.yaml 2>/dev/null || echo "  (未找到 .yaml 文件)"
    exit 1
fi

# 检查 Python 环境
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}✗ 未找到 Python 3${NC}"
    exit 1
fi

# 检查必要的 Python 包
echo -e "${YELLOW}检查 Python 依赖...${NC}"
python3 << 'EOF'
import sys
required_packages = ['fastapi', 'uvicorn', 'ray', 'transformers', 'torch']
missing = []

for package in required_packages:
    try:
        if package == 'torch':
            import torch
        else:
            __import__(package)
        print(f"  ✓ {package}")
    except ImportError:
        print(f"  ✗ {package} (缺失)")
        missing.append(package)

if missing:
    print(f"\n请安装缺失的包：")
    print(f"  pip install {' '.join(missing)}")
    sys.exit(1)
EOF

if [ $? -ne 0 ]; then
    exit 1
fi

# 设置环境变量
echo ""
echo -e "${YELLOW}设置环境变量...${NC}"
export ZK_EVIDENCE_ROOT_DIR="$SCRIPT_DIR/zk_evidence"
export ZK_ATTN_EVIDENCE=1
echo "  ZK_EVIDENCE_ROOT_DIR=$ZK_EVIDENCE_ROOT_DIR"
echo "  ZK_ATTN_EVIDENCE=$ZK_ATTN_EVIDENCE"

# 检查 zk_evidence 目录
if [ ! -d "$ZK_EVIDENCE_ROOT_DIR" ]; then
    echo -e "${YELLOW}创建 zk_evidence 目录...${NC}"
    mkdir -p "$ZK_EVIDENCE_ROOT_DIR"
fi

# 打印启动信息
echo ""
echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}GaC API 服务器${NC}"
echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}"
echo ""
echo -e "${YELLOW}配置：${NC}"
echo "  配置文件: $CONFIG_FILE"
echo "  监听地址: $HOST:$PORT"
echo "  API 端点: http://$HOST:$PORT/api/generate/"
echo "  状态端点: http://$HOST:$PORT/status"
echo ""
echo -e "${YELLOW}按 Ctrl+C 停止服务器${NC}"
echo ""

# 启动服务器
echo -e "${GREEN}启动服务器...${NC}"
echo ""

python3 gac_api_server.py \
    --config-path "$CONFIG_FILE" \
    --host "$HOST" \
    --port "$PORT"
