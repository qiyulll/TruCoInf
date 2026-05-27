#!/bin/bash
# 快速启动基准测试脚本

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 获取脚本所在目录
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# 默认配置
API_URL="http://localhost:8000/api/generate/"
CONFIG_FILE="config.yaml"
BENCHMARK_TYPE="all"
MAX_SAMPLES=""
MMLU_SUBJECTS=""
QUICK_TEST=false

# 打印帮助信息
print_help() {
    cat << EOF
用法: ./run_benchmark.sh [选项]

基准测试类型：
  --all              同时运行 MMLU 和 GSM8K（默认）
  --mmlu             只运行 MMLU 基准测试
  --gsm8k            只运行 GSM8K 基准测试

选项：
  --url URL          API 服务器地址（默认：http://localhost:8000/api/generate/）
  --config FILE      配置文件路径（默认：config.yaml）
  --max-samples N    限制样本数（用于快速测试）
  --quick            快速测试模式（各基准10个样本）
  --help             显示此帮助信息

示例：
  ./run_benchmark.sh --all              # 运行完整基准测试
  ./run_benchmark.sh --mmlu --quick     # 快速测试 MMLU
  ./run_benchmark.sh --gsm8k --max-samples 50  # 测试 GSM8K 50个样本

EOF
}

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --all)
            BENCHMARK_TYPE="all"
            shift
            ;;
        --mmlu)
            BENCHMARK_TYPE="mmlu"
            shift
            ;;
        --gsm8k)
            BENCHMARK_TYPE="gsm8k"
            shift
            ;;
        --url)
            API_URL="$2"
            shift 2
            ;;
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --max-samples)
            MAX_SAMPLES="$2"
            shift 2
            ;;
        --quick)
            QUICK_TEST=true
            MAX_SAMPLES="10"
            shift
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

# 打印配置
echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}zkhook 基准测试启动脚本${NC}"
echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}"
echo ""
echo -e "${YELLOW}配置信息：${NC}"
echo "  API URL: $API_URL"
echo "  配置文件: $CONFIG_FILE"
echo "  测试类型: $BENCHMARK_TYPE"
if [ ! -z "$MAX_SAMPLES" ]; then
    echo "  最大样本数: $MAX_SAMPLES"
fi
if [ "$QUICK_TEST" = true ]; then
    echo "  模式: 快速测试"
fi
echo ""

# 检查配置文件
if [ ! -f "$CONFIG_FILE" ]; then
    echo -e "${RED}✗ 配置文件不存在: $CONFIG_FILE${NC}"
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
required_packages = ['requests', 'datasets', 'tqdm']
missing = []

for package in required_packages:
    try:
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
    echo -e "${RED}缺失依赖包，请按上述提示安装${NC}"
    exit 1
fi

# 检查 API 服务器连接
echo ""
echo -e "${YELLOW}检查 API 服务器连接...${NC}"
if ! curl -s "${API_URL%/api/generate/}/status" > /dev/null 2>&1; then
    echo -e "${RED}✗ 无法连接到 API 服务器: $API_URL${NC}"
    echo -e "${YELLOW}请确保服务器已启动。在另一个终端运行：${NC}"
    echo ""
    echo "  export ZK_EVIDENCE_ROOT_DIR=\"$SCRIPT_DIR/zk_evidence\""
    echo "  export ZK_ATTN_EVIDENCE=1"
    echo "  python3 gac_api_server.py --config-path $CONFIG_FILE --port 8000"
    echo ""
    exit 1
fi
echo -e "${GREEN}✓ 服务器连接正常${NC}"

# 构建命令行参数
BENCHMARK_ARGS="--url $API_URL"
if [ ! -z "$MAX_SAMPLES" ]; then
    BENCHMARK_ARGS="$BENCHMARK_ARGS --max-samples $MAX_SAMPLES"
fi

# 记录开始时间
START_TIME=$(date +%s)
echo ""
echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}开始基准测试${NC}"
echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}"
echo ""

# 运行基准测试
case $BENCHMARK_TYPE in
    all)
        echo -e "${YELLOW}运行综合基准测试...${NC}"
        python3 benchmark_all.py --all $BENCHMARK_ARGS
        ;;
    mmlu)
        echo -e "${YELLOW}运行 MMLU 基准测试...${NC}"
        python3 benchmark_mmlu.py $BENCHMARK_ARGS
        ;;
    gsm8k)
        echo -e "${YELLOW}运行 GSM8K 基准测试...${NC}"
        python3 benchmark_gsm8k.py $BENCHMARK_ARGS
        ;;
esac

# 计算耗时
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
HOURS=$((ELAPSED / 3600))
MINUTES=$(( (ELAPSED % 3600) / 60 ))
SECONDS=$((ELAPSED % 60))

# 打印总结
echo ""
echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}基准测试完成！${NC}"
echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}"
echo ""
echo -e "${YELLOW}总耗时: ${HOURS}:$(printf "%02d" $MINUTES):$(printf "%02d" $SECONDS)${NC}"
echo ""
echo -e "${YELLOW}结果文件：${NC}"
ls -lh *.json 2>/dev/null | tail -5 || echo "  (未生成结果文件)"
echo ""
echo -e "${YELLOW}查看更多信息请参考: BENCHMARK_README.md${NC}"
echo ""
