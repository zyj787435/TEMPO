#!/bin/bash
# 注意，第5步需要自己手动修改代理，第6步需要手动修改API key，虚拟环境名字用.venv


BLUE='\033[0;34m'
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 使用方法：
#   source build.sh

# 检查是否在虚拟环境中
if [[ "$VIRTUAL_ENV" != "" ]]; then
    echo -e "${RED}[WARNING] 当前已在虚拟环境中,请先退出虚拟环境后再运行此脚本"
    exit 1
fi
echo -e "${BLUE} 开始配置项目环境..."

# 1. 检查 Python 环境
echo -e "${BLUE} 检查 Python 环境..."

if ! command -v python3 &> /dev/null; then
    echo -e "${RED}[ERROR] 未找到 python3，请先安装 Python 3："
    exit 1
fi

echo -e "${GREEN}[SUCC] Python 版本: $(python3 --version)"

# 检查 venv 模块
if ! python3 -m venv --help &> /dev/null; then
    echo -e "${RED}[ERROR] python3-venv 未安装"
    exit 1
fi

echo -e "${GREEN}[SUCC] venv 模块可用"

# 2. 创建虚拟环境
if [ ! -d ".venv" ]; then
    echo "创建 Python 虚拟环境..."
    if ! python3 -m venv .venv; then
        echo -e "${RED}[ERROR] 虚拟环境创建失败"
        exit 1
    fi
else
    echo -e "${GREEN}[SUCC] 虚拟环境已存在"
fi

# 3. 激活虚拟环境
echo "激活虚拟环境..."
source .venv/bin/activate

# 4. 安装依赖
echo "安装项目依赖..."
pip install --upgrade pip -q
pip install -U sentence-transformers

# 5. 设置代理环境变量
echo "配置代理环境变量..."
export HTTP_PROXY="http://127.0.0.1:7890"
export http_proxy="http://127.0.0.1:7890"
export HTTPS_PROXY="http://127.0.0.1:7890"
export https_proxy="http://127.0.0.1:7890"

# 清除 SOCKS 代理（重要！避免与前面的HTTP或HTTPS冲突）
unset ALL_PROXY
unset all_proxy


# 6. 显示环境信息
echo ""
echo -e "${BLUE}[INFO] 环境信息："
echo "       Python: $(python --version)"
echo "       Pip: $(pip --version | cut -d' ' -f1-2)"
echo "       虚拟环境: $(which python)"
echo "       HTTP_PROXY: $HTTP_PROXY"
echo "       HTTPS_PROXY: $HTTPS_PROXY"
