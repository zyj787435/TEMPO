#!/bin/bash
# ============================================================
# run_exp_queued.sh — 按 API key 排队启动实验，防止限流
#
# 用法:
#   bash scripts/run_exp_queued.sh <config.yaml> <task> <model> [key_id]
#
# 参数:
#   config.yaml  — eval_configs 下的配置文件路径
#   task         — scienceworld / alfworld / tool-query
#   model        — YAML 中 llm 块的 key 名 (如 GPT4oMini, Qwen25-72B)
#   key_id       — 可选，API key 的唯一标识符（用于排队判断）
#                  默认从 YAML 中自动提取 api_key 字段
#
# 工作原理:
#   1. 从 YAML 提取 api_key 作为排队 key
#   2. 检查是否有其他进程正在使用同一个 api_key
#   3. 有冲突则等待直到对方完成，再启动
#   4. 启动后输出 PID 和日志路径
# ============================================================

set -e

CONFIG=$1
TASK=$2
MODEL=$3

if [ -z "$CONFIG" ] || [ -z "$TASK" ] || [ -z "$MODEL" ]; then
    echo "用法: bash scripts/run_exp_queued.sh <config.yaml> <task> <model>"
    exit 1
fi

cd "$(dirname "$0")/.." && export PROJECT_PATH=$(pwd)

# 提取 log_path 和 api_key
LOG_PATH=$(grep "log_path:" "$CONFIG" | head -1 | awk '{print $2}' | envsubst)
API_KEY=$(grep "api_key:" "$CONFIG" | head -1 | awk '{print $2}')
KEY_SHORT="${API_KEY:0:12}..."

echo "[QUEUE] 实验配置: $CONFIG"
echo "[QUEUE] API Key:  $KEY_SHORT"
echo "[QUEUE] 日志路径: $LOG_PATH"

mkdir -p "$LOG_PATH"

# 等待同一 key 的其他进程完成
WAIT_COUNT=0
while true; do
    # 查找使用同一 api_key 的其他 eval_main 进程
    CONFLICT_PIDS=$(ps aux | grep "eval_main" | grep -v grep | awk '{print $2}')
    FOUND_CONFLICT=0
    for PID in $CONFLICT_PIDS; do
        CMDLINE=$(cat /proc/$PID/cmdline 2>/dev/null | tr '\0' ' ')
        CFG_OF_PID=$(echo "$CMDLINE" | grep -o 'exp[0-9]*[^ ]*\.yaml\|eval_configs/[^ ]*\.yaml' | head -1)
        if [ -n "$CFG_OF_PID" ] && [ "$CFG_OF_PID" != "$CONFIG" ]; then
            KEY_OF_PID=$(grep "api_key:" "data/eval_configs/$(basename $CFG_OF_PID)" 2>/dev/null | head -1 | awk '{print $2}')
            if [ "$KEY_OF_PID" = "$API_KEY" ]; then
                FOUND_CONFLICT=1
                CONFLICT_CFG=$(basename $CFG_OF_PID)
                break
            fi
        fi
    done

    if [ $FOUND_CONFLICT -eq 0 ]; then
        break
    fi

    WAIT_COUNT=$((WAIT_COUNT + 1))
    if [ $((WAIT_COUNT % 6)) -eq 1 ]; then
        echo "[QUEUE] 等待 $CONFLICT_CFG 完成（同key: $KEY_SHORT），已等待 $((WAIT_COUNT * 10))s..."
    fi
    sleep 10
done

echo "[QUEUE] 无冲突，启动实验..."

nohup /root/miniconda3/envs/autotool/bin/python -u agentboard/eval_main.py \
    --cfg-path "$CONFIG" \
    --tasks "$TASK" \
    --model "$MODEL" \
    > "$LOG_PATH/run.log" 2>&1 &

EXP_PID=$!
echo "[QUEUE] 已启动 PID=$EXP_PID"
echo "[QUEUE] 日志: $LOG_PATH/run.log"
echo $EXP_PID > "$LOG_PATH/pid"
