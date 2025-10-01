#!/bin/bash

# -------------------------- 配置参数 --------------------------
# 1. 目标脚本路径（需替换为你的 train.sh 实际路径）
TRAIN_SCRIPT="/root/DeepWerewolf/examples/werewolf/train.sh"
# 2. 日志文件路径（记录重启时间、原因，便于排查）
LOG_FILE="/root/DeepWerewolf/examples/werewolf/train_monitor.log"
# 3. 检测间隔（秒）：避免频繁检测占用资源，建议 5-30 秒
CHECK_INTERVAL=10
# --------------------------------------------------------------

# 日志函数：自动添加时间戳
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" >> "$LOG_FILE"
}

# 初始化日志
log "=== 监控脚本启动 ==="
log "目标脚本：$TRAIN_SCRIPT"
log "检测间隔：$CHECK_INTERVAL 秒"

# 循环监控
while true; do
    # 关键：检测 train.sh 的进程是否存在（排除当前监控脚本自身）
    # pgrep -f：匹配完整命令行；grep -v：排除监控脚本进程；wc -l：统计存活进程数
    PROCESS_NUM=$(pgrep -f "$TRAIN_SCRIPT" | grep -v "monitor_train.sh" | wc -l)

    if [ $PROCESS_NUM -eq 0 ]; then
        # 进程不存在：记录日志并重启
        log "ERROR: train.sh 进程已退出，正在重启..."
        # 后台启动 train.sh（避免阻塞监控脚本），并将训练日志追加到文件（可选）
        nohup bash "$TRAIN_SCRIPT" >> "/root/DeepWerewolf/examples/werewolf/train.log" 2>&1 &
        log "train.sh 重启完成，新进程ID：$!"  # $! 表示刚启动进程的PID
    else
        # 进程正常：可选记录日志（减少冗余，可注释）
        # log "train.sh 运行正常，存活进程数：$PROCESS_NUM"
        :  # 空操作，仅占位
    fi

    # 间隔指定时间后再次检测
    sleep $CHECK_INTERVAL
done