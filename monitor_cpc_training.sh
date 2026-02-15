#!/bin/bash
# 实时监控 CPC 训练进度

LOG_FILE="/data/projects/punim2637/nnliang/SLSforASVspoof-2021-DF/logs/train_cpc_20410243.out"
ERR_FILE="/data/projects/punim2637/nnliang/SLSforASVspoof-2021-DF/logs/train_cpc_20410243.err"
CSV_FILE="/data/projects/punim2637/nnliang/SLSforASVspoof-2021-DF/models/cpc_window_w8_LA_e40_bs14_lr1e-06_saeW0.1_cpcW0.5_dict4096_k128_cpc/training_log.csv"

echo "========================================"
echo "CPC 训练进度监控"
echo "========================================"
echo ""

# 检查任务状态
echo "📊 任务状态:"
squeue -u $USER | grep -E "JOBID|cpc" || echo "任务已完成或未找到"
echo ""

# 显示训练进度（从 CSV）
if [ -f "$CSV_FILE" ]; then
    echo "📈 训练历史 (最近 5 个 epoch):"
    echo "Epoch | Train Loss | Train EER | Val Loss | Val EER | Val Acc"
    echo "------|------------|-----------|----------|---------|--------"
    tail -5 "$CSV_FILE" | awk -F',' 'NR>1 {printf "  %2s  |   %6.4f   |  %5.2f%%  | %6.4f  | %5.2f%% | %5.2f%%\n", $1, $3, $7, $8, $12, $11}'
    echo ""
    
    # 显示最佳结果
    BEST_LINE=$(tail -n +2 "$CSV_FILE" | sort -t',' -k12 -n | head -1)
    if [ -n "$BEST_LINE" ]; then
        BEST_EPOCH=$(echo "$BEST_LINE" | cut -d',' -f1)
        BEST_EER=$(echo "$BEST_LINE" | cut -d',' -f12)
        echo "🏆 最佳验证 EER: $BEST_EER% (Epoch $BEST_EPOCH)"
    fi
else
    echo "⏳ 训练日志尚未生成，等待第一个 epoch 完成..."
fi

echo ""
echo "========================================"
echo "📝 最新日志 (最后 30 行):"
echo "========================================"
tail -30 "$LOG_FILE"

# 检查是否有错误
if [ -f "$ERR_FILE" ] && [ -s "$ERR_FILE" ]; then
    echo ""
    echo "⚠️  错误日志 (最后 10 行):"
    tail -10 "$ERR_FILE"
fi

echo ""
echo "========================================"
echo "刷新时间: $(date)"
echo "使用 'watch -n 30 bash monitor_cpc_training.sh' 自动刷新"
echo "========================================"
