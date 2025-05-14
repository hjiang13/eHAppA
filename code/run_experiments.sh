#!/bin/bash

# 配置
POOLING_TYPES=("mean" "max" "attn" "lstm")
OVERLAP_RATIOS=(0 0.25 0.5 0.75)
LORA_OPTIONS=("off" "on")
RESULT_LOG="mse_results_summary.txt"

# 清空旧结果并写表头
echo "Pooling | Overlap | LoRA | Final MSE" > $RESULT_LOG
echo "--------|---------|------|-----------" >> $RESULT_LOG

# 遍历所有组合
for pooling in "${POOLING_TYPES[@]}"; do
  for overlap in "${OVERLAP_RATIOS[@]}"; do
    for lora in "${LORA_OPTIONS[@]}"; do
      
      echo ">>> Running: pooling=$pooling, overlap=$overlap, lora=$lora"

      CMD="python3 BERTRegression_chunck_functional.py \
            --pooling_type $pooling \
            --overlap_ratio $overlap \
            --epochs 5 \
            --batch_size 8"

      if [ "$lora" == "on" ]; then
        CMD="$CMD --use_lora"
      fi

      # 执行并捕获输出
      OUTPUT=$($CMD 2>&1)

      # 提取 Final MSE
      FINAL_MSE=$(echo "$OUTPUT" | grep "Final MSE:" | awk '{printf "%.4f", $3}')

      # 写入结果表格
      printf "%-7s | %-7s | %-4s | %s\n" "$pooling" "$overlap" "$lora" "$FINAL_MSE" >> $RESULT_LOG
    done
  done
done

echo -e "\n✅ 所有组合测试完成，结果写入：$RESULT_LOG"