# 2. 定义要遍历的超参数组合，可按需扩展
A_MIN_VALUES=(20 30 40)
CAP_MIN_VALUES=(20 30 40)

# 3. 创建日志保存目录，避免输出混乱
mkdir -p ./experiment_logs

# 4. 双重循环遍历所有 3×3=9 组参数组合
for a_min in "${A_MIN_VALUES[@]}"; do
  for cap_min in "${CAP_MIN_VALUES[@]}"; do
    echo "======================================"
    echo "开始实验: A_min=$a_min, cap_min=$cap_min"
    echo "开始时间: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "======================================"

    # 运行Python脚本：固定参数原样保留，动态参数替换变量
    python onlineFL_step2_Online_FL-FINAL.py \
      --dataset cifar10 \
      --num_classes 10 \
      --num_channels 3 \
      --model vgg11 \
      --total_slots 100 \
      --T 10 \
      --gpu 0 \
      --p 20 \
      --alpha 1.0 \
      --total_mb 586 \
      --lr 0.001 \
      --local_bs 32 \
      --A_min "$a_min" \
      --cap_min "$cap_min" \
      > ./experiment_logs/result_A${a_min}_cap${cap_min}.log 2>&1

    echo "实验完成: A_min=$a_min, cap_min=$cap_min"
    echo "结束时间: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "日志路径: ./experiment_logs/result_A${a_min}_cap${cap_min}.log"
    echo ""
  done
done

echo "所有 9 组实验全部运行完成！"