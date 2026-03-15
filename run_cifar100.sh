### 陶的代码




#### 一趟完整实验
# python3 dml_fl.py --dataset cifar100  --num_classes 100 --num_channels 3 --model resnet18 --epochs 120 --gpu 1 --p 10 --ai 30 --alpha 0.3 --total_mb 586 --local_bs 32 --iid
python3 dml_fl.py --dataset cifar100  --num_classes 100 --num_channels 3 --model resnet18 --epochs 120 --gpu 1 --p 10 --ai 30 --alpha 0.3 --total_mb 586 --local_bs 32
python3 dml_fl.py --dataset cifar100  --num_classes 100 --num_channels 3 --model resnet18 --epochs 120 --gpu 1 --p 5 --ai 30 --alpha 0.3 --total_mb 586 --local_bs 32 --iid
python3 dml_fl.py --dataset cifar100  --num_classes 100 --num_channels 3 --model resnet18 --epochs 120 --gpu 1 --p 5 --ai 30 --alpha 0.3 --total_mb 586 --local_bs 32

python3 dml_fl_benchmark.py --dataset cifar100  --num_classes 100 --num_channels 3 --model resnet18 --epochs 120 --gpu 1 --p 10 --ai 30 --alpha 0.3 --total_mb 586 --local_bs 32 --iid
python3 dml_fl_benchmark.py --dataset cifar100  --num_classes 100 --num_channels 3 --model resnet18 --epochs 120 --gpu 1 --p 10 --ai 30 --alpha 0.3 --total_mb 586 --local_bs 32
python3 dml_fl_benchmark.py --dataset cifar100  --num_classes 100 --num_channels 3 --model resnet18 --epochs 120 --gpu 1 --p 5 --ai 30 --alpha 0.3 --total_mb 586 --local_bs 32 --iid
python3 dml_fl_benchmark.py --dataset cifar100  --num_classes 100 --num_channels 3 --model resnet18 --epochs 120 --gpu 1 --p 5 --ai 30 --alpha 0.3 --total_mb 586 --local_bs 32

# python3 dml_fl_RAMFL_version1_hash.py --dataset cifar100  --num_classes 100 --num_channels 3 --model resnet18 --epochs 120 --gpu 1 --p 10 --ai 30 --alpha 0.3 --total_mb 586 --local_bs 32 --iid
# python3 dml_fl_RAMFL_version1_hash.py --dataset cifar100  --num_classes 100 --num_channels 3 --model resnet18 --epochs 120 --gpu 1 --p 10 --ai 30 --alpha 0.3 --total_mb 586 --local_bs 32
# python3 dml_fl_RAMFL_version1_hash.py --dataset cifar100  --num_classes 100 --num_channels 3 --model resnet18 --epochs 120 --gpu 1 --p 5 --ai 30 --alpha 0.3 --total_mb 586 --local_bs 32 --iid
# python3 dml_fl_RAMFL_version1_hash.py --dataset cifar100  --num_classes 100 --num_channels 3 --model resnet18 --epochs 120 --gpu 1 --p 5 --ai 30 --alpha 0.3 --total_mb 586 --local_bs 32
echo "一趟完整的实验执行完毕（12个，3方法*4设定）"


# #### 一趟完整实验
# python3 dml_fl.py --dataset cifar100  --num_classes 100 --num_channels 3 --model resnet18 --epochs 120 --gpu 1 --p 10 --ai 30 --alpha 0.3 --total_mb 586 --local_bs 32 --iid
# python3 dml_fl.py --dataset cifar100  --num_classes 100 --num_channels 3 --model resnet18 --epochs 120 --gpu 1 --p 10 --ai 30 --alpha 0.3 --total_mb 586 --local_bs 32
# python3 dml_fl.py --dataset cifar100  --num_classes 100 --num_channels 3 --model resnet18 --epochs 120 --gpu 1 --p 5 --ai 30 --alpha 0.3 --total_mb 586 --local_bs 32 --iid
# python3 dml_fl.py --dataset cifar100  --num_classes 100 --num_channels 3 --model resnet18 --epochs 120 --gpu 1 --p 5 --ai 30 --alpha 0.3 --total_mb 586 --local_bs 32

# python3 dml_fl_benchmark.py --dataset cifar100  --num_classes 100 --num_channels 3 --model resnet18 --epochs 120 --gpu 1 --p 10 --ai 30 --alpha 0.3 --total_mb 586 --local_bs 32 --iid
# python3 dml_fl_benchmark.py --dataset cifar100  --num_classes 100 --num_channels 3 --model resnet18 --epochs 120 --gpu 1 --p 10 --ai 30 --alpha 0.3 --total_mb 586 --local_bs 32
# python3 dml_fl_benchmark.py --dataset cifar100  --num_classes 100 --num_channels 3 --model resnet18 --epochs 120 --gpu 1 --p 5 --ai 30 --alpha 0.3 --total_mb 586 --local_bs 32 --iid
# python3 dml_fl_benchmark.py --dataset cifar100  --num_classes 100 --num_channels 3 --model resnet18 --epochs 120 --gpu 1 --p 5 --ai 30 --alpha 0.3 --total_mb 586 --local_bs 32

# python3 dml_fl_RAMFL_version1_hash.py --dataset cifar100  --num_classes 100 --num_channels 3 --model resnet18 --epochs 120 --gpu 1 --p 10 --ai 30 --alpha 0.3 --total_mb 586 --local_bs 32 --iid
# python3 dml_fl_RAMFL_version1_hash.py --dataset cifar100  --num_classes 100 --num_channels 3 --model resnet18 --epochs 120 --gpu 1 --p 10 --ai 30 --alpha 0.3 --total_mb 586 --local_bs 32
# python3 dml_fl_RAMFL_version1_hash.py --dataset cifar100  --num_classes 100 --num_channels 3 --model resnet18 --epochs 120 --gpu 1 --p 5 --ai 30 --alpha 0.3 --total_mb 586 --local_bs 32 --iid
# python3 dml_fl_RAMFL_version1_hash.py --dataset cifar100  --num_classes 100 --num_channels 3 --model resnet18 --epochs 120 --gpu 1 --p 5 --ai 30 --alpha 0.3 --total_mb 586 --local_bs 32
# echo "一趟完整的实验执行完毕（12个，3方法*4设定）"

# #### 一趟完整实验
# python3 dml_fl.py --dataset cifar100  --num_classes 100 --num_channels 3 --model resnet18 --epochs 120 --gpu 1 --p 10 --ai 30 --alpha 0.3 --total_mb 586 --local_bs 32 --iid
# python3 dml_fl.py --dataset cifar100  --num_classes 100 --num_channels 3 --model resnet18 --epochs 120 --gpu 1 --p 10 --ai 30 --alpha 0.3 --total_mb 586 --local_bs 32
# python3 dml_fl.py --dataset cifar100  --num_classes 100 --num_channels 3 --model resnet18 --epochs 120 --gpu 1 --p 5 --ai 30 --alpha 0.3 --total_mb 586 --local_bs 32 --iid
# python3 dml_fl.py --dataset cifar100  --num_classes 100 --num_channels 3 --model resnet18 --epochs 120 --gpu 1 --p 5 --ai 30 --alpha 0.3 --total_mb 586 --local_bs 32

# python3 dml_fl_benchmark.py --dataset cifar100  --num_classes 100 --num_channels 3 --model resnet18 --epochs 120 --gpu 1 --p 10 --ai 30 --alpha 0.3 --total_mb 586 --local_bs 32 --iid
# python3 dml_fl_benchmark.py --dataset cifar100  --num_classes 100 --num_channels 3 --model resnet18 --epochs 120 --gpu 1 --p 10 --ai 30 --alpha 0.3 --total_mb 586 --local_bs 32
# python3 dml_fl_benchmark.py --dataset cifar100  --num_classes 100 --num_channels 3 --model resnet18 --epochs 120 --gpu 1 --p 5 --ai 30 --alpha 0.3 --total_mb 586 --local_bs 32 --iid
# python3 dml_fl_benchmark.py --dataset cifar100  --num_classes 100 --num_channels 3 --model resnet18 --epochs 120 --gpu 1 --p 5 --ai 30 --alpha 0.3 --total_mb 586 --local_bs 32

# python3 dml_fl_RAMFL_version1_hash.py --dataset cifar100  --num_classes 100 --num_channels 3 --model resnet18 --epochs 120 --gpu 1 --p 10 --ai 30 --alpha 0.3 --total_mb 586 --local_bs 32 --iid
# python3 dml_fl_RAMFL_version1_hash.py --dataset cifar100  --num_classes 100 --num_channels 3 --model resnet18 --epochs 120 --gpu 1 --p 10 --ai 30 --alpha 0.3 --total_mb 586 --local_bs 32
# python3 dml_fl_RAMFL_version1_hash.py --dataset cifar100  --num_classes 100 --num_channels 3 --model resnet18 --epochs 120 --gpu 1 --p 5 --ai 30 --alpha 0.3 --total_mb 586 --local_bs 32 --iid
# python3 dml_fl_RAMFL_version1_hash.py --dataset cifar100  --num_classes 100 --num_channels 3 --model resnet18 --epochs 120 --gpu 1 --p 5 --ai 30 --alpha 0.3 --total_mb 586 --local_bs 32
# echo "一趟完整的实验执行完毕（12个，3方法*4设定）"


# #### 一趟完整实验
# python3 dml_fl.py --dataset cifar100  --num_classes 100 --num_channels 3 --model resnet18 --epochs 120 --gpu 1 --p 10 --ai 30 --alpha 0.3 --total_mb 586 --local_bs 32 --iid
# python3 dml_fl.py --dataset cifar100  --num_classes 100 --num_channels 3 --model resnet18 --epochs 120 --gpu 1 --p 10 --ai 30 --alpha 0.3 --total_mb 586 --local_bs 32
# python3 dml_fl.py --dataset cifar100  --num_classes 100 --num_channels 3 --model resnet18 --epochs 120 --gpu 1 --p 5 --ai 30 --alpha 0.3 --total_mb 586 --local_bs 32 --iid
# python3 dml_fl.py --dataset cifar100  --num_classes 100 --num_channels 3 --model resnet18 --epochs 120 --gpu 1 --p 5 --ai 30 --alpha 0.3 --total_mb 586 --local_bs 32

# python3 dml_fl_benchmark.py --dataset cifar100  --num_classes 100 --num_channels 3 --model resnet18 --epochs 120 --gpu 1 --p 10 --ai 30 --alpha 0.3 --total_mb 586 --local_bs 32 --iid
# python3 dml_fl_benchmark.py --dataset cifar100  --num_classes 100 --num_channels 3 --model resnet18 --epochs 120 --gpu 1 --p 10 --ai 30 --alpha 0.3 --total_mb 586 --local_bs 32
# python3 dml_fl_benchmark.py --dataset cifar100  --num_classes 100 --num_channels 3 --model resnet18 --epochs 120 --gpu 1 --p 5 --ai 30 --alpha 0.3 --total_mb 586 --local_bs 32 --iid
# python3 dml_fl_benchmark.py --dataset cifar100  --num_classes 100 --num_channels 3 --model resnet18 --epochs 120 --gpu 1 --p 5 --ai 30 --alpha 0.3 --total_mb 586 --local_bs 32

# python3 dml_fl_RAMFL_version1_hash.py --dataset cifar100  --num_classes 100 --num_channels 3 --model resnet18 --epochs 120 --gpu 1 --p 10 --ai 30 --alpha 0.3 --total_mb 586 --local_bs 32 --iid
# python3 dml_fl_RAMFL_version1_hash.py --dataset cifar100  --num_classes 100 --num_channels 3 --model resnet18 --epochs 120 --gpu 1 --p 10 --ai 30 --alpha 0.3 --total_mb 586 --local_bs 32
# python3 dml_fl_RAMFL_version1_hash.py --dataset cifar100  --num_classes 100 --num_channels 3 --model resnet18 --epochs 120 --gpu 1 --p 5 --ai 30 --alpha 0.3 --total_mb 586 --local_bs 32 --iid
# python3 dml_fl_RAMFL_version1_hash.py --dataset cifar100  --num_classes 100 --num_channels 3 --model resnet18 --epochs 120 --gpu 1 --p 5 --ai 30 --alpha 0.3 --total_mb 586 --local_bs 32
# echo "一趟完整的实验执行完毕（12个，3方法*4设定）"