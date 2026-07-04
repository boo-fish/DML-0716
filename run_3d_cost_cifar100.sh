## cifar and vgg


#### 完整的3D
python3 dml_fl.py --dataset cifar100  --num_classes 100  --num_channels 3 --model resnet18 --epochs 100 --gpu 3 --p 10 --ai 30 --alpha 0.3 --total_mb 586  --lr 0.001 --local_bs 32
python3 dml_fl.py --dataset cifar100  --num_classes 100  --num_channels 3 --model resnet18 --epochs 100 --gpu 3 --p 10 --ai 30 --alpha 0.6 --total_mb 586  --lr 0.001 --local_bs 32
python3 dml_fl.py --dataset cifar100  --num_classes 100  --num_channels 3 --model resnet18 --epochs 100 --gpu 3 --p 10 --ai 30 --alpha 0.8 --total_mb 586  --lr 0.001 --local_bs 32
python3 dml_fl.py --dataset cifar100  --num_classes 100  --num_channels 3 --model resnet18 --epochs 100 --gpu 3 --p 10 --ai 30 --alpha 1.0 --total_mb 586  --lr 0.001 --local_bs 32


python3 dml_fl.py --dataset cifar100  --num_classes 100  --num_channels 3 --model resnet18 --epochs 100 --gpu 3 --p 5 --ai 30 --alpha 0.3 --total_mb 586  --lr 0.001 --local_bs 32
python3 dml_fl.py --dataset cifar100  --num_classes 100  --num_channels 3 --model resnet18 --epochs 100 --gpu 3 --p 5 --ai 30 --alpha 0.6 --total_mb 586  --lr 0.001 --local_bs 32
python3 dml_fl.py --dataset cifar100  --num_classes 100  --num_channels 3 --model resnet18 --epochs 100 --gpu 3 --p 5 --ai 30 --alpha 0.8 --total_mb 586  --lr 0.001 --local_bs 32
python3 dml_fl.py --dataset cifar100  --num_classes 100  --num_channels 3 --model resnet18 --epochs 100 --gpu 3 --p 5 --ai 30 --alpha 1.0 --total_mb 586  --lr 0.001 --local_bs 32



python3 dml_fl.py --dataset cifar100  --num_classes 100  --num_channels 3 --model resnet18 --epochs 100 --gpu 3 --p 3 --ai 30 --alpha 0.3 --total_mb 586  --lr 0.001 --local_bs 32
python3 dml_fl.py --dataset cifar100  --num_classes 100  --num_channels 3 --model resnet18 --epochs 100 --gpu 3 --p 3 --ai 30 --alpha 0.6 --total_mb 586  --lr 0.001 --local_bs 32
python3 dml_fl.py --dataset cifar100  --num_classes 100  --num_channels 3 --model resnet18 --epochs 100 --gpu 3 --p 3 --ai 30 --alpha 0.8 --total_mb 586  --lr 0.001 --local_bs 32
python3 dml_fl.py --dataset cifar100  --num_classes 100  --num_channels 3 --model resnet18 --epochs 100 --gpu 3 --p 3 --ai 30 --alpha 1.0 --total_mb 586  --lr 0.001 --local_bs 32
#### 完整的3D



#### 追加跑RAMFL的500轮实验
python3 dml_fl_RAMFL_version1_hash.py --dataset cifar100  --num_classes 100 --num_channels 3 --model resnet18 --epochs 500 --gpu 3 --p 10 --ai 30 --alpha 0.3 --total_mb 586 --local_bs 32 --iid
python3 dml_fl_RAMFL_version1_hash.py --dataset cifar100  --num_classes 100 --num_channels 3 --model resnet18 --epochs 500 --gpu 3 --p 10 --ai 30 --alpha 0.3 --total_mb 586 --local_bs 32
python3 dml_fl_RAMFL_version1_hash.py --dataset cifar100  --num_classes 100 --num_channels 3 --model resnet18 --epochs 500 --gpu 3 --p 5 --ai 30 --alpha 0.3 --total_mb 586 --local_bs 32 --iid
python3 dml_fl_RAMFL_version1_hash.py --dataset cifar100  --num_classes 100 --num_channels 3 --model resnet18 --epochs 500 --gpu 3 --p 5 --ai 30 --alpha 0.3 --total_mb 586 --local_bs 32


echo "一趟完整的实验执行完毕（12 + 4）"
