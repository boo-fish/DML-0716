## cifar and vgg


#### DML 完整的3D
# python3 dml_fl.py --dataset cifar100  --num_classes 100  --num_channels 3 --model resnet18 --epochs 100 --gpu 0 --p 10 --ai 30 --alpha 0.3 --total_mb 586  --lr 0.001 --local_bs 32
# python3 dml_fl.py --dataset cifar100  --num_classes 100  --num_channels 3 --model resnet18 --epochs 100 --gpu 0 --p 10 --ai 30 --alpha 0.6 --total_mb 586  --lr 0.001 --local_bs 32
# python3 dml_fl.py --dataset cifar100  --num_classes 100  --num_channels 3 --model resnet18 --epochs 100 --gpu 0 --p 10 --ai 30 --alpha 0.8 --total_mb 586  --lr 0.001 --local_bs 32
# python3 dml_fl.py --dataset cifar100  --num_classes 100  --num_channels 3 --model resnet18 --epochs 100 --gpu 0 --p 10 --ai 30 --alpha 1.0 --total_mb 586  --lr 0.001 --local_bs 32


# python3 dml_fl.py --dataset cifar100  --num_classes 100  --num_channels 3 --model resnet18 --epochs 100 --gpu 0 --p 5 --ai 30 --alpha 0.3 --total_mb 586  --lr 0.001 --local_bs 32
# python3 dml_fl.py --dataset cifar100  --num_classes 100  --num_channels 3 --model resnet18 --epochs 100 --gpu 0 --p 5 --ai 30 --alpha 0.6 --total_mb 586  --lr 0.001 --local_bs 32
# python3 dml_fl.py --dataset cifar100  --num_classes 100  --num_channels 3 --model resnet18 --epochs 100 --gpu 0 --p 5 --ai 30 --alpha 0.8 --total_mb 586  --lr 0.001 --local_bs 32
# python3 dml_fl.py --dataset cifar100  --num_classes 100  --num_channels 3 --model resnet18 --epochs 100 --gpu 0 --p 5 --ai 30 --alpha 1.0 --total_mb 586  --lr 0.001 --local_bs 32



# python3 dml_fl.py --dataset cifar100  --num_classes 100  --num_channels 3 --model resnet18 --epochs 100 --gpu 0 --p 3 --ai 30 --alpha 0.3 --total_mb 586  --lr 0.001 --local_bs 32
# python3 dml_fl.py --dataset cifar100  --num_classes 100  --num_channels 3 --model resnet18 --epochs 100 --gpu 0 --p 3 --ai 30 --alpha 0.6 --total_mb 586  --lr 0.001 --local_bs 32
# python3 dml_fl.py --dataset cifar100  --num_classes 100  --num_channels 3 --model resnet18 --epochs 100 --gpu 0 --p 3 --ai 30 --alpha 0.8 --total_mb 586  --lr 0.001 --local_bs 32
# python3 dml_fl.py --dataset cifar100  --num_classes 100  --num_channels 3 --model resnet18 --epochs 100 --gpu 0 --p 3 --ai 30 --alpha 1.0 --total_mb 586  --lr 0.001 --local_bs 32
#### 完整的3D

### Online 完整的  24个
# T = 10
python onlineFL_step2_Online_FL-FINAL.py --dataset cifar10  --num_classes 10 --num_channels 3 --model vgg11 --total_slots 160 --T 10 --gpu 0 --p 20 --alpha 0.3 --total_mb 586  --lr 0.001 --local_bs 32
python onlineFL_step2_Online_FL-FINAL.py --dataset cifar10  --num_classes 10 --num_channels 3 --model vgg11 --total_slots 160 --T 10 --gpu 0 --p 20 --alpha 0.6 --total_mb 586  --lr 0.001 --local_bs 32
python onlineFL_step2_Online_FL-FINAL.py --dataset cifar10  --num_classes 10 --num_channels 3 --model vgg11 --total_slots 160 --T 10 --gpu 0 --p 20 --alpha 0.8 --total_mb 586  --lr 0.001 --local_bs 32
python onlineFL_step2_Online_FL-FINAL.py --dataset cifar10  --num_classes 10 --num_channels 3 --model vgg11 --total_slots 160 --T 10 --gpu 0 --p 20 --alpha 1.0 --total_mb 586  --lr 0.001 --local_bs 32

python onlineFL_step2_Online_FL-FINAL.py --dataset cifar10  --num_classes 10 --num_channels 3 --model vgg11 --total_slots 160 --T 10 --gpu 0 --p 10 --alpha 0.3 --total_mb 586  --lr 0.001 --local_bs 32
python onlineFL_step2_Online_FL-FINAL.py --dataset cifar10  --num_classes 10 --num_channels 3 --model vgg11 --total_slots 160 --T 10 --gpu 0 --p 10 --alpha 0.6 --total_mb 586  --lr 0.001 --local_bs 32
python onlineFL_step2_Online_FL-FINAL.py --dataset cifar10  --num_classes 10 --num_channels 3 --model vgg11 --total_slots 160 --T 10 --gpu 0 --p 10 --alpha 0.8 --total_mb 586  --lr 0.001 --local_bs 32
python onlineFL_step2_Online_FL-FINAL.py --dataset cifar10  --num_classes 10 --num_channels 3 --model vgg11 --total_slots 160 --T 10 --gpu 0 --p 10 --alpha 1.0 --total_mb 586  --lr 0.001 --local_bs 32

python onlineFL_step2_Online_FL-FINAL.py --dataset cifar10  --num_classes 10 --num_channels 3 --model vgg11 --total_slots 160 --T 10 --gpu 0 --p 5 --alpha 0.3 --total_mb 586  --lr 0.001 --local_bs 32
python onlineFL_step2_Online_FL-FINAL.py --dataset cifar10  --num_classes 10 --num_channels 3 --model vgg11 --total_slots 160 --T 10 --gpu 0 --p 5 --alpha 0.6 --total_mb 586  --lr 0.001 --local_bs 32
python onlineFL_step2_Online_FL-FINAL.py --dataset cifar10  --num_classes 10 --num_channels 3 --model vgg11 --total_slots 160 --T 10 --gpu 0 --p 5 --alpha 0.8 --total_mb 586  --lr 0.001 --local_bs 32
python onlineFL_step2_Online_FL-FINAL.py --dataset cifar10  --num_classes 10 --num_channels 3 --model vgg11 --total_slots 160 --T 10 --gpu 0 --p 5 --alpha 1.0 --total_mb 586  --lr 0.001 --local_bs 32





# T = 20
python onlineFL_step2_Online_FL-FINAL.py --dataset cifar10  --num_classes 10 --num_channels 3 --model vgg11 --total_slots 160 --T 20 --gpu 0 --p 20 --alpha 0.3 --total_mb 586  --lr 0.001 --local_bs 32
python onlineFL_step2_Online_FL-FINAL.py --dataset cifar10  --num_classes 10 --num_channels 3 --model vgg11 --total_slots 160 --T 20 --gpu 0 --p 20 --alpha 0.6 --total_mb 586  --lr 0.001 --local_bs 32
python onlineFL_step2_Online_FL-FINAL.py --dataset cifar10  --num_classes 10 --num_channels 3 --model vgg11 --total_slots 160 --T 20 --gpu 0 --p 20 --alpha 0.8 --total_mb 586  --lr 0.001 --local_bs 32
python onlineFL_step2_Online_FL-FINAL.py --dataset cifar10  --num_classes 10 --num_channels 3 --model vgg11 --total_slots 160 --T 20 --gpu 0 --p 20 --alpha 1.0 --total_mb 586  --lr 0.001 --local_bs 32

python onlineFL_step2_Online_FL-FINAL.py --dataset cifar10  --num_classes 10 --num_channels 3 --model vgg11 --total_slots 160 --T 20 --gpu 0 --p 10 --alpha 0.3 --total_mb 586  --lr 0.001 --local_bs 32
python onlineFL_step2_Online_FL-FINAL.py --dataset cifar10  --num_classes 10 --num_channels 3 --model vgg11 --total_slots 160 --T 20 --gpu 0 --p 10 --alpha 0.6 --total_mb 586  --lr 0.001 --local_bs 32
python onlineFL_step2_Online_FL-FINAL.py --dataset cifar10  --num_classes 10 --num_channels 3 --model vgg11 --total_slots 160 --T 20 --gpu 0 --p 10 --alpha 0.8 --total_mb 586  --lr 0.001 --local_bs 32
python onlineFL_step2_Online_FL-FINAL.py --dataset cifar10  --num_classes 10 --num_channels 3 --model vgg11 --total_slots 160 --T 20 --gpu 0 --p 10 --alpha 1.0 --total_mb 586  --lr 0.001 --local_bs 32

python onlineFL_step2_Online_FL-FINAL.py --dataset cifar10  --num_classes 10 --num_channels 3 --model vgg11 --total_slots 160 --T 20 --gpu 0 --p 5 --alpha 0.3 --total_mb 586  --lr 0.001 --local_bs 32
python onlineFL_step2_Online_FL-FINAL.py --dataset cifar10  --num_classes 10 --num_channels 3 --model vgg11 --total_slots 160 --T 20 --gpu 0 --p 5 --alpha 0.6 --total_mb 586  --lr 0.001 --local_bs 32
python onlineFL_step2_Online_FL-FINAL.py --dataset cifar10  --num_classes 10 --num_channels 3 --model vgg11 --total_slots 160 --T 20 --gpu 0 --p 5 --alpha 0.8 --total_mb 586  --lr 0.001 --local_bs 32
python onlineFL_step2_Online_FL-FINAL.py --dataset cifar10  --num_classes 10 --num_channels 3 --model vgg11 --total_slots 160 --T 20 --gpu 0 --p 5 --alpha 1.0 --total_mb 586  --lr 0.001 --local_bs 32



echo "一趟完整的实验执行完毕（24）"
