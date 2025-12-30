## cifar and vgg


###警告：一定要改mnist_get_offload_dict_dml.py中的参数为15p
python3 dml_fl.py --dataset cifar10  --num_channels 3 --model vgg11 --epochs 200 --gpu 0 --p 10 --ai 24 --alpha 0.3 --total_mb 586  --lr 0.001 --local_bs 32 --iid
python3 dml_fl.py --dataset cifar10  --num_channels 3 --model vgg11 --epochs 200 --gpu 0 --p 10 --ai 24 --alpha 0.3 --total_mb 586  --lr 0.001 --local_bs 32
python3 dml_fl_benchmark.py --dataset cifar10  --num_channels 3 --model vgg11 --epochs 200 --gpu 0 --p 10 --ai 24 --alpha 0.3 --total_mb 586  --lr 0.001 --local_bs 32 --iid
python3 dml_fl_benchmark.py --dataset cifar10  --num_channels 3 --model vgg11 --epochs 200 --gpu 0 --p 5 --ai 24 --alpha 0.3 --total_mb 586  --lr 0.001 --local_bs 32 --iid


python3 dml_fl.py --dataset cifar10  --num_channels 3 --model vgg11 --epochs 200 --gpu 0 --p 10 --ai 24 --alpha 0.3 --total_mb 586  --lr 0.001 --local_bs 32 --iid
python3 dml_fl.py --dataset cifar10  --num_channels 3 --model vgg11 --epochs 200 --gpu 0 --p 10 --ai 24 --alpha 0.3 --total_mb 586  --lr 0.001 --local_bs 32
python3 dml_fl_benchmark.py --dataset cifar10  --num_channels 3 --model vgg11 --epochs 200 --gpu 0 --p 10 --ai 24 --alpha 0.3 --total_mb 586  --lr 0.001 --local_bs 32 --iid
python3 dml_fl_benchmark.py --dataset cifar10  --num_channels 3 --model vgg11 --epochs 200 --gpu 0 --p 5 --ai 24 --alpha 0.3 --total_mb 586  --lr 0.001 --local_bs 32 --iid




python3 dml_fl.py --dataset cifar10  --num_channels 3 --model vgg11 --epochs 100 --gpu 0 --p 5 --ai 24 --alpha 0.3 --total_mb 586  --lr 0.001 --local_bs 32
python3 dml_fl.py --dataset cifar10  --num_channels 3 --model vgg11 --epochs 100 --gpu 0 --p 5 --ai 24 --alpha 0.5 --total_mb 586  --lr 0.001 --local_bs 32
python3 dml_fl.py --dataset cifar10  --num_channels 3 --model vgg11 --epochs 100 --gpu 0 --p 5 --ai 24 --alpha 0.8 --total_mb 586  --lr 0.001 --local_bs 32
python3 dml_fl.py --dataset cifar10  --num_channels 3 --model vgg11 --epochs 100 --gpu 0 --p 5 --ai 24 --alpha 1.0 --total_mb 586  --lr 0.001 --local_bs 32

python3 dml_fl.py --dataset cifar10  --num_channels 3 --model vgg11 --epochs 100 --gpu 0 --p 3 --ai 24 --alpha 0.3 --total_mb 586  --lr 0.001 --local_bs 32
python3 dml_fl.py --dataset cifar10  --num_channels 3 --model vgg11 --epochs 100 --gpu 0 --p 3 --ai 24 --alpha 0.5 --total_mb 586  --lr 0.001 --local_bs 32
python3 dml_fl.py --dataset cifar10  --num_channels 3 --model vgg11 --epochs 100 --gpu 0 --p 3 --ai 24 --alpha 0.8 --total_mb 586  --lr 0.001 --local_bs 32
python3 dml_fl.py --dataset cifar10  --num_channels 3 --model vgg11 --epochs 100 --gpu 0 --p 3 --ai 24 --alpha 1.0 --total_mb 586  --lr 0.001 --local_bs 32









