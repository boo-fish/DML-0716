total_size.pkl中存储的是mnist数据集中训练集所占用的存储空间，向下取整。

这是本文所提方法的运行命令行： 
python3 dml_fl.py --dataset mnist  --num_channels 1 --model cnn --epochs 2 --gpu 0 --p 10 --ai 6 --alpha 0.3 
python3 dml_fl.py --dataset mnist  --num_channels 1 --model cnn --epochs 50 --gpu 0 --p 10 --ai 6 --alpha 0.3
python3 dml_fl.py --dataset mnist  --num_channels 1 --model cnn --epochs 50 --gpu 0 --p 10 --ai 6 --alpha 0.3 --iid
这里的alpha是非独立同分布程度系数，和算吞吐量，成本效率的缩放比α不是一个东西



这是传统联邦学习基准方法的命令行 
python3 dml_fl_benchmark.py --dataset mnist  --num_channels 1 --model cnn --epochs 50 --gpu 1 --p 10 --ai 6 --alpha 0.3
python3 dml_fl_benchmark.py --dataset mnist  --num_channels 1 --model cnn --epochs 50 --gpu 1 --p 10 --ai 6 --alpha 0.3 --iid



CADIB方法 已作废 
python3 dml_fl_CADIB.py --dataset mnist  --num_channels 1 --model cnn --epochs 50 --gpu 1 --p 10 --ai 15 --alpha 0.3
python3 dml_fl_CADIB.py --dataset mnist  --num_channels 1 --model cnn --epochs 50 --gpu 1 --p 10 --ai 15 --alpha 0.3 --iid

命令行中带iid表示为独立同分布，不带iid的表示为非独立同分布
