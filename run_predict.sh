# 激活虚拟环境
# 请按照步骤自行安装anaconda以及配置虚拟环境
conda activate fold_spi

# 请将数据放在./data_prepare/my_data中，以便后续的操作顺利
# 处理SPI数据
# 请将从STRING中下载相互作用的数据命名为input_PPI.txt
# 请将从STRING中下载的序列数据命名为sequence.fasta
# 生成过滤后的SPI相互作用文件filter_SPI.txt，蛋白质序列文件protein.fasta，SEP序列文件SEP.fasta
python ./data_prepare/filter_SPI.py --input_PPI ./data_prepare/my_data/input_PPI.txt --seq_file ./data_prepare/my_data/sequence.fasta --output_SPI ./data_prepare/my_data/filter_SPI.txt --output_pro_seq ./data_prepare/my_data/protein.fasta --output_SEP_seq ./data_prepare/my_data/SEP.fasta

# 获取数据集
mkdir ./data_prepare/my_data/SPI
python interaction_to_all.py -- data_dir ./data_prepare/my_data

# 划分训练集和测试集
mkdir ./data_prepare/my_data/split_by_SEP
python split_train_test.py -- data_dir ./data_prepare/my_data

# 创建用于计算PSSM数据的序列文件
mkdir ./data_prepare/my_data/SEP_fasta
mkdir ./data_prepare/my_data/protein_fasta
python ./data_prepare/get_fasta.py --SEP_file ./data_prepare/my_data/SEP.fasta --protein_file ./data_prepare/my_data/protein.fasta --output_dir ./data_prepare/my_data

# 根据流程安装BLAST软件，以计算PSSM，并将计算结果放在./data_prepare/my_data/protein_PSSM和./data_prepare/my_data/SEP_PSSM中
mkdir ./data_prepare/my_data/SEP_PSSM
mkdir ./data_prepare/my_data/protein_PSSM
python ./data_prepare/PSSM_feature.py --seq_file ./data_prepare/my_data/SEP.fasta --PSSM_dir ./data_prepare/my_data/SEP_PSSM --output_pssm_dict ./data_prepare/my_data/SEP_pssm
python ./data_prepare/PSSM_feature.py --seq_file ./data_prepare/my_data/protein.fasta --PSSM_dir ./data_prepare/my_data/protein_PSSM --output_pssm_dict ./data_prepare/my_data/protein_pssm

# 计算蛋白质和SEP的结构信息
mkdir ./data_prepare/my_data/SEP_PDB
esm-fold -i ./data_prepare/my_data/SEP.fasta -o ./data_prepare/my_data/SEP_PDB
mkdir ./data_prepare/my_data/protein_PDB
esm-fold -i ./data_prepare/my_data/protein.fasta -o ./data_prepare/my_data/protein_PDB

# 提取结构特征
# 根据流程安装ScanNet，以计算结构特征，并将结果放在./data_prepare/my_data/protein_ScanNet和./data_prepare/my_data/SEP_ScanNet中
mkdir ./data_prepare/my_data/SEP_ScanNet
mkdir ./data_prepare/my_data/protein_ScanNet

# 提取内在紊乱信息
# 根据流程在IUPred3在线网站中，提取特征，并将结果放在./data_prepare/my_data/中，并命名为SEP_short.txt、SEP_long.txt、protein_short.txt和protein_long.txt
python ./data_prepare/intrinsic_feature.py -- fasta_filename ./data_prepare/my_data/SEP --output_intrisic_dict ./data_prepare/my_data/SEP_intrinsic
python ./data_prepare/intrinsic_feature.py -- fasta_filename ./data_prepare/my_data/protein --output_intrisic_dict ./data_prepare/my_data/protein_intrinsic

# 整合全部的特征
mkdir ./data_prepare/my_data/feature
python ./data_prepare/preprocess_features.py --data_dir ./data_prepare/my_data

# 训练模型
python ./train/train.py --feature_dir ./data_prepare/my_data/feature --train_file ../data_prepare/my_data/split_by_SEP/train_SEP_1.txt --test_file ../data_prepare/my_data/split_by_SEP/train_SEP_1.txt --data_name my_data

# 可以在result文件夹中查看结果