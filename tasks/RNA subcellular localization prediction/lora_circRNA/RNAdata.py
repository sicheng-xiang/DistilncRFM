import os 

import numpy as np
import pandas as pd
from Bio import SeqIO
import pickle

from torch.utils.data import Dataset


class RNADataset(Dataset):
    def __init__(self, data_dir, dataset_type, prefix, tokenizer, seed=0, train=True, max_nt_exclusive=None):
        """
        RNA数据集加载器
        data_dir: 数据目录
        dataset_type: 'lncRNA'、'miRNA' 或 'circRNA'
        max_nt_exclusive: 只保留核苷酸长度严格小于此值的序列；None 不筛选
        prefix: 'train' 或 'test'
        tokenizer: 分词器
        train: True=训练集, False=测试集
        """
        super(RNADataset, self).__init__()

        self.tokenizer = tokenizer
        self.seed = seed
        self.dataset_type = dataset_type
        
        # 数据文件映射
        file_mapping = {
            "lncRNA": ("lncTrain.pkl", "lncTest.pkl"),
            "miRNA": ("miTrain.pkl", "miTest.pkl"),
            # 修改原因：支持 circRNA 独立数据文件；修改时间：2026-10-14
            "circRNA": ("circTrain.pkl", "circTest.pkl")
        }
        
        train_file, test_file = file_mapping[dataset_type]
        pkl_file = os.path.join(data_dir, train_file if train else test_file)
        
        # 加载pickle数据
        with open(pkl_file, 'rb') as f:
            seqs = pickle.load(f)
            names = pickle.load(f)
            labels = pickle.load(f)
        
        # 根据数据集类型处理标签
        self.data = []
        if dataset_type in ("lncRNA", "circRNA"):
            for seq, name, label in zip(seqs, names, labels):
                label_str = "nucleus" if label == 0 else "cytoplasm"
                self.data.append((seq, name, label_str))
        elif dataset_type == "miRNA":
            # 标签映射: 0=extracellular(细胞外), 1=intracellular(细胞内)
            for seq, name, label in zip(seqs, names, labels):
                label_str = "extracellular" if label == 0 else "intracellular"
                self.data.append((seq, name, label_str))
        
        # 修改原因：按原始 nt 长度筛选，避免将截断后的长序列当作短序列；修改时间：2026-10-14
        # 默认不筛选，保持其他训练脚本的原有行为。
        if max_nt_exclusive is not None:
            before = len(self.data)
            self.data = [item for item in self.data if len(item[0]) < max_nt_exclusive]
            split_name = "train" if train else "test"
            print(f"{dataset_type} {split_name}: nt长度 < {max_nt_exclusive}，"
                  f"筛选前 {before}，保留 {len(self.data)}，移除 {before - len(self.data)}")
            if not self.data:
                raise ValueError(f"{dataset_type} {split_name} 筛选后为空")

        # 随机打乱
        np_rng = np.random.RandomState(seed=seed)
        np_rng.shuffle(self.data)
        
        # 打印数据集信息
        label_counts = {}
        for item in self.data:
            label = item[2]
            label_counts[label] = label_counts.get(label, 0) + 1
        
        split_name = "train" if train else "test"
        print(f"\n=== {dataset_type} {split_name} set info ===")
        print(f"Total samples: {len(self.data)}")
        for label, count in label_counts.items():
            print(f"  {label}: {count}")
        print()

    def __getitem__(self, idx):
        instance = self.data[idx]
        seq = instance[0]
        name = instance[1]
        label = instance[2]
        return {"seq": seq, "label": label, 'name': name}

    def __len__(self):
        return len(self.data)


# 保留原LncRNADataset以兼容
class LncRNADataset(RNADataset):
    def __init__(self, data_dir, prefix, tokenizer, seed=0, train=True):
        super().__init__(data_dir, "lncRNA", prefix, tokenizer, seed, train)

    def __getitem__(self, idx):
        instance = self.data[idx]
        seq = instance[0]
        name = instance[1]
        label = instance[2]
        return {"seq": seq, "label": label, 'name': name}

    def __len__(self):
        return len(self.data)


class IRESDataset(Dataset):
    def __init__(self, fasta_dir, prefix, tokenizer, seed=0, train=True):
        super(IRESDataset, self).__init__()

        self.fasta_dir = fasta_dir
        self.prefix = prefix
        self.tokenizer = tokenizer

        # 区分 train / test
        split = "train" if train else "test"
        
        # 正负样本文件路径
        pos_fa = osp.join(fasta_dir, split, "positive", f"positive_{split}_data.fa")
        neg_fa = osp.join(fasta_dir, split, "negative", f"negative_{split}_data.fa")

        # 格式完全对齐原代码：(seq, name, label)
        self.data = []

        # 正样本 label = "1"
        for x in SeqIO.parse(pos_fa, "fasta"):
            seq = str(x.seq)
            name = x.description.split(" ")[0]
            label = "positive" 
            self.data.append((seq, name, label))

        # 负样本 label = "0"
        for x in SeqIO.parse(neg_fa, "fasta"):
            seq = str(x.seq)
            name = x.description.split(" ")[0]
            label = "negative"
            self.data.append((seq, name, label))

        # 随机打乱
        np_rng = np.random.RandomState(seed=seed)
        np_rng.shuffle(self.data)

    def __getitem__(self, idx):
        instance = self.data[idx]
        seq = instance[0]
        name = instance[1]
        label = instance[2]
        return {"seq": seq, "label": label, 'name': name}

    def __len__(self):
        return len(self.data)


class SeqClsDataset(Dataset):
    def __init__(self, fasta_dir, prefix, tokenizer, seed=0, train=True):
        super(SeqClsDataset, self).__init__()

        self.fasta_dir = fasta_dir
        self.prefix = prefix
        self.tokenizer = tokenizer

        file_name = "train.fa" if train else "test.fa"
        fasta = osp.join(osp.join(fasta_dir, prefix), file_name)
        records = list(SeqIO.parse(fasta, "fasta"))
        self.data = [(str(x.seq), *x.description.split(" ")[0:2]) for x in records]
        np_rng = np.random.RandomState(seed=seed)
        np_rng.shuffle(self.data)

    def __getitem__(self, idx):
        instance = self.data[idx]
        seq = instance[0]
        name = instance[1]
        label = instance[2]
        return {"seq": seq, "label": label, 'name': name}

    def __len__(self):
        return len(self.data)
