import os 

import numpy as np
import pandas as pd
from Bio import SeqIO
import pickle

from torch.utils.data import Dataset


class RNADataset(Dataset):
    def __init__(self, data_dir, dataset_type, prefix, tokenizer, seed=0, train=True, max_nt_exclusive=None):
        """Load RNA sequences and labels for lncRNA, miRNA, or circRNA. max_nt_exclusive optionally retains sequences shorter than the specified nucleotide length. prefix selects the train or test input split."""
        super(RNADataset, self).__init__()

        self.tokenizer = tokenizer
        self.seed = seed
        self.dataset_type = dataset_type
        
        # Map dataset names to input files.
        file_mapping = {
            "lncRNA": ("lncTrain.pkl", "lncTest.pkl"),
            "miRNA": ("miTrain.pkl", "miTest.pkl"),
            
            "circRNA": ("circTrain.pkl", "circTest.pkl")
        }
        
        train_file, test_file = file_mapping[dataset_type]
        pkl_file = os.path.join(data_dir, train_file if train else test_file)
        
        # Load serialized dataset records.
        with open(pkl_file, 'rb') as f:
            seqs = pickle.load(f)
            names = pickle.load(f)
            labels = pickle.load(f)
        
        # Map labels for the selected dataset.
        self.data = []
        if dataset_type in ("lncRNA", "circRNA"):
            for seq, name, label in zip(seqs, names, labels):
                label_str = "nucleus" if label == 0 else "cytoplasm"
                self.data.append((seq, name, label_str))
        elif dataset_type == "miRNA":
            # Labels: extracellular=0, intracellular=1.
            for seq, name, label in zip(seqs, names, labels):
                label_str = "extracellular" if label == 0 else "intracellular"
                self.data.append((seq, name, label_str))
        
        
        
        if max_nt_exclusive is not None:
            before = len(self.data)
            self.data = [item for item in self.data if len(item[0]) < max_nt_exclusive]
            split_name = "train" if train else "test"
            print(f"{dataset_type} {split_name}: nucleotide length < {max_nt_exclusive}; "
                  f"before filtering:  {before}; retained:  {len(self.data)}; removed:  {before - len(self.data)}")
            if not self.data:
                raise ValueError(f"{dataset_type} {split_name} 筛选后为空")

        # Shuffle examples.
        np_rng = np.random.RandomState(seed=seed)
        np_rng.shuffle(self.data)
        
        
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


# Retain the LncRNADataset interface.
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

        # Select the training or test split.
        split = "train" if train else "test"
        
        # Paths for positive and negative examples.
        pos_fa = osp.join(fasta_dir, split, "positive", f"positive_{split}_data.fa")
        neg_fa = osp.join(fasta_dir, split, "negative", f"negative_{split}_data.fa")

        
        self.data = []

        # Positive label = "1".
        for x in SeqIO.parse(pos_fa, "fasta"):
            seq = str(x.seq)
            name = x.description.split(" ")[0]
            label = "positive" 
            self.data.append((seq, name, label))

        # Negative label = "0".
        for x in SeqIO.parse(neg_fa, "fasta"):
            seq = str(x.seq)
            name = x.description.split(" ")[0]
            label = "negative"
            self.data.append((seq, name, label))

        # Shuffle examples.
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
