"""Portable readers for the same four local datasets; no retokenized label files."""
from pathlib import Path
import csv
import hashlib
import pickle
import random
import torch

ROOT=Path(__file__).resolve().parents[2]
LABELS=['5S_rRNA','5_8S_rRNA','tRNA','ribozyme','CD-box','Intron_gpI','Intron_gpII',
        'riboswitch','IRES','HACA-box','scaRNA','leader','miRNA']

def fasta(path):
    name=None;seq=[]
    for line in Path(path).read_text(encoding='utf-8-sig').splitlines():
        if line.startswith('>'):
            if name is not None:yield name,''.join(seq)
            name=line[1:];seq=[]
        elif line.strip():seq.append(line.strip())
    if name is not None:yield name,''.join(seq)

def load_raw(task,split):
    rows=[]
    if task=='family':
        for name,seq in fasta(ROOT/f'data/data/nRC/{split}.fa'):
            label=name.split()[1]
            rows.append(dict(name=name.split()[0],seq=seq,label=LABELS.index(label)))
    elif task=='localization':
        path=ROOT/'tasks/RNA subcellular localization prediction/RNALoc-LM-main/data'/f'circ{split.title()}.pkl'
        with path.open('rb') as f:seqs=pickle.load(f);names=pickle.load(f);labels=pickle.load(f)
        if not len(seqs)==len(names)==len(labels):raise ValueError('Pickle lengths differ')
        rows=[dict(name=str(n),seq=str(s),label=int(y)) for s,n,y in zip(seqs,names,labels) if len(s)<512]
    elif task=='ires':
        for cls,label in [('negative',0),('positive',1)]:
            path=ROOT/f'data/data/ires_seq/{split}/{cls}/{cls}_{split}_data.fa'
            rows.extend(dict(name=name.split()[0],seq=seq,label=label) for name,seq in fasta(path))
    elif task=='splice':
        with (ROOT/f'data/data/SPL/{split}.csv').open(encoding='utf-8-sig',newline='') as f:
            for i,row in enumerate(csv.DictReader(f)):
                seq=row['sequence'];labels=[int(x) for x in row['labels']]
                if len(seq)!=len(labels) or any(y not in [0,1,2] for y in labels):raise ValueError(f'Invalid splice labels: {split}:{i}')
                rows.append(dict(name=f'{split}:{i}',seq=seq,label=labels))
    else:raise ValueError(task)
    return rows

def key(row):return hashlib.sha256(row['seq'].upper().replace('U','T').encode()).hexdigest()

def load_splits(task,seed=42,validation_fraction=.1):
    train=load_raw(task,'train');test=load_raw(task,'test')
    if task=='splice':val=load_raw(task,'val');method='provided val.csv'
    else:
        # Group identical sequences; stratify groups per label for stable class coverage.
        groups={}
        for r in train:groups.setdefault(key(r),[]).append(r)
        strata={}
        for k,rs in groups.items():strata.setdefault(rs[0]['label'],[]).append(k)
        rng=random.Random(seed);selected=set()
        for keys in strata.values():
            keys.sort();rng.shuffle(keys)
            count=min(len(keys)-1,max(1,round(len(keys)*validation_fraction)))
            selected.update(keys[:count])
        val=[r for r in train if key(r) in selected];train=[r for r in train if key(r) not in selected]
        method='seeded sequence-grouped stratified holdout from original training split'
    if not train or not val or not test:raise ValueError('Empty data split')
    meta=dict(task=task,seed=seed,validation_fraction=validation_fraction,validation_method=method,
        counts={n:len(rs) for n,rs in [('train',train),('validation',val),('test',test)]},
        names={n:[r['name'] for r in rs] for n,rs in [('train',train),('validation',val),('test',test)]},
        train_test_exact_sequence_overlap=len({key(r) for r in train}&{key(r) for r in test}),
        train_validation_exact_sequence_overlap=len({key(r) for r in train}&{key(r) for r in val}))
    return train,val,test,meta

class Collator:
    def __init__(self,alphabet,max_nt=510,task='family',pad_to_max=False):
        self.alphabet=alphabet;self.max_nt=max_nt;self.task=task;self.pad_to_max=pad_to_max
    def __call__(self,rows):
        seqs=[r['seq'][:self.max_nt] for r in rows]
        n=self.max_nt if self.pad_to_max else max(map(len,seqs))
        tokens=torch.tensor([self.alphabet.encode(s,pad_to_len=n+2) for s in seqs],dtype=torch.long)
        mask=torch.arange(n)[None,:]<torch.tensor([len(s) for s in seqs])[:,None]
        if self.task=='splice':
            labels=torch.full((len(rows),n),-100,dtype=torch.long)
            for i,(r,s) in enumerate(zip(rows,seqs)):labels[i,:len(s)]=torch.tensor(r['label'][:len(s)])
        else:labels=torch.tensor([r['label'] for r in rows],dtype=torch.long)
        return dict(tokens=tokens,base_mask=mask,labels=labels)
