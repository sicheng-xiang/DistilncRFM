"""Four-task RiNALMo-Micro training and merged-LoRA inference benchmark."""
import argparse
import contextlib
from datetime import datetime
import hashlib
from itertools import islice
import json
import math
from pathlib import Path
import random
import sys
import time

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from tasks.rinalmo_downstream.data import load_splits,Collator
from tasks.rinalmo_downstream.metrics import calculate
from src.rinalmo_adapter.model import make_model,merge_lora

PRESETS={
 'family':dict(mode='lora',rank=32,alpha=64,dropout=.05,lr=1e-4,head_lr=1e-4,max_nt=510,epochs=100,selection='f1',pos_weight=1.,smoothing=0.),
 'localization':dict(mode='lora',rank=128,alpha=256,dropout=.3,lr=1e-5,head_lr=1e-5,max_nt=510,epochs=20,selection='f1',pos_weight=1.5,smoothing=.1),
 'splice':dict(mode='lora',rank=32,alpha=128,dropout=.1,lr=2e-4,head_lr=2e-4,max_nt=100,epochs=100,selection='avg_topk_acc',pos_weight=1.,smoothing=0.),
 'ires':dict(mode='full',rank=128,alpha=256,dropout=.2,lr=5e-6,head_lr=1e-4,max_nt=510,epochs=15,selection='auroc',pos_weight=2.5,smoothing=0.)}

def save(path,obj):path.write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
def resolve(path):
    p=Path(path).expanduser();return (p if p.is_absolute() else ROOT/p).resolve()
def sync(device):
    if device.type=='cuda':torch.cuda.synchronize(device)
def reset_peak(device):
    if device.type=='cuda':torch.cuda.reset_peak_memory_stats(device)
def memory(device):
    return dict(peak_allocated_MiB=torch.cuda.max_memory_allocated(device)/1024**2,
                peak_reserved_MiB=torch.cuda.max_memory_reserved(device)/1024**2) if device.type=='cuda' else dict(peak_allocated_MiB=0.,peak_reserved_MiB=0.)
def move(batch,device):return {k:v.to(device) for k,v in batch.items()}
def weight_denominator(labels,weights):
    valid=labels!=-100
    return weights[labels[valid]].sum() if weights is not None else valid.sum()
def loss_sum(logits,labels,weights,smoothing):
    return F.cross_entropy(logits.reshape(-1,logits.shape[-1]),labels.reshape(-1),weight=weights,
                           ignore_index=-100,label_smoothing=smoothing,reduction='sum')

def diagnostic_subset(rows,task,count=4):
    if task=='splice':
        selected=[]
        for label in [1,2]:
            index=next((i for i,r in enumerate(rows) if label in r['label'] and i not in selected),None)
            if index is not None:selected.append(index)
        selected+=list(i for i in range(len(rows)) if i not in selected)[:count-len(selected)]
        return [rows[i] for i in selected[:count]]
    groups={}
    for r in rows:groups.setdefault(r['label'],[]).append(r)
    selected=[]
    for offset in range(count):
        for group in groups.values():
            if offset<len(group):selected.append(group[offset])
            if len(selected)==count:return selected
    return selected

@torch.no_grad()
def evaluate(model,loader,device,task,weights,smoothing):
    model.eval();reset_peak(device);sync(device);start=time.perf_counter()
    predictions=[];labels_all=[];numerator=0.;denominator=0.
    for batch in loader:
        batch=move(batch,device);logits=model(batch['tokens'],batch['base_mask'])
        numerator+=float(loss_sum(logits,batch['labels'],weights,smoothing))
        denominator+=float(weight_denominator(batch['labels'],weights))
        # Original splice Trainer ranks raw class logits globally, not softmax probabilities.
        prob=(logits.float() if task=='splice' else logits.float().softmax(-1)).cpu().numpy()
        labels=batch['labels'].cpu().numpy()
        if task=='splice':
            # Pad variable-length batches before global top-k calculation.
            for p,y in zip(prob,labels):predictions.append(p[y!=-100]);labels_all.append(y[y!=-100])
        else:predictions.extend(prob);labels_all.extend(labels)
    sync(device);seconds=time.perf_counter()-start
    prob=np.concatenate(predictions) if task=='splice' else np.asarray(predictions)
    labels=np.concatenate(labels_all) if task=='splice' else np.asarray(labels_all)
    metrics=calculate(task,prob,labels);metrics.update(loss=numerator/max(denominator,1),seconds=seconds,**memory(device))
    return metrics,prob,labels

@torch.no_grad()
def benchmark(model,loader,device,warmup=3):
    model.eval()
    if device.type=='cuda':torch.cuda.empty_cache()
    first=next(iter(loader));first=move(first,device)
    for _ in range(warmup):model(first['tokens'],first['base_mask'])
    del first
    sync(device);reset_peak(device);start=time.perf_counter();samples=0;batches=0;forward_seconds=0.
    for batch in loader:
        sync(device);t=time.perf_counter();tokens=batch['tokens'].to(device);mask=batch['base_mask'].to(device)
        logits=model(tokens,mask);sync(device);forward_seconds+=time.perf_counter()-t
        samples+=tokens.shape[0];batches+=1
        del logits,tokens,mask
    sync(device);seconds=time.perf_counter()-start
    return dict(samples=samples,batches=batches,seconds=seconds,batches_per_second=batches/seconds,
        samples_per_second=samples/seconds,forward_and_transfer_seconds=forward_seconds,
        forward_and_transfer_samples_per_second=samples/forward_seconds,warmup_forwards=warmup,
        scope='loader/tokenization + host-to-device + model forward; excludes metrics and saving',**memory(device))

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--task',choices=PRESETS,required=True)
    p.add_argument('--weights',default='model/RiNALMo-Micro/rinalmo_micro_pretrained.pt')
    p.add_argument('--device',default='cuda:0');p.add_argument('--batch_size',type=int,default=16)
    p.add_argument('--gradient_accumulation_steps',type=int,default=2);p.add_argument('--eval_batch_size',type=int,default=16)
    p.add_argument('--epochs',type=int);p.add_argument('--lr',type=float);p.add_argument('--head_lr',type=float)
    p.add_argument('--schedule_epochs', type=int, help='Epoch budget for the splice LR schedule when timing a shorter run')
    p.add_argument('--gradient-checkpointing', action=argparse.BooleanOptionalAction, default=True,
                   help='Recompute Transformer activations during training backward')
    p.add_argument('--mode',choices=['lora','full']);p.add_argument('--rank',type=int);p.add_argument('--alpha',type=int)
    p.add_argument('--dropout',type=float);p.add_argument('--max_nt',type=int);p.add_argument('--seed',type=int,default=42)
    p.add_argument('--patience',type=int,default=5);p.add_argument('--validation_fraction',type=float,default=.1)
    p.add_argument('--pooling',choices=['cls','mean'],default='cls');p.add_argument('--output_dir')
    p.add_argument('--check_only',action='store_true');p.add_argument('--smoke_test',action='store_true')
    p.add_argument('--eval_checkpoint',help='Evaluate saved best_model.pt without training, then merged inference benchmark')
    p.add_argument('--threads',type=int,default=4);p.add_argument('--warmup_batches',type=int,default=3)
    p.add_argument('--pad_to_max',action=argparse.BooleanOptionalAction,default=True)
    a=p.parse_args(argv)
    for k,v in PRESETS[a.task].items():
        if not hasattr(a,k) or getattr(a,k) is None:setattr(a,k,v)
    if min(a.batch_size,a.eval_batch_size,a.gradient_accumulation_steps,a.epochs,a.max_nt,a.threads)<1: p.error('Positive batch sizes/epochs/length/threads required')
    if a.schedule_epochs is not None and (a.task != 'splice' or a.schedule_epochs < a.epochs):
        p.error('--schedule_epochs is only for splice and must be at least --epochs')
    if not 0<a.validation_fraction<1 or a.patience<0 or a.warmup_batches<0:p.error('Invalid validation_fraction/patience/warmup')
    if a.lr<=0 or a.head_lr<=0 or not 0<=a.dropout<1 or a.rank<1 or a.alpha<1:p.error('Invalid optimizer/LoRA settings')
    if a.task=='splice' and a.max_nt<100:p.error('Do not truncate splice benchmark labels below 100 nt')
    started=time.perf_counter();torch.set_num_threads(a.threads)
    random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed)
    train,val,test,meta=load_splits(a.task,a.seed,a.validation_fraction)
    if a.check_only:print(json.dumps(meta['counts']));return
    weights_path=resolve(a.weights)
    if not weights_path.is_file():raise FileNotFoundError(f'{weights_path}; run setup_rinalmo_linux.sh first')
    # Persist architecture/training metadata so evaluation cannot silently use a wrong LoRA rank.
    saved=None
    if a.eval_checkpoint:
        saved=torch.load(resolve(a.eval_checkpoint),map_location='cpu',weights_only=True)
        c=saved['config']
        if a.task!=c['task']:raise ValueError('Checkpoint task mismatch')
        for key in ['mode','rank','alpha','dropout','pooling','max_nt','pad_to_max','seed','validation_fraction']:
            setattr(a,key,c[key])
        train,val,test,meta=load_splits(a.task,a.seed,a.validation_fraction)
    if a.smoke_test:
        train,val,test=[diagnostic_subset(rs,a.task) for rs in [train,val,test]]
        a.epochs=1;a.batch_size=2;a.eval_batch_size=2
    meta['used_counts']={'train':len(train),'validation':len(val),'test':len(test)}
    meta['smoke_test']=a.smoke_test
    device=torch.device(a.device)
    if device.type=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
    tag='eval' if saved else 'smoke' if a.smoke_test else 'train'
    out=resolve(a.output_dir or f'results/rinalmo_micro/{a.task}_{tag}_{datetime.now():%Y%m%d_%H%M%S_%f}')
    if out.exists() and any(out.iterdir()):raise FileExistsError('Output directory must be new')
    out.mkdir(parents=True,exist_ok=True);save(out/'data_split.json',meta)
    nlabels=13 if a.task=='family' else 3 if a.task=='splice' else 2
    model,alphabet,targets=make_model(weights_path,nlabels,a.task,a.mode,a.rank,a.alpha,a.dropout,a.pooling)
    backbone = model.backbone.get_base_model() if a.mode == 'lora' else model.backbone
    backbone.transformer.gradient_checkpointing = a.gradient_checkpointing
    if saved:model.load_state_dict(saved['model'],strict=True)
    model=model.float().to(device)
    collate=Collator(alphabet,a.max_nt,a.task,a.pad_to_max)
    generator=torch.Generator().manual_seed(a.seed)
    loaders=[DataLoader(rows,batch_size=bs,shuffle=shuffle,collate_fn=collate,num_workers=0,
        generator=generator if shuffle else None) for rows,bs,shuffle in
        [(train,a.batch_size,True),(val,a.eval_batch_size,False),(test,a.eval_batch_size,False)]]
    cfg=vars(a).copy();cfg.update(precision='fp32',attention_backend='pytorch_sdpa',selection_split='validation',
        lora_target_modules=targets,total_parameters=sum(x.numel() for x in model.parameters()),
        trainable_parameters=sum(x.numel() for x in model.parameters() if x.requires_grad),
        evaluation_protocol='validation_selected; test used once after restoring best',
        base_sha256=hashlib.file_digest(weights_path.open('rb'),'sha256').hexdigest(),
        attention_mask=True)
    if a.task != 'ires':
        save(out/'run_config.json',cfg)
    print(json.dumps({k:cfg[k] for k in ['task','mode','rank','alpha','batch_size','gradient_accumulation_steps','precision','total_parameters','trainable_parameters']}),flush=True)
    class_weights=torch.tensor([1.,a.pos_weight],device=device) if nlabels==2 else None
    history=[];best=-math.inf;bad=0;train_seconds=0.;training_peak=0.;training_reserved_peak=0.;updates_total=0;micro_total=0;samples_total=0
    setup_seconds=time.perf_counter()-started;fit_start=time.perf_counter()
    if not saved:
        optimizer=torch.optim.AdamW([
            {'params':[x for x in model.backbone.parameters() if x.requires_grad],'lr':a.lr},
            {'params':model.head.parameters(),'lr':a.head_lr}],weight_decay=.05 if a.task=='localization' else .01)
        # Preserve splice launcher's 50-step warmup and linear Trainer schedule.
        scheduler=None
        if a.task=='splice':
            from transformers import get_linear_schedule_with_warmup
            total_updates=math.ceil(len(loaders[0])/a.gradient_accumulation_steps)*(a.schedule_epochs or a.epochs)
            scheduler=get_linear_schedule_with_warmup(optimizer,num_warmup_steps=50,num_training_steps=total_updates)
        cpu_weights=class_weights.cpu() if class_weights is not None else None
        for ep in range(1,a.epochs+1):
            epoch_start=time.perf_counter();model.train();optimizer.zero_grad(set_to_none=True)
            reset_peak(device);sync(device);t=time.perf_counter();total_loss=0.;total_denom=0.;updates=0;micro=0;samples=0
            iterator=iter(loaders[0]);bar=tqdm(total=len(train),desc=f'train {ep}/{a.epochs}')
            while True:
                group=list(islice(iterator,a.gradient_accumulation_steps))
                if not group:break
                # Normalize across the entire accumulation window, including a short final window.
                denom=sum(float(weight_denominator(b['labels'],cpu_weights)) for b in group)
                if denom<=0:raise ValueError('No supervised tokens in update')
                for batch in group:
                    batch=move(batch,device);logits=model(batch['tokens'],batch['base_mask'])
                    loss=loss_sum(logits,batch['labels'],class_weights,a.smoothing)
                    if not torch.isfinite(loss):raise FloatingPointError('Non-finite training loss')
                    (loss/denom).backward();total_loss+=float(loss.detach());micro+=1;samples+=batch['tokens'].shape[0]
                    bar.update(batch['tokens'].shape[0]);del loss,logits,batch
                torch.nn.utils.clip_grad_norm_([x for x in model.parameters() if x.requires_grad],1.,error_if_nonfinite=True)
                optimizer.step()
                if scheduler is not None:scheduler.step()
                optimizer.zero_grad(set_to_none=True);updates+=1;total_denom+=denom
            bar.close();sync(device);seconds=time.perf_counter()-t;peak=memory(device)
            training_peak=max(training_peak,peak['peak_allocated_MiB']);training_reserved_peak=max(training_reserved_peak,peak['peak_reserved_MiB'])
            train_seconds+=seconds;updates_total+=updates;micro_total+=micro;samples_total+=samples
            validation,_,_=evaluate(model,loaders[1],device,a.task,class_weights,a.smoothing)
            score=validation[a.selection];improved=score>best
            if improved:
                best=score;bad=0
                torch.save({'model':{k:v.detach().cpu() for k,v in model.state_dict().items()},
                    'config':cfg,'epoch':ep,'validation':validation},out/'best_model.pt')
            else:bad+=1
            row=dict(epoch=ep,train_loss=total_loss/max(total_denom,1),train_seconds=seconds,updates=updates,
                micro_batches=micro,samples=samples,update_batches_per_second=updates/seconds,
                micro_batches_per_second=micro/seconds,samples_per_second=samples/seconds,
                validation=validation,improved=improved,epochs_without_improvement=bad,
                epoch_total_seconds=time.perf_counter()-epoch_start,**peak)
            history.append(row);save(out/'history.json',history)
            print(f'Epoch {ep}: train {seconds:.2f}s, {updates/seconds:.3f} update batch/s, peak {peak}, validation={score:.6f}',flush=True)
            if a.patience and bad>=a.patience:break
        del optimizer,scheduler
        saved=torch.load(out/'best_model.pt',map_location='cpu',weights_only=True)
        model.load_state_dict(saved['model'],strict=True)
    for param in model.parameters():param.grad=None
    # Release optimizer state before inference; count only the deployed model's memory.
    if device.type=='cuda':torch.cuda.empty_cache()
    merged=merge_lora(model)
    test_metrics,prob,labels=evaluate(model,loaders[2],device,a.task,class_weights,a.smoothing)
    save(out/'test_metrics.json',dict(best_epoch=saved['epoch'],selected_on='validation',metrics=test_metrics,lora_merged=merged))
    np.savez_compressed(out/'test_predictions.npz',**{'logits' if a.task=='splice' else 'probabilities':prob,'labels':labels})
    fit_seconds=time.perf_counter()-fit_start
    if (ROOT/'tasks/inference_efficiency.py').is_file():
        sys.path.append(str(ROOT))
        from tasks.inference_efficiency import InferenceMeter
        if device.type=='cuda':
            torch.cuda.empty_cache()
        model._inference_lora_merged=merged
        checkpoint = resolve(a.eval_checkpoint) if a.eval_checkpoint else out/'best_model.pt'
        meter=InferenceMeter(model,device,a.eval_batch_size,a.warmup_batches,
            task=a.task,checkpoint=str(checkpoint),attention_backend='pytorch_sdpa')
        def prepare(batch):
            return {key:batch[key].to(device) for key in ('tokens','base_mask')}
        for _ in meter.predict(loaders[2],prepare,input_key='tokens'):
            pass
        inference=meter.save(out)
    else:
        inference=benchmark(model,loaders[2],device,a.warmup_batches)
        inference.update(lora_merged=merged,precision='fp32',batch_size=a.eval_batch_size,attention_backend='pytorch_sdpa')
        save(out/'inference_efficiency.json',inference)
    efficiency=dict(epochs=len(history),gradient_checkpointing=a.gradient_checkpointing,
        train_seconds=train_seconds,fit_seconds=fit_seconds,setup_seconds=setup_seconds,
        total_seconds_excluding_inference_benchmark=setup_seconds+fit_seconds,
        average_epoch_train_seconds=train_seconds/max(len(history),1),
        average_epoch_total_seconds=sum(h['epoch_total_seconds'] for h in history)/max(len(history),1),
        update_batches_per_second=updates_total/train_seconds if train_seconds else None,
        micro_batches_per_second=micro_total/train_seconds if train_seconds else None,
        samples_per_second=samples_total/train_seconds if train_seconds else None,
        train_peak_allocated_MiB=training_peak,train_peak_reserved_MiB=training_reserved_peak,
        validation_peak_allocated_MiB=max((h['validation']['peak_allocated_MiB'] for h in history),default=0.),
        validation_peak_reserved_MiB=max((h['validation']['peak_reserved_MiB'] for h in history),default=0.),
        effective_batch_size=a.batch_size*a.gradient_accumulation_steps,
        batch_definition='one parameter update; last update may contain fewer samples',
        memory_scope='PyTorch allocations in this process on the selected device; excludes other processes and driver-only allocations')
    if len(history)>1:
        efficiency['update_batches_per_second_excluding_first_epoch']=sum(h['updates'] for h in history[1:])/sum(h['train_seconds'] for h in history[1:])
    save(out/'training_efficiency.json',efficiency)
    print('Test:',json.dumps(test_metrics),'\nTraining efficiency:',json.dumps(efficiency),'\nInference efficiency:',json.dumps(inference),'\nResults:',out,flush=True)

if __name__=='__main__':main()
