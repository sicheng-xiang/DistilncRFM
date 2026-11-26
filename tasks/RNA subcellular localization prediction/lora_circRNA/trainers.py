#trainer.py
import os
import time
import math
from itertools import islice
from tqdm import tqdm

import torch

from base_classes import BaseTrainer
from collections import defaultdict

import pandas as pd
import numpy as np
from metrics import compare_bpseq


class SeqClsTrainer(BaseTrainer):
    def __init__(self, args, tokenizer, model, train_dataset, eval_dataset, data_collator, loss_fn, optimizer, compute_metrics, lr_scheduler=None):
        super().__init__(
            args=args,
            tokenizer=tokenizer,
            model=model,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            loss_fn=loss_fn,
            optimizer=optimizer,
            compute_metrics=compute_metrics
        )
        self.metric = compute_metrics
        self.max_metric = 0.0
        self.best_epoch = 0
        self.lr_scheduler = lr_scheduler

    def train(self, epoch):
        self.model.train()
        time_st = time.time()
        accum = getattr(self.args, 'gradient_accumulation_steps', 1)
        steps = samples = micro_batches = 0
        loss_total = denominator_total = 0.0
        # Weighted CE (also with label smoothing) divides by the sum of target-class
        # weights, not sample count. Match the loss of one effective batch exactly.
        weights = getattr(self.loss_fn, 'pos_weight', None)
        if weights is not None:
            weights = weights.detach().cpu()
        def denominator(data):
            labels = data['labels'].long().cpu()
            return float(weights[labels].sum()) if weights is not None else len(labels)
        batches = iter(self.train_dataloader)
        self.optimizer.zero_grad(set_to_none=True)
        with tqdm(total=len(self.train_dataset), disable=self.args.disable_tqdm) as pbar:
            for start in range(0, len(self.train_dataloader), accum):
                group = list(islice(batches, accum))
                group_denominator = sum(denominator(data) for data in group)
                for data in group:
                    labels = data['labels'].to(self.args.device).long()
                    outputs = self.model(data['input_ids'].to(self.args.device))
                    logits = outputs.logits if hasattr(outputs, 'logits') else outputs
                    loss = self.loss_fn(logits, labels)
                    batch_denominator = denominator(data)
                    (loss * (batch_denominator / group_denominator)).backward()
                    value = loss.item()
                    if not math.isfinite(value):
                        raise FloatingPointError('Non-finite training loss')
                    loss_total += value * batch_denominator
                    denominator_total += batch_denominator
                    samples += len(labels)
                    micro_batches += 1
                    pbar.update(len(labels))
                    pbar.set_postfix(train_loss=f'{loss_total / denominator_total:.4f}')
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)
                steps += 1
                if self.lr_scheduler is not None:
                    self.lr_scheduler.step()
        current_lr = self.optimizer.param_groups[0]['lr']
        time_ed = time.time() - time_st
        mean_loss = loss_total / denominator_total
        print('Train\tMean loss: {:.6f}; Time: {:.4f}s; LR: {:.2e}'.format(mean_loss, time_ed, current_lr))
        return dict(train_steps=steps, train_micro_batches=micro_batches,
                    train_samples=samples, train_loss=mean_loss)

    def eval(self, epoch):
        self.model.eval()
        time_st = time.time()
        num_total = 0
        with tqdm(total=len(self.eval_dataset), disable=self.args.disable_tqdm) as pbar:
            outputs_dataset, labels_dataset = [], []
            outputs_names, outputs_seqs = [], []
            for i, data in enumerate(self.eval_dataloader):
                input_ids = data["input_ids"].to(self.args.device)
                labels = data["labels"].to(self.args.device)

                with torch.no_grad():
                    outputs = self.model(input_ids)
                    logits = outputs.logits if hasattr(outputs, 'logits') else outputs

                num_total += self.args.batch_size
                outputs_dataset.append(logits)
                labels_dataset.append(labels)
                outputs_names += data['name']
                outputs_seqs += data['seq']

                if num_total >= self.args.logging_steps:
                    pbar.update(self.args.logging_steps)
                    num_total = 0

        outputs_dataset = torch.concat(outputs_dataset, axis=0)
        labels_dataset = torch.concat(labels_dataset, axis=0)
        
        if hasattr(self.compute_metrics, 'compute'):
            self.compute_metrics(outputs_dataset, labels_dataset)
            metrics_dataset = self.compute_metrics.compute()
        else:
            metrics_dataset = self.compute_metrics(outputs_dataset, labels_dataset)
        
        if hasattr(self.metric, 'compute_auc_aupr'):
            auc_aupr = self.metric.compute_auc_aupr()
            metrics_dataset.update(auc_aupr)
        
        self.save_model(metrics_dataset, epoch)
        df = pd.DataFrame({
            "name": outputs_names,
            "seq": outputs_seqs,
            "true_label": labels_dataset.detach().cpu().numpy(),
            **{f"class_{i}_logit": outputs_dataset[:, i].detach().cpu().numpy() for i in range(outputs_dataset.shape[1])},
            "predicted_class": np.argmax(outputs_dataset.detach().cpu().numpy(), axis=1),
        })
        df.to_csv(os.path.join(self.args.output_dir, f'test_metric_{epoch}_{self.max_metric:.4f}.csv'), index=False)

        results = {}
        log = 'Test\t' + self.args.dataset + "\t"
        tag_value = defaultdict(float)
        for k, v in metrics_dataset.items():
            log += k + ": {" + k + ":.4f}\t"
            results[k] = v
            tag = "eval/" + k
            tag_value[tag] = v

        time_ed = time.time() - time_st
        print(log.format(**results), "; Time: {:.4f}s".format(time_ed))
        self.compute_metrics.reset()

    def save_model(self, metrics, epoch):
        if self.args.save_strategy not in ["epoch", True]:
            return
    
        # 保存f1s最高的模型
        current_metric = metrics.get("f1s", 0.0)
    
        best_f1s = self.max_metric
    
        if current_metric > best_f1s:
            self.max_metric = current_metric
            self.best_epoch = epoch
            if not getattr(self.args, 'save_checkpoints', True):
                return
    
            auc_val = metrics.get("auc", 0.0)
            f1_val = metrics.get("f1s", 0.0)
    
            model_name = f"best_lora_f1{f1_val:.4f}_auc{auc_val:.4f}"
            save_dir = os.path.join(self.args.output_dir, model_name)
    
            os.makedirs(save_dir, exist_ok=True)
            try:
                if isinstance(self.model, torch.nn.DataParallel):
                    self.model.module.save_pretrained(save_dir, safe_serialization=False)
                else:
                    self.model.save_pretrained(save_dir, safe_serialization=False)
                print(f"新最优模型！Epoch {epoch} | F1: {f1_val:.4f}")
                print(f"已保存至: {save_dir}")
            except Exception as e:
                print(f"保存失败: {e}")


class RRInterTrainer(BaseTrainer):
    def train(self, epoch):
        self.model.train()
        time_st = time.time()
        num_total, loss_total = 0, 0

        with tqdm(total=len(self.train_dataset), disable=False) as pbar:
            for i, data in enumerate(self.train_dataloader):
                tokens = data["tokens"].to(self.args.device)
                input_ids = data["input_ids"].to(self.args.device)
                labels = data["labels"].to(self.args.device)

                preds = self.model(tokens, input_ids)
                loss = self.loss_fn(preds, labels)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                num_total += self.args.batch_size
                loss_total += loss.item()

                if num_total >= self.args.logging_steps:
                    pbar.set_postfix(train_loss='{:.4f}'.format(
                        loss_total / num_total))
                    pbar.update(self.args.logging_steps)
                    num_total, loss_total = 0, 0

        time_ed = time.time() - time_st
        print('Train\tLoss: {:.6f}; Time: {:.4f}s'.format(loss.item(), time_ed))

    def eval(self, epoch):
        self.model.eval()
        time_st = time.time()

        with tqdm(total=len(self.eval_dataset), disable=True) as pbar:
            names_dataset, outputs_dataset, labels_dataset = [], [], []
            for i, data in enumerate(self.eval_dataloader):
                names = data["names"]
                tokens = data["tokens"].to(self.args.device)
                input_ids = data["input_ids"].to(self.args.device)
                labels = data["labels"].to(self.args.device)

                with torch.no_grad():
                    output = self.model(tokens, input_ids)

                names_dataset += names
                outputs_dataset.append(output)
                labels_dataset.append(labels)
                pbar.update(self.args.batch_size)

            outputs_dataset = torch.concat(outputs_dataset, axis=0)
            labels_dataset = torch.concat(labels_dataset, axis=0)
            metrics_dataset = self.compute_metrics(outputs_dataset, labels_dataset)
            self.save_model(metrics_dataset, epoch)

            preds = outputs_dataset.detach().cpu().numpy()
            df = pd.DataFrame({'pred0': preds[:, 0], 'pred1': preds[:,1], 'label': labels_dataset.detach().cpu().numpy()})
            key_name = [('F1', 'f1s'), ('auc', 'auc'), ('P', 'precision'), ('R', 'recall'), ('acc', 'accuracy')]
            label = '_'.join([f'{key}{metrics_dataset[name]:.4f}' for key, name in key_name])
            df.to_csv(os.path.join(self.args.output_dir, f'test_metric_{epoch}_{label}.csv'), index=False)

        results = {}
        log = 'Test\t' + self.args.dataset + "\t"
        tag_value = defaultdict(float)
        for k, v in metrics_dataset.items():
            log += k + ": {" + k + ":.4f}\t"
            results[k] = v
            tag = "eval/" + k
            tag_value[tag] = v

        time_ed = time.time() - time_st
        print(log.format(**results), "; Time: {:.4f}s".format(time_ed))


class SspTrainer(BaseTrainer):
    def __init__(self, args, tokenizer, model, pretrained_model=None,
                 indicator=None, train_dataset=None, eval_dataset=None,
                 data_collator=None,
                 loss_fn=None,
                 optimizer=None,
                 compute_metrics=None,
                 visual_writer=None):

        super(SspTrainer, self).__init__(
            args=args,
            tokenizer=tokenizer,
            model=model,
            pretrained_model=pretrained_model,
            indicator=indicator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            loss_fn=loss_fn,
            optimizer=optimizer,
            compute_metrics=compute_metrics,
            visual_writer=visual_writer
        )
        self.is_break = False

    def get_status(self):
        return self.is_break

    def train(self, epoch):

        self.model.train()
        time_st = time.time()
        loss_total, num_total = 0, 0

        with tqdm(total=len(self.train_dataset), disable=self.args.disable_tqdm) as pbar:
            for i, instance in enumerate(self.train_dataloader):

                headers = instance["names"]
                refs = (instance["labels"], )
                seqs = (instance["seqs"][0], )
                input_ids = (instance["input_ids"], )

                input_tensor = torch.tensor(
                    input_ids[0]).unsqueeze(0).to(self.args.device)

                if instance["input_ids"].shape[0] < 600:
                    with torch.no_grad():
                        embeddings = self.pretrained_model(input_tensor)
                    self.optimizer.zero_grad()
                    loss = self.loss_fn(
                        self.model, seqs, refs, embeddings, fname=headers)
                    if loss.item() > 0.:
                        loss.backward()
                        self.optimizer.step()
                    if torch.isnan(loss):
                        self.is_break = True
                else:
                    print('training without len>=600', headers)
                num_total += self.args.batch_size
                loss_total += loss.item()
                pbar.set_postfix(train_loss='{:.3e}'.format(
                    loss_total / num_total))
                pbar.update(self.args.batch_size)
                if num_total >= self.args.logging_steps:
                    loss_total, num_total = 0, 0

        elapsed_time = time.time() - time_st
        print('Train Epoch: {}\tTime: {:.3f}s'.format(epoch, elapsed_time))

    def eval(self, epoch):
        self.model.eval()
        n_dataset = len(self.eval_dataloader.dataset)
        with tqdm(total=n_dataset) as pbar:
            res = defaultdict(list)
            metrics_dataset = {}
            for instance in self.eval_dataloader:
                headers = instance["names"]
                refs = (instance["labels"], )
                seqs = (instance["seqs"][0], )
                input_ids = (instance["input_ids"], )
                seqs = (seqs[0], )
                refs = (refs[0], )
                input_tensor = torch.from_numpy(
                    input_ids[0]).unsqueeze(0).to(self.args.device)
                if instance["input_ids"].shape[0] < 600:

                    with torch.no_grad():
                        embeddings = self.pretrained_model(input_tensor)
                    scs, preds, bps = self.model(seqs, embeddings)
                    for header, seq, ref, sc, pred, bp in zip(headers, seqs, refs, scs, preds, bps):
                        x = compare_bpseq(ref, bp)
                        ret = self.compute_metrics(*x)
                        for k, v in ret.items():
                            res[k].append(v)

                avg_val = np.mean(res[self.name_pbar])
                pbar.set_postfix(eval_fls='{:.3e}'.format(avg_val))
                metrics_dataset[self.name_pbar] = avg_val
                pbar.update(self.args.batch_size)

            self.save_model(metrics_dataset, epoch)
            df = pd.DataFrame(res)
            df.to_csv(os.path.join(self.args.output_dir, f'test_metric_{epoch}_{self.max_metric:.4f}.csv'), index=False)
            results = {}
            log = 'Test\t' + self.args.task_name + "\t"
            tag_value = defaultdict(float)
            for k, v in metrics_dataset.items():
                log += k + ": {" + k + ":.4f}\t"
                results[k] = v
                tag = "eval/" + k
                tag_value[tag] = v
            print(log.format(**results))
