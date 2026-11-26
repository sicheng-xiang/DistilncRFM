# metrics.py
import math
import torch
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, f1_score, precision_score, recall_score, classification_report
from base_classes import BaseMetrics


class SeqClsMetrics(BaseMetrics):
    def __init__(self, metrics):
        super().__init__(metrics=metrics)
        self.y_score = []
        self.y_true = []
        self.pred = None

    def __call__(self, outputs, labels):
        # Retain full logits with shape [batch, 2].
        self.y_score.extend(outputs.detach().cpu().numpy())
        self.y_true.extend(labels.detach().cpu().numpy())
        return {}

    def compute(self):
        y_true = np.array(self.y_true)
        y_score = np.array(self.y_score)

        # Convert logits to probabilities with softmax.
        exp_scores = np.exp(y_score - np.max(y_score, axis=1, keepdims=True))
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        y_preds = np.argmax(probs, axis=1)  # Select predicted labels with argmax.

        # Compute macro-averaged classification metrics.
        acc = accuracy_score(y_true, y_preds)
        precision = precision_score(y_true, y_preds, average='macro', zero_division=0)
        recall = recall_score(y_true, y_preds, average='macro', zero_division=0)
        f1s = f1_score(y_true, y_preds, average='macro', zero_division=0)
        
        # Compute MCC separately.
        tp = ((y_preds == 1) & (y_true == 1)).sum()
        tn = ((y_preds == 0) & (y_true == 0)).sum()
        fp = ((y_preds == 1) & (y_true == 0)).sum()
        fn = ((y_preds == 0) & (y_true == 1)).sum()
        mcc = ((tp * tn) - (fp * fn)) / math.sqrt(
            (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) if (
                      (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) > 0 else 0.0

        metric_map = {
            "accuracy": acc,
            "recall": recall,
            "precision": precision,
            "f1s": f1s,
            "mcc": mcc
        }

        ret = {m: metric_map[m.lower()] for m in self.metrics if m.lower() in metric_map}
        
        
        print("\n详细分类报告:")
        report = classification_report(y_true, y_preds, zero_division=0)
        print(report)
        
        return ret

    def compute_auc_aupr(self):
        y_true = np.array(self.y_true)
        y_score = np.array(self.y_score)
        
        auc = 0.0
        aupr = 0.0
        try:
            # Use class-1 probabilities for AUROC and AUPR.
            exp_scores = np.exp(y_score - np.max(y_score, axis=1, keepdims=True))
            probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
            probs_class1 = probs[:, 1]
            auc = roc_auc_score(y_true, probs_class1)
            aupr = average_precision_score(y_true, probs_class1)
        except:
            pass
        return {"auc": auc, "aupr": aupr}

    def reset(self):
        self.y_score.clear()
        self.y_true.clear()


class RRInterMetrics(BaseMetrics):
    def __call__(self, outputs, labels):
        return super().__call__(outputs, labels)


def compare_bpseq(ref, pred):
    L = len(ref) - 1
    tp = fp = fn = 0
    if (len(ref) > 0 and isinstance(ref[0], list)) or (isinstance(ref, torch.Tensor) and ref.ndim == 2):
        if isinstance(ref, torch.Tensor):
            ref = ref.tolist()
        ref = {(min(i, j), max(i, j)) for i, j in ref}
        pred = {(i, j) for i, j in enumerate(pred) if i < j}
        tp = len(ref & pred)
        fp = len(pred - ref)
        fn = len(ref - pred)
    else:
        assert (len(ref) == len(pred)), print(len(ref), ref, len(pred), pred)
        for i, (j1, j2) in enumerate(zip(ref, pred)):
            if j1 > 0 and i < j1:
                if j1 == j2:
                    tp += 1
                elif j2 > 0 and i < j2:
                    fp += 1
                    fn += 1
                else:
                    fn += 1
            elif j2 > 0 and i < j2:
                fp += 1
        tn = L * (L - 1) // 2 - tp - fp - fn
    return tp, tn, fp, fn


class SspMetrics(BaseMetrics):
    def __init__(self, metrics):
        super(SspMetrics, self).__init__(metrics=metrics)

    def __call__(self, tp, tn, fp, fn):
        ret = {}
        acc = (tp + tn) / (tp + tn + fp + fn) if tp + tn + fp + fn > 0. else 0.
        sen = tp / (tp + fn) if tp + fn > 0. else 0.
        ppv = tp / (tp + fp) if tp + fp > 0. else 0.
        fval = 2 * sen * ppv / (sen + ppv) if sen + ppv > 0. else 0.
        mcc = ((tp * tn) - (fp * fn)) / math.sqrt(
            (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) if (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn) > 0. else 0.
        for m in self.metrics:
            if m == "accuracy":
                ret[m] = acc
            elif m == "recall":
                ret[m] = sen
            elif m == "precision":
                ret[m] = ppv
            elif m == "f1s":
                ret[m] = fval
            elif m == "mcc":
                ret[m] = mcc
        return ret