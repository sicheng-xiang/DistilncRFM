import math
import torch
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score,f1_score
from base_classes import BaseMetrics


# 修改原因：统一CE对应的softmax分数、固定阈值和PR指标口径，禁止测试集调阈值；修改时间：2026-10-15
class SeqClsMetrics(BaseMetrics):
    def __init__(self, metrics, threshold=0.5):
        super().__init__(metrics=metrics)
        if not 0 <= threshold <= 1:
            raise ValueError("threshold must be in [0,1]")
        self.threshold = threshold
        self.y_score, self.y_true = [], []

    def __call__(self, outputs, labels):
        if outputs.ndim != 2 or outputs.shape[1] != 2 or not torch.isfinite(outputs).all():
            raise ValueError("Expected finite [N,2] logits")
        scores = torch.softmax(outputs.float(), dim=1)[:, 1]
        self.y_score.extend(scores.detach().cpu().tolist())
        self.y_true.extend(labels.detach().cpu().tolist())
        return {}

    def compute(self):
        from sklearn.metrics import accuracy_score, precision_score, recall_score, matthews_corrcoef
        if not self.y_true:
            raise ValueError("Empty evaluation dataset")
        y = np.asarray(self.y_true)
        pred = (np.asarray(self.y_score) >= self.threshold).astype(int)
        return {"accuracy": accuracy_score(y, pred),
                "precision": precision_score(y, pred, zero_division=0),
                "recall": recall_score(y, pred, zero_division=0),
                "f1s": f1_score(y, pred, zero_division=0),
                "mcc": matthews_corrcoef(y, pred)}

    def compute_auc_aupr(self):
        from sklearn.metrics import precision_recall_curve, auc
        if set(self.y_true) != {0, 1}:
            raise ValueError("AUROC requires both true classes 0 and 1")
        precision, recall, _ = precision_recall_curve(self.y_true, self.y_score)
        return {"auc": roc_auc_score(self.y_true, self.y_score),
                "aupr": auc(recall, precision),
                "ap": average_precision_score(self.y_true, self.y_score)}

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
            if j1 > 0 and i < j1:  # pos
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