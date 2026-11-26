#losses.py
from torch import nn
import torch
import torch.nn.functional as F


class SeqClsLoss(nn.Module):
    def __init__(self, pos_weight=2.0,label_smoothing=0.0):
        super(SeqClsLoss, self).__init__()
        # 给正样本更高权重（lncRNA二分类: nucleus vs cytoplasm）
        # 数据: nucleus(636) > cytoplasm(367), cytoplasm是少数类(标签1)
        self.pos_weight = torch.tensor([1.0, pos_weight])  # [nucleus权重, cytoplasm权重]
          # Label smoothing
        self.label_smoothing = label_smoothing

    def forward(self, outputs, labels):
        """
        outputs: 模型输出 [B, 2]
        labels: 真实标签 [B]
        """
        weight = self.pos_weight.to(outputs.device)
        loss = F.cross_entropy(
            outputs,
            labels,
            weight=weight,
            label_smoothing=self.label_smoothing
        )

        return loss


class RRInterLoss(nn.Module):

    def __init__(self):
        super(RRInterLoss, self).__init__()

    def forward(self, outputs, labels):
        loss = F.cross_entropy(outputs, labels)
        return loss


class StructuredLoss(nn.Module):
    def __init__(self,
                 loss_pos_paired=0,
                 loss_neg_paired=0,
                 loss_pos_unpaired=0,
                 loss_neg_unpaired=0,
                 l1_weight=0.,
                 l2_weight=0.):

        super(StructuredLoss, self).__init__()
        self.loss_pos_paired = loss_pos_paired
        self.loss_neg_paired = loss_neg_paired
        self.loss_pos_unpaired = loss_pos_unpaired
        self.loss_neg_unpaired = loss_neg_unpaired
        self.l1_weight = l1_weight
        self.l2_weight = l2_weight

    def forward(self, model, seq, pairs, embeddings, fname=None):
        pred, pred_s, _, param = model(seq,
                                       embeddings,
                                       return_param=True,
                                       reference=pairs,
                                       loss_pos_paired=self.loss_pos_paired,
                                       loss_neg_paired=self.loss_neg_paired,
                                       loss_pos_unpaired=self.loss_pos_unpaired,
                                       loss_neg_unpaired=self.loss_neg_unpaired)
        ref, ref_s, _ = model(seq, embeddings, param=param,
                              constraint=pairs, max_internal_length=None)
        length = torch.tensor([len(s) for s in seq]).to(pred.device)
        loss = (pred - ref) / length

        if loss.item() > 1e10 or torch.isnan(loss):
            print()
            print(fname)
            print(loss.item(), pred.item(), ref.item())
            print(seq)

        if self.l1_weight > 0.0:
            for p in model.parameters():
                loss += self.l1_weight * torch.sum(torch.abs(p))
        return torch.sum(loss)