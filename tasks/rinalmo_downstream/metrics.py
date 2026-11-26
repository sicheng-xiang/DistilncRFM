import numpy as np
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,matthews_corrcoef,roc_auc_score,average_precision_score,precision_recall_curve,auc

def calculate(task,prob,labels):
    if task=='splice':
        valid=labels!=-100;target=labels[valid];scores=prob[valid]
        result={}
        for label,name in [(1,'acceptor'),(2,'donor')]:
            truth=target==label;k=int(truth.sum())
            # Same global-per-class top-k definition as original train_splice_sites.py.
            result[name+'_topk_acc']=float(truth[np.argsort(scores[:,label])[::-1][:k]].mean()) if k else 0.
        result['avg_topk_acc']=(result['acceptor_topk_acc']+result['donor_topk_acc'])/2
        return result
    pred=prob.argmax(-1)
    result=dict(accuracy=float(accuracy_score(labels,pred)),
        precision=float(precision_score(labels,pred,average='macro',zero_division=0)),
        recall=float(recall_score(labels,pred,average='macro',zero_division=0)),
        f1=float(f1_score(labels,pred,average='macro',zero_division=0)),mcc=float(matthews_corrcoef(labels,pred)))
    if prob.shape[-1]==2:
        result['positive_f1']=float(f1_score(labels,pred,zero_division=0))
        if len(np.unique(labels))==2:
            result['auroc']=float(roc_auc_score(labels,prob[:,1]))
            result['average_precision']=float(average_precision_score(labels,prob[:,1]))
            p,r,_=precision_recall_curve(labels,prob[:,1]);result['auprc']=float(auc(r,p))
        else:result['auroc']=0.;result['average_precision']=0.;result['auprc']=0.
    return result
