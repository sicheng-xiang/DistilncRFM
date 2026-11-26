"""CPU checks: official architecture, LoRA reload/merge, masks, labels and accumulation."""
import copy
import math
from pathlib import Path
import sys
import unittest
import torch
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from src.rinalmo_adapter.model import make_model,official_model,SDPASelfAttention,merge_lora
from tasks.rinalmo_downstream.data import Collator
from tasks.rinalmo_downstream.metrics import calculate
from tasks.rinalmo_downstream.run_rinalmo import weight_denominator,loss_sum

class AdapterTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        _,cls.alphabet,c=official_model()
        c.globals.embed_dim=40;c.model.transformer.num_blocks=2;c.model.transformer.num_heads=5
        c.model.transformer.attention_dropout=0.;c.model.transformer.residual_dropout=0.
        cls.c=c

    def new(self,task='family',mode='lora'):
        return make_model(None,3,task,mode,rank=2,alpha=4,dropout=0.,config=copy.deepcopy(self.c),load_weights=False)[0]

    def test_sdpa_matches_reference(self):
        torch.manual_seed(3);layer=SDPASelfAttention(40,5).eval()
        x=torch.randn(2,7,40);keep=torch.tensor([[1]*7,[1]*4+[0]*3],dtype=torch.bool)
        actual,_=layer(x,keep)
        q,k,v=layer.Wqkv(x).reshape(2,7,3,5,8).permute(2,0,3,1,4).unbind(0)
        q,k=layer.rotate(q),layer.rotate(k)
        scores=(q@k.transpose(-1,-2))/math.sqrt(8)
        scores=scores.masked_fill(~keep[:,None,None,:],-torch.inf)
        ref=(scores.softmax(-1)@v).transpose(1,2).reshape(2,7,40).masked_fill(~keep[...,None],0.)
        ref=layer.out_proj(ref)
        torch.testing.assert_close(actual,ref,atol=2e-6,rtol=2e-6)

    def test_rope_matches_official_reference(self):
        from rinalmo.model.rope import RotaryPositionEmbedding
        q=torch.randn(2,5,11,8);k=torch.randn_like(q)
        expected_q,expected_k=RotaryPositionEmbedding(8)(q,k)
        layer=SDPASelfAttention(40,5)
        torch.testing.assert_close(layer.rotate(q),expected_q,atol=0,rtol=0)
        torch.testing.assert_close(layer.rotate(k),expected_k,atol=0,rtol=0)

    def test_collation_base_alignment(self):
        c=Collator(self.alphabet,5,'splice')
        b=c([dict(seq='ACGUAC',label=[0,1,2,0,1,2]),dict(seq='AC',label=[2,1])])
        self.assertEqual(tuple(b['tokens'].shape),(2,7));self.assertEqual(b['labels'].dtype,torch.int64)
        self.assertEqual(b['labels'][0].tolist(),[0,1,2,0,1])
        self.assertEqual(b['labels'][1].tolist(),[2,1,-100,-100,-100])
        self.assertEqual(int(b['tokens'][0,1]),self.alphabet.get_idx('A'))
        model=self.new('splice');self.assertEqual(tuple(model(b['tokens'],b['base_mask']).shape),(2,5,3))

    def test_lora_reload_merge_and_frozen_base(self):
        torch.manual_seed(4);model=self.new();c=Collator(self.alphabet,8,'family')
        b=c([dict(seq='ACGUACGU',label=0),dict(seq='UGCA',label=1)])
        frozen={k:v.clone() for k,v in model.named_parameters() if not v.requires_grad}
        opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.01)
        model.train();F.cross_entropy(model(b['tokens'],b['base_mask']),b['labels']).backward();opt.step()
        for k,v in model.named_parameters():
            if k in frozen:
                self.assertIsNone(v.grad);torch.testing.assert_close(v,frozen[k],atol=0,rtol=0)
        model.eval();before=model(b['tokens'],b['base_mask']).detach()
        restored=self.new();restored.load_state_dict(model.state_dict());restored.eval()
        torch.testing.assert_close(before,restored(b['tokens'],b['base_mask']),atol=0,rtol=0)
        self.assertTrue(merge_lora(restored))
        after=restored(b['tokens'],b['base_mask'])
        torch.testing.assert_close(before,after,atol=2e-5,rtol=2e-5)

    def test_padding_invariance(self):
        model=self.new().eval()
        rows=[dict(seq='ACGU',label=0)]
        a=Collator(self.alphabet,4,'family',True)(rows)
        b=Collator(self.alphabet,12,'family',True)(rows)
        torch.testing.assert_close(model(a['tokens'],a['base_mask']),model(b['tokens'],b['base_mask']),atol=3e-6,rtol=3e-6)

    def test_weighted_short_accumulation(self):
        x=torch.randn(3,4);y=torch.tensor([0,1,1]);weights=torch.tensor([1.,2.5])
        a=torch.nn.Linear(4,2);b=copy.deepcopy(a)
        F.cross_entropy(a(x),y,weight=weights).backward()
        denom=float(weight_denominator(y,weights))
        for s in [slice(0,2),slice(2,3)]: (loss_sum(b(x[s]),y[s],weights,0.)/denom).backward()
        torch.testing.assert_close(a.weight.grad,b.weight.grad)

    def test_global_splice_topk(self):
        import numpy as np
        scores=np.array([[1,9,0],[0,8,7],[0,3,9],[0,0,8]])
        labels=np.array([1,0,2,0])
        m=calculate('splice',scores,labels)
        self.assertEqual(m['avg_topk_acc'],1.)

if __name__=='__main__':unittest.main()
