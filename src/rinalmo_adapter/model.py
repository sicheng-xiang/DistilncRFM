"""Official RiNALMo weights, with a FlashAttention-compatible native PyTorch backend."""
from pathlib import Path
import sys
import types
import torch
from torch import nn
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parents[2]

class SDPASelfAttention(nn.Module):
    """Preserve official FlashMultiHeadSelfAttention Wqkv/out_proj parameter names.

    Non-interleaved RoPE, base 10000; True padding mask means KEEP, as upstream.
    No flash-attn package, dtype conversion, or custom CUDA kernel required.
    """
    def __init__(self,embed_dim,num_heads,attention_dropout=0.,causal=False,use_rot_emb=True,bias=False):
        super().__init__();self.num_heads=num_heads;self.head_dim=embed_dim//num_heads
        self.use_rot_emb=use_rot_emb;self.causal=causal;self.dropout=attention_dropout
        self.Wqkv=nn.Linear(embed_dim,3*embed_dim,bias=bias)
        self.out_proj=nn.Linear(embed_dim,embed_dim,bias=bias)

    def rotate(self,x):
        d=x.shape[-1]
        inv=1/(10000**(torch.arange(0,d,2,device=x.device,dtype=torch.float32)/d))
        phase=torch.outer(torch.arange(x.shape[-2],device=x.device,dtype=torch.float32),inv)
        cos=torch.cat([phase.cos(),phase.cos()],-1).to(x.dtype)
        sin=torch.cat([phase.sin(),phase.sin()],-1).to(x.dtype)
        first,second=x.chunk(2,-1)
        return x*cos+torch.cat([-second,first],-1)*sin

    def forward(self,x,key_padding_mask=None,return_attn_probs=False):
        if return_attn_probs:raise ValueError('Attention maps are not part of downstream efficiency runs')
        b,s,d=x.shape
        q,k,v=self.Wqkv(x).reshape(b,s,3,self.num_heads,self.head_dim).permute(2,0,3,1,4).unbind(0)
        if self.use_rot_emb:q,k=self.rotate(q),self.rotate(k)
        mask=key_padding_mask[:,None,None,:].bool() if key_padding_mask is not None else None
        out=F.scaled_dot_product_attention(q,k,v,attn_mask=mask,
            dropout_p=self.dropout if self.training else 0.,is_causal=self.causal)
        out=out.transpose(1,2).reshape(b,s,d)
        if key_padding_mask is not None:out=out.masked_fill(~key_padding_mask[...,None],0.)
        return self.out_proj(out),None

class SplitQKV(nn.Module):
    def __init__(self,linear):
        super().__init__();d=linear.in_features
        for index,name in enumerate(['query','key','value']):
            part=nn.Linear(d,d,bias=linear.bias is not None,device=linear.weight.device,dtype=linear.weight.dtype)
            with torch.no_grad():
                part.weight.copy_(linear.weight[index*d:(index+1)*d])
                if linear.bias is not None:part.bias.copy_(linear.bias[index*d:(index+1)*d])
            setattr(self,name,part)
    def forward(self,x):return torch.cat([self.query(x),self.key(x),self.value(x)],-1)

def official_model(config=None):
    """Use vendored upstream model/config; replace only its attention implementation."""
    vendor=ROOT/'external_models/RiNALMo';sys.path.insert(0,str(vendor))
    # The original attention module imports flash-attn unconditionally. Register a
    # small attention module BEFORE loading upstream modules, without fake flash kernels.
    import rinalmo.model
    name='rinalmo.model.attention'
    if name not in sys.modules:
        module=types.ModuleType(name)
        module.FlashMultiHeadSelfAttention=SDPASelfAttention
        module.MultiHeadSelfAttention=SDPASelfAttention
        sys.modules[name]=module
    from rinalmo.config import model_config
    from rinalmo.model.model import RiNALMo
    from rinalmo.data.alphabet import Alphabet
    c=model_config('micro') if config is None else config
    c.model.transformer.use_flash_attn=True  # preserve upstream KEEP-mask convention
    return RiNALMo(c),Alphabet(**c['alphabet']),c

class DownstreamModel(nn.Module):
    def __init__(self,backbone,hidden_size,num_labels,task,pooling='cls'):
        super().__init__();self.backbone=backbone;self.task=task;self.pooling=pooling
        self.head=nn.Sequential(nn.Linear(hidden_size,hidden_size),nn.GELU(),nn.LayerNorm(hidden_size),
                                nn.Dropout(.1),nn.Linear(hidden_size,num_labels)) if task!='splice' else nn.Linear(hidden_size,num_labels)
    def forward(self,tokens,base_mask):
        # Avoid the unused masked-LM head in upstream forward; no model architecture change.
        b=self.backbone.get_base_model() if hasattr(self.backbone,'get_base_model') else self.backbone
        x=b.token_dropout(b.embedding(tokens),tokens)
        hidden,_=b.transformer(x,key_padding_mask=tokens.ne(b.pad_tkn_idx),need_attn_weights=False)
        if self.task=='splice':return self.head(hidden[:,1:1+base_mask.shape[1]])
        if self.pooling=='mean':
            values=hidden[:,1:1+base_mask.shape[1]].float().masked_fill(~base_mask[...,None],0.)
            pooled=values.sum(1)/base_mask.sum(1,keepdim=True).clamp_min(1)
        else:pooled=hidden[:,0]
        return self.head(pooled)

def make_model(weights,num_labels,task,mode='lora',rank=32,alpha=64,dropout=.05,pooling='cls',config=None,load_weights=True):
    backbone,alphabet,c=official_model(config)
    if load_weights:
        state=torch.load(weights,map_location='cpu',weights_only=True)
        backbone.load_state_dict(state,strict=True)
    # The task objective does not use the masked language-model head.
    backbone.lm_mask_head.requires_grad_(False)
    targets=[]
    if mode=='lora':
        from peft import LoraConfig,get_peft_model
        for block in backbone.transformer.blocks:
            block.mh_attn.Wqkv=SplitQKV(block.mh_attn.Wqkv)
        # Map BERT Q/K/V, attention output, and FFN dense layers to RiNALMo's SwiGLU.
        for name,module in backbone.named_modules():
            if isinstance(module,nn.Linear) and not name.startswith('lm_mask_head'):
                targets.append(name)
        backbone=get_peft_model(backbone,LoraConfig(r=rank,lora_alpha=alpha,lora_dropout=dropout,
            target_modules=targets,bias='none',inference_mode=False))
    elif mode!='full':raise ValueError(mode)
    model=DownstreamModel(backbone,c.globals.embed_dim,num_labels,task,pooling)
    return model,alphabet,targets

def merge_lora(model):
    if hasattr(model.backbone,'merge_and_unload'):
        model.backbone=model.backbone.merge_and_unload(safe_merge=True)
        return True
    return False
