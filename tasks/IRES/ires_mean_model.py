"""IRES-only nucleotide mean pooling; shared structRFM implementation is untouched."""
# 修改原因：旧[1:-1].mean会包含SEP/PAD；只对有效核苷酸求均值，并保持LoRA模块名称；修改时间：2026-10-15
import torch


def nucleotide_mask(input_ids, attention_mask, nucleotide_ids):
    if attention_mask is None or attention_mask.shape != input_ids.shape:
        raise ValueError('Nucleotide mean pooling requires a matching attention_mask')
    valid = torch.zeros_like(input_ids, dtype=torch.bool)
    for token_id in nucleotide_ids:
        valid |= input_ids.eq(token_id)
    valid &= attention_mask.bool()
    if torch.any(valid.sum(dim=1) == 0):
        raise ValueError('Sequence has no valid nucleotide tokens after masking')
    return valid


def nucleotide_mean(hidden_states, valid):
    # float32 accumulation for stable averaging; masked positions contribute no value or gradient.
    selected = hidden_states.float().masked_fill(~valid.unsqueeze(-1), 0.0)
    return (selected.sum(dim=1) / valid.sum(dim=1, keepdim=True)).to(hidden_states.dtype)


def get_ires_mean_model(num_class, tokenizer, freeze_base=True, use_mean_feature=True, **kwargs):
    from src.structRFM.model import structRFM_for_cls
    if not use_mean_feature:
        raise ValueError('This factory is for nucleotide mean pooling only')
    vocab = tokenizer.get_vocab()
    special_ids = set(tokenizer.all_special_ids)
    # Vocabulary whitelist: special/unknown tokens must not silently become nucleotide tokens.
    ids = tuple(sorted({vocab[base] for base in 'ACGTU' if base in vocab and vocab[base] not in special_ids}))
    if not all(base in vocab and vocab[base] not in special_ids for base in 'ACG') or not any(
            base in vocab and vocab[base] not in special_ids for base in 'TU'):
        raise ValueError('Expected single-nucleotide A/C/G/T-or-U vocabulary')

    class IRESMeanClassifier(structRFM_for_cls):
        # Keep structRFM.* and cls.* parameter names unchanged for PEFT save/load.
        def forward(self, input_ids, attention_mask=None, return_dict=True, **unused):
            valid = nucleotide_mask(input_ids, attention_mask, self.nucleotide_ids)
            outputs = self.structRFM(input_ids=input_ids, attention_mask=attention_mask)
            pooled = nucleotide_mean(outputs.hidden_states[-1], valid)
            return self.cls(pooled)

    model = IRESMeanClassifier(num_class=num_class, tokenizer=tokenizer,
                              use_mean_feature=True, **kwargs)
    model.nucleotide_ids = ids
    model.pooling = 'nucleotide_mean_v1'
    if freeze_base:
        for param in model.structRFM.parameters():
            param.requires_grad = False
    return model
