import torch
import pandas as pd
from torch.utils.data import DataLoader

from .model import get_structRFM
from .data.tokenizer import get_mlm_tokenizer
from .data.RNAdata import process_mlm_input_seq
def save_seqs_to_csv(path, seqs, names=None):
    if names is None:
        names = [f'seq{i}' for i in range(len(seqs))]
    df = pd.DataFrame({'name': names, 'seq': seqs})
    df.to_csv(path, index=False)


class structRFM_infer:
    def __init__(self, from_pretrained, max_length=514, dim=768, layer=12, num_attention_heads=12, guide = None,motif_tree = None,output_hidden_states=True, device=None):
        # set output_hidden_states=True to get the hidden states (features)

        # self.tokenizer = AutoTokenizer(from_pretrained)
        # self.model = AutoModel.from_pretrained(from_pretrained, output_hidden_states=True) # why won't output logtits?
        self.guide = guide

        if  not guide :
            print('not use guide')
            self.tokenizer = get_mlm_tokenizer(max_length=max_length)
            self.model = get_structRFM(dim=dim, layer=layer, num_attention_heads=num_attention_heads, from_pretrained=from_pretrained, tokenizer=self.tokenizer,output_hidden_states=output_hidden_states)
        else:
            # Optional guide dependencies are not needed by the Rfam guide=False task.
            from .RNABert.model import get_rnabert_model
            from .RNABert.data_p import GenomeSequenceProcessor
            from .RNABert.motif_tree import load_motif
            print('use guide')
            processor = GenomeSequenceProcessor(
                tokenizer_path=r'C:\Users\xsc\Desktop\参考文献\structRFM-master\src\structRFM\RNABert\tokenizer_mlm.json',
                max_length=max_length  # 12200 + [CLS] + [SEP]
            )
            self.tokenizer = processor.tokenizer
            motif_tree_dict = load_motif(r"C:\Users\xsc\Desktop\参考文献\structRFM-master\src\structRFM\RNABert\motif", processor.tokenizer)
            self.motif_tree_dict = motif_tree_dict
            self.model = get_rnabert_model(dim=dim, layer=layer, num_attention_heads=num_attention_heads, from_pretrained=from_pretrained,  max_length=max_length,tokenizer=self.tokenizer, output_hidden_states=output_hidden_states)
        if device is None:
            if torch.cuda.is_available():
                self.model.cuda()
        else:
            self.model.to(device)
        print(f'Running on {self.model.device}')


    def get_tokenizer(self):
        return self.tokenizer

    def get_model(self):
        return self.model

    def unmask(self, masked_seq, top_k=1):
        '''
            model.output.logtis
        '''
        self.model.eval()
        text = process_mlm_input_seq(masked_seq)
        inputs = self.tokenizer(text, return_tensors='pt')
        mask_positions = torch.where(inputs['input_ids'][0] == self.tokenizer.mask_token_id)[0]

        with torch.no_grad(), torch.cuda.amp.autocast():
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            outputs = self.model(**inputs)
        logits = outputs.logits # B x (seq_len+2) x vocab
        predicted_tokens = []
        top_tokens_list = []
        for pos in mask_positions:
            logit = logits[0, pos]
            # predicted_id = logit.argmax().item()
            top_ids = logit.topk(top_k).indices.tolist()
            predicted_token = self.tokenizer.decode(top_ids[0])
            top_tokens_list.append(self.tokenizer.convert_ids_to_tokens(top_ids))
            predicted_tokens.append(predicted_token)

        parts = masked_seq.upper().split(self.tokenizer.mask_token)
        new_parts = []
        for part, pred in zip(parts, predicted_tokens):
            new_parts.append(part)
            new_parts.append(pred)
        new_parts.append(parts[-1])
        return ''.join(new_parts), predicted_tokens, top_tokens_list


    @torch.no_grad()
    def model_forward(self, seq, return_inputs=False, is_cal_loss=False, output_attentions=False, is_training=False):
        '''
            Get model outputs.
            seq: str
                seq_len
            return outputs, (inputs): dict, (Tensor)
                {'logits': logits, 'hidden_states': hidden_states', 'attentions': attentions}
        '''
        if is_training:
            self.model.train()
        else:
            self.model.eval()
        text = process_mlm_input_seq(seq)
        inputs = self.tokenizer(text, return_tensors='pt')
        if self.guide:
            from .RNABert.motifDNA import find_motifs_in_masked_sequence
            batch_size = inputs["input_ids"].shape[0]
            seq_len = inputs["input_ids"].shape[1]
            all_position_arrays = []

            for i in range(batch_size):
                # Use masked input IDs.
                input_ids = inputs["input_ids"][i]

                # Compute token position annotations.
                position_array, motifs = find_motifs_in_masked_sequence(
                    input_ids,
                    self.motif_tree_dict
                )

                # Align annotation and token lengths.
                if len(position_array) != seq_len:
                    if len(position_array) < seq_len:
                        # Pad to the required length.
                        padding = torch.zeros(seq_len - len(position_array),
                                              dtype=position_array.dtype)
                        position_array = torch.cat([position_array, padding])
                    else:
                        # Truncate to the required length.
                        position_array = position_array[:seq_len]

                all_position_arrays.append(position_array)

            # Add annotations to the batch.
            inputs["position_array"] = torch.tensor(all_position_arrays)
            # print('!!!!!'*5)
            print(f"DEBUG: 添加 position_array: {inputs['position_array']}")
            # print('!!!!!' * 5)
            
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        attention = None
        outputs = self.model(**inputs, labels=inputs['input_ids'], output_attentions=output_attentions) if is_cal_loss else self.model(**inputs, output_attentions=output_attentions)
        if return_inputs:
            return outputs, inputs
        else:
            return outputs


    def extract_raw_feature(self, seq, include_special=True, return_all=False, output_attentions=False, is_training=False):
        '''
            Extract hidden feature of input seq.
            seq: str
                seq_len
            return: Tensor
               (all_layers=13) x seq_len x hidden_size768
        '''
        attention = None
        # Respect the caller: Rfam needs features, not attention matrices.
        outputs = self.model_forward(seq, output_attentions=output_attentions, is_training=is_training)
        hidden_states = outputs.hidden_states  # tuple(B x (seq_len+2) x hidden_size768), tuple_len = (1+layer12)
        # print('!!!!!'*5)
        # print('hidden_states:',hidden_states)
        # print('!!!!!' * 5)
        if hidden_states is None:
            raise ValueError(
                "模型未返回隐藏层状态！请检查 forward 函数是否开启 output_hidden_states=True 并正确收集 hidden_states")
        ## batch_size = 1, use the first one 0 of this batch
        sls = slice(None, None) if include_special else slice(1, -1)
        out_features = [hidd[0, sls, :] for hidd in hidden_states] # tuple(seq_len x hidden_size), tuple_len = (1+layer12)
        final_out_features = out_features if return_all else out_features[-1]
        # print('!!!!!' * 5)
        # print('final_out_features',final_out_features)
        # print('!!!!!' * 5)
        #out_attentions = outputs.attentions if return_all else outputs.attentions[-1]
        if output_attentions:
            pass
            #return final_out_features, out_attentions
        else:
            return final_out_features

    def extract_feature(self, seq):
        '''
            return feature_dic, with keys: cls_feat, seq_feat, mat_feat
        '''
        # feat  tuple: layer=12, tuple[i]: batch x L x hidden_dim(=768)
        features = self.extract_raw_feature(seq, return_all=True)
        ## (1+L+1)x 768,  [CLS] seq [SEP]
        last_feat = features[-1]
        cls_feat = last_feat[0,:] # 1x768
        seq_feat = last_feat[1:-1, :] # Lx768
        mat_feat = seq_feat @ seq_feat.transpose(-1,-2) # LxL
        # L = len(seq)
        # mat_feat = seq_feat.unsqueeze(-2).unsqueeze(-1)*seq_feat.unsqueeze(-3).unsqueeze(-2) # LxLx768x768 ## cost too much memory
        # mat_feat = mat_feat.reshape(L, L, -1).mean(dim=-1)
        return {
                'cls_feat': cls_feat, # 1x768
                'seq_feat': seq_feat, # Lx768
                'mat_feat': mat_feat, # LxL
               }
