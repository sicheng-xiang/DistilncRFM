"""SPL metrics and FP32 inference efficiency after merging the LoRA adapter."""
import argparse, os, json, sys
from pathlib import Path
import torch
import numpy as np
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from peft import PeftModel
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tasks.inference_efficiency import InferenceMeter, add_inference_arguments, merge_for_inference, output_directory
from model.structRFM.modeling_structRFM import structRFMForNucleotideLevel
from train_splice_sites import SupervisedDataset, compute_metrics, DataCollatorForSupervisedDataset
def main():
    p=argparse.ArgumentParser('Portable SPL evaluation')
    p.add_argument('--LM_path',required=True)
    p.add_argument('--lora_path',required=True)
    p.add_argument('--tokenizer_path',required=True)
    p.add_argument('--test_data',required=True)
    p.add_argument('--output_dir',default=None)
    p.add_argument('--batch_size',type=int,default=8)
    p.add_argument('--device',default='cuda:0' if torch.cuda.is_available() else 'cpu')
    add_inference_arguments(p)
    a=p.parse_args()
    if a.batch_size <= 0:
        raise ValueError('batch_size must be positive')
    for key in ('LM_path', 'lora_path', 'tokenizer_path', 'test_data'):
        value = Path(getattr(a, key)).expanduser()
        setattr(a, key, str((value if value.is_absolute() else ROOT / value).resolve()))
    out = output_directory(ROOT, 'splice', a.output_dir)
    device = torch.device(a.device)
    os.environ['STRUCTRFM_BASE_PATH']=str(Path(a.LM_path).resolve())
    tokenizer=AutoTokenizer.from_pretrained(a.tokenizer_path)
    a.token_type='single';a.model_max_length=514
    ds=SupervisedDataset(tokenizer=tokenizer,args=a,data_path=a.test_data,kmer=-1)
    model=structRFMForNucleotideLevel.from_pretrained(a.LM_path,num_labels=3,
        problem_type='single_label_classification',tokenizer=tokenizer,output_hidden_states=False)
    model=PeftModel.from_pretrained(model,a.lora_path,is_trainable=False)
    model = merge_for_inference(model).float().to(device).eval()
    loader = DataLoader(ds, batch_size=a.batch_size, shuffle=False, num_workers=0,
                        collate_fn=DataCollatorForSupervisedDataset(tokenizer=tokenizer,args=a))
    def prepare(batch):
        # No labels: time inference only; calculate the original loss metric separately.
        return {key: value.to(device) for key, value in batch.items() if key != 'labels'}
    meter = InferenceMeter(model, device, a.batch_size, a.warmup_batches,
                           task='SPL', LM_path=a.LM_path, lora_path=a.lora_path,
                           tokenizer_path=a.tokenizer_path, test_data=a.test_data, attention_mask=True)
    logits_all, labels_all = [], []
    for batch, logits in meter.predict(loader, prepare):
        logits_all.append(logits)
        labels_all.append(batch['labels'].cpu())
    meter.save(out)
    logits, labels = torch.cat(logits_all), torch.cat(labels_all)
    predictions, truth = logits.numpy(), labels.numpy()
    metrics = {f'test_{key}': float(value) for key,value in compute_metrics((predictions,truth)).items()}
    metrics['test_loss'] = torch.nn.functional.cross_entropy(logits.reshape(-1,3), labels.reshape(-1).long()).item()
    metrics['test_runtime'] = meter.result['inference_pipeline_time_s']
    metrics['test_samples_per_second'] = meter.result['pipeline_throughput_samples_per_s']
    metrics['test_steps_per_second'] = meter.result['pipeline_throughput_batch_per_s']
    (out/'test_results.json').write_text(json.dumps(metrics,indent=2)+'\n',encoding='utf-8')
    np.savez(out/'pred_data.npz',pred=predictions,label=truth)
    print(metrics)
if __name__=='__main__':main()
