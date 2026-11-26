
import argparse, json, subprocess, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
def main():
    p=argparse.ArgumentParser(description='Portable IRES evaluation/fine-tuning launcher')
    p.add_argument('action',choices=['eval','train'])
    p.add_argument('model',choices=['student','teacher','dis_w','dis','all'])
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--batch_size',type=int)
    p.add_argument('--gradient_accumulation_steps', type=int, default=1,
                   help='Training only; default 1 preserves original registered experiment configuration')
    p.add_argument('--num_train_epochs', type=int)
    p.add_argument('--output_dir',help='Required for training; root-relative or absolute')
    a=p.parse_args()
    registry=json.loads((ROOT/'migration/ires_models.json').read_text(encoding='utf-8'))
    models=registry['models']
    if a.action=='train' and (a.model=='all' or not a.output_dir):
        p.error('Train one model at a time and provide --output_dir')
    for key in (models if a.model=='all' else [a.model]):
        m=models[key]
        if a.action=='eval':
            cmd=[sys.executable,str(ROOT/'tasks/IRES/evaluate_ires.py'),'--checkpoint',str(ROOT/m['checkpoint']),
                 '--device',a.device,'--batch_size',str(a.batch_size or 4)]
            if a.output_dir: cmd+=['--output_dir',str(Path(a.output_dir)/key)]
        else:
            c=registry['training_defaults'] | m.get('training_overrides', {})
            cmd=[sys.executable,str(ROOT/'tasks/IRES/train_ires.py'),
                 '--LM_path',str(ROOT/m['base']),'--model_scale',m['model_scale'],
                 '--dataset_dir',str(ROOT/registry['dataset_dir']),'--device',a.device,
                 '--save_checkpoints','True','--output_dir',a.output_dir,'--batch_size',str(a.batch_size or c['batch_size']),
                 '--gradient_accumulation_steps',str(a.gradient_accumulation_steps)]
            for k in ['learning_rate','head_learning_rate','eval_steps','num_train_epochs','pos_weight','seed','max_seq_len','pooling','replace_T','replace_U','use_lora','freeze_base','dataloader_num_workers']:
                value = a.num_train_epochs if k == 'num_train_epochs' and a.num_train_epochs is not None else c[k]
                cmd += ['--'+k,str(value)]
        print('Running:',subprocess.list2cmdline(cmd),flush=True)
        subprocess.run(cmd,cwd=ROOT,check=True)
if __name__=='__main__': main()
