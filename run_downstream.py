# 修改原因：统一 Windows 入口，显式匹配四类底座与适配器；修改时间：2026-11-09
import argparse, json, os, subprocess, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
def main():
    p=argparse.ArgumentParser(description='Portable nRC/SPL/circRNA launcher')
    p.add_argument('task',choices=['nrc','spl','circ'])
    p.add_argument('action',choices=['train','eval'])
    p.add_argument('model',choices=['student','teacher','dis','dis_w','all'])
    p.add_argument('--batch_size',type=int)
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--output_dir',required=True,help='New output folder; relative to project root')
    a=p.parse_args()
    records=json.loads((ROOT/'migration/other_tasks_models.json').read_text(encoding='utf-8'))[a.task]
    if a.model=='all':
        if a.action=='train':p.error('Train one model at a time')
        keys=['student','teacher','dis','dis_w']
    else:keys=[a.model]
    if any(k not in records for k in keys):p.error('Model is missing from the task manifest')
    out=Path(a.output_dir);out=out if out.is_absolute() else ROOT/out
    for key in keys:
        m=records[key];dest=out/key if a.model=='all' else out
        if dest.exists() and any(dest.iterdir()):raise FileExistsError('Use a new output directory: '+str(dest))
        if a.task=='nrc':folder=ROOT/'tasks/seqcls_ssp'
        elif a.task=='spl':folder=ROOT/'tasks/splice_site_prediction'
        else:folder=ROOT/'tasks/RNA subcellular localization prediction/lora_circRNA'
        base=str(ROOT/m['base']);adapter=str(ROOT/m['adapter'])
        env=os.environ.copy();env['PYTHONUTF8']='1';env['TOKENIZERS_PARALLELISM']='false'
        if a.task=='spl':env['STRUCTRFM_BASE_PATH']=base
        if a.action=='eval':
            script={'nrc':'evaluate_family_classification.py','spl':'evaluate_splice_sites.py','circ':'evaluate_localization.py'}[a.task]
            cmd=[sys.executable,str(folder/script),'--LM_path',base,'--lora_path',adapter,
                '--batch_size',str(a.batch_size or 4),'--output_dir',str(dest),'--device',a.device]
            if a.task=='spl':cmd+=['--tokenizer_path',str(ROOT/m['tokenizer']),'--test_data',str(ROOT/'data/data/SPL/test.csv')]
            else:cmd+=['--model_scale',m['model_scale']]
        elif a.task in ['nrc','circ']:
            script='train_family_classification.py' if a.task=='nrc' else 'train_localization.py'
            ac=json.loads((Path(adapter)/'adapter_config.json').read_text(encoding='utf-8'))
            cmd=[sys.executable,str(folder/script),'--LM_path',base,'--model_scale',m['model_scale'],
                '--output_dir',str(dest),'--device',a.device,'--batch_size',str(a.batch_size or 32),
                '--save_checkpoints','True','--use_lora','True','--freeze_base','True','--load_in_4bit','False',
                '--lora_r',str(ac['r']),'--lora_alpha',str(ac['lora_alpha']),
                '--lora_dropout',str(ac['lora_dropout']),'--lora_target_modules',','.join(sorted(ac['target_modules']))]
            if a.task=='nrc':cmd+=['--dataset_dir',str(ROOT/'data/data'),'--dataset','nRC','--learning_rate','1e-4','--num_train_epochs','100']
            else:cmd+=['--data_dir',str(folder.parent/'RNALoc-LM-main/data'),'--dataset','circRNA','--learning_rate','1e-5','--num_train_epochs','20']
        else:
            if a.device.startswith('cuda'):
                env['CUDA_VISIBLE_DEVICES']=a.device.split(':')[1] if ':' in a.device else '0'
            cmd=[sys.executable,str(folder/'train_splice_sites.py'),'--model_name_or_path',base,
                '--data_path',str(ROOT/'data/data/SPL'),'--model_type','structRFM',
                '--output_dir',str(dest),'--run_name',dest.name,'--model_max_length','514',
                '--per_device_train_batch_size',str(a.batch_size or 256),'--per_device_eval_batch_size',str(a.batch_size or 256),
                '--learning_rate','2e-4','--num_train_epochs','100' if key=='dis_w' else '50',
                '--save_strategy','steps','--eval_strategy','steps','--save_steps','500','--eval_steps','500',
                '--warmup_steps','50','--logging_steps','100','--seed','666','--token_type','single',
                '--freeze_base','True','--dataloader_num_workers','0','--report_to','none',
                '--fp16','False' if a.device=='cpu' else 'True']
            if a.device=='cpu':cmd+=['--use_cpu','True']
        print('Running:',subprocess.list2cmdline(cmd),flush=True)
        subprocess.run(cmd,cwd=folder,env=env,check=True)
if __name__=='__main__':main()
