# Reproduction guide

Run commands from the repository root using the environment described in the README. Use a new output directory for every run.

## Data and weights

| Task | Input |
|---|---|
| RNA family | data/data/nRC/train.fa, test.fa |
| Localization | tasks/RNA subcellular localization prediction/RNALoc-LM-main/data/ |
| Splice sites | data/data/SPL/train.csv, val.csv, test.csv |
| IRES | data/data/ires_seq/train/ and test/ |
| Rfam | tasks/zeroshot/task2_rfam/data/random_sample_max10_n24607_rf4170.pkl |

Base weights are under model/. Task checkpoints are under tasks/.../checkpoints/. Keep adapter configs, tokenizers, and IRES run_config.json files with their weights.

Path registries:
- migration/other_tasks_models.json: three LoRA tasks.
- migration/ires_models.json: IRES paths and training defaults.
- migration/rinalmo_models.json: RiNALMo downstream checkpoints.

Official download links and dataset provenance documentation are pending. Pretraining data is not included.

## Training

These commands demonstrate the current entry points, not a verified reproduction of every historical run:

```shell
python run_downstream.py nrc train dis_w --batch_size 16 --output_dir results/nrc_train
python run_downstream.py circ train dis_w --batch_size 16 --output_dir results/circ_train
python run_downstream.py spl train dis_w --batch_size 16 --output_dir results/spl_train
python tasks/IRES/run_ires.py train dis_w --batch_size 16 --gradient_accumulation_steps 2 --output_dir results/ires_train
python tasks/rinalmo_downstream/run_rinalmo.py --task family --batch_size 16 --gradient_accumulation_steps 2 --output_dir results/rinalmo_family_train
```

RiNALMo task choices: family, localization, splice, ires. Defaults are defined in run_rinalmo.py; evaluation restores configuration embedded in best_model.pt.

The unified splice training launcher currently retains historical GPU FP16 and step-based validation settings. It is not the FP32, epoch-validation efficiency configuration. Match precision, effective batch size, gradient checkpointing, and validation frequency before comparing training efficiency.

IRES defaults are in migration/ires_models.json. Historical self-model checkpoints include test-selected runs; RiNALMo selects on validation. These are different selection protocols.

## Distillation

```shell
python -m src.structRFM.train_distillation --tag mlm --use_tree_adaptive_weight --teacher_model_path model/structRFM_86M --data_path data/pretraining/RNAcentral_512_SS_connects.csv --run_name results/pretrain_adaptive
```

The student is randomly initialized with ordinary MLM masking. Omit --use_tree_adaptive_weight for equal weighting. Historical distributed launch arguments and complete ablation training configurations are pending.

## Evaluation notes

See README for batch evaluation and Rfam commands. LoRA adapters are merged before inference benchmarking. Compare efficiency on the same hardware and input settings. PyTorch allocated memory measures active allocations; reserved memory includes its cache pool.

Rfam feature extraction timing excludes cosine scoring and feature export. Repeat --features NAME FILE to compare models. Threshold-selected F1 on the evaluation pairs is exploratory, not a held-out threshold estimate.

Clean-environment installation, full evaluation, and retraining remain to be validated.
