# DistilncRFM

RNA foundation model distillation with four downstream tasks and Rfam zero-shot homology evaluation.

## Setup

Use Python 3.11 and install PyTorch separately for your NVIDIA driver.

```shell
conda create -n distilncrfm python=3.11 pip
conda activate distilncrfm
python -m pip install -r requirements.txt
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

## Evaluation

```shell
python evaluate_downstream.py --device cuda:0 --batch_size 16
```

This evaluates all five models. Select a task with --tasks nrc, circ, spl, or ires. Results and inference efficiency are saved under results/, including summary.json.

Base models belong in model/; downstream weights belong in each task's checkpoints/ directory. Aliases teacher, student, dis, and dis_w map to structRFM_86M, structRFM_11M, DistilncRFM-EW, and DistilncRFM. RiNALMo-Micro is the external baseline. Weight download links are pending.

## Rfam

```shell
python tasks/zeroshot/extract_rfam_features.py --model dis_w --device cuda:0 --batch_size 16 --save_features --output_dir results/rfam_dis_w
python tasks/zeroshot/evaluate_rfam_homology.py --features DistilncRFM results/rfam_dis_w/dis_w_features_fft128.pkl --output-dir results/rfam_metrics
```

## Documentation

- [Training, data layout, and protocol limitations](docs/REPRODUCING.md)
- [Migration map](docs/MIGRATION.md)
- [Release checklist](docs/RELEASE_READINESS.md)

Some historical checkpoints were selected on test scores. Full retraining and clean-environment validation are pending.

## License

[MIT](LICENSE), retaining the upstream structRFM copyright notice. Third-party components retain their own licenses, including external_models/RiNALMo/LICENSE. Weights and datasets retain their respective terms.
