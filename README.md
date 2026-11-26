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

## Downloads

Download the current code from the repository's main branch using Code > Download ZIP, or:

```shell
git clone https://github.com/sicheng-xiang/DistilncRFM.git
cd DistilncRFM
```

The automatic source archives attached to v1.0.0 refer to the initial commit; use main for the complete code.

| Archive | Contents |
|---|---|
| [base_models.zip](https://github.com/sicheng-xiang/DistilncRFM/releases/download/v1.0.0/base_models.zip) | Five main base models |
| [nrc_checkpoints.zip](https://github.com/sicheng-xiang/DistilncRFM/releases/download/v1.0.0/nrc_checkpoints.zip) | RNA family checkpoints |
| [localization_checkpoints.zip](https://github.com/sicheng-xiang/DistilncRFM/releases/download/v1.0.0/localization_checkpoints.zip) | Localization checkpoints |
| [splice_checkpoints.zip](https://github.com/sicheng-xiang/DistilncRFM/releases/download/v1.0.0/splice_checkpoints.zip) | Splice-site checkpoints |
| [ires_checkpoints.zip](https://github.com/sicheng-xiang/DistilncRFM/releases/download/v1.0.0/ires_checkpoints.zip) | IRES checkpoints |
| [datasets_20261126_125410.zip](https://github.com/sicheng-xiang/DistilncRFM/releases/download/v1.0.0/datasets_20261126_125410.zip) | Four downstream datasets and Rfam inputs |
| [ablation_models.zip](https://github.com/sicheng-xiang/DistilncRFM/releases/download/v1.0.0/ablation_models.zip) | Optional ablation base models |

Extract each required archive into the repository root, preserving model/, tasks/, and data/ paths. Do not create an extra enclosing folder. Evaluation of all tasks requires the base, four checkpoint, and dataset archives.

[SHA256SUMS.txt](https://github.com/sicheng-xiang/DistilncRFM/releases/download/v1.0.0/SHA256SUMS.txt) covers the six model/checkpoint ZIP archives; it does not include the subsequently added dataset archive.

## Evaluation

```shell
python evaluate_downstream.py --device cuda:0 --batch_size 16
```

This evaluates all five models. Select a task with --tasks nrc, circ, spl, or ires. Results and inference efficiency are saved under results/, including summary.json.

Base models belong in model/; downstream weights belong in each task's checkpoints/ directory. Aliases teacher, student, dis, and dis_w map to structRFM_86M, structRFM_11M, DistilncRFM-EW, and DistilncRFM. RiNALMo-Micro is the external baseline. Download weights and datasets from the release assets below.

## Rfam

```shell
python tasks/zeroshot/extract_rfam_features.py --model dis_w --device cuda:0 --batch_size 16 --save_features --output_dir results/rfam_dis_w
python tasks/zeroshot/evaluate_rfam_homology.py --features DistilncRFM results/rfam_dis_w/dis_w_features_fft128.pkl --output-dir results/rfam_metrics
```

## Pretraining data

The original structRFM RNAcentral/BPfold sequence-structure dataset is available from [Zenodo](https://doi.org/10.5281/zenodo.16754363) (approximately 21 million pairs, sequence length up to 512 nt; about 4.5 GB according to the upstream README). It is not bundled here.

Upstream v0.0.8 also announces MUSES-based retraining and links to [a separate dataset record](https://zenodo.org/records/18272040). Do not silently substitute that dataset for the earlier BPfold-based corpus. The exact file/version used for our historical training has not yet been verified against the local original download.

## Documentation

- [Training, data layout, and protocol limitations](docs/REPRODUCING.md)
- [Migration map](docs/MIGRATION.md)
- [Release checklist](docs/RELEASE_READINESS.md)

Some historical checkpoints were selected on test scores. Full retraining and clean-environment validation are pending.

## License

[MIT](LICENSE), retaining the upstream structRFM copyright notice. Third-party components retain their own licenses, including external_models/RiNALMo/LICENSE. Weights and datasets retain their respective terms.
