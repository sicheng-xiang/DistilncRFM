"""Evaluate homology from features exported by extract_rfam_features.py."""
import argparse
import hashlib
import json
import pickle
import random
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
from sklearn.metrics import (roc_auc_score, precision_recall_curve, auc,
                             average_precision_score)

DEFAULT_SEEDS = (42, 43, 44, 45, 46)
DEFAULT_N_SAMPLES = 100000  # Per class, per seed.
OR_LIMITS = (-0.5, 1.0)  # Same interval as the original evaluation.
OR_BINS = 200  # Original seaborn KDE default gridsize; independent of display limits.
THRESHOLD_STEP = 0.01  # Original maximum-F1 threshold grid, NOT a held-out threshold.


def sample_pairs(keys, n_samples, seed):
    """Preserve original family-uniform sampling, including positive family size >2."""
    families = {}
    for i, key in enumerate(keys):
        families.setdefault(key.split('_')[0], []).append(i)
    rfs = list(families)
    if len(rfs) < 2 or not any(len(v) > 2 for v in families.values()):
        raise ValueError('Need at least two families and a positive family with >2 sequences.')
    rng = random.Random(seed)
    positive, negative = [], []
    while len(positive) < n_samples:
        rf = rng.choice(rfs)
        if len(families[rf]) > 2:
            positive.append(rng.sample(families[rf], 2))
            a, b = rng.sample(rfs, 2)
            negative.append((rng.choice(families[a]), rng.choice(families[b])))
    return np.asarray(positive, dtype=np.int64), np.asarray(negative, dtype=np.int64)


def cosine_scores(embeddings, pairs):
    # Original helper allocated all pairs x 2 x dimension at once; chunking is equivalent.
    scores = np.empty(len(pairs), dtype=np.float64)
    for start in range(0, len(pairs), 4096):
        batch = pairs[start:start + 4096]
        a = np.asarray(embeddings[batch[:, 0]], dtype=np.float64)
        b = np.asarray(embeddings[batch[:, 1]], dtype=np.float64)
        denominator = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
        if np.any(denominator == 0):
            raise ValueError('Zero-norm embedding encountered; cannot define cosine similarity.')
        scores[start:start + len(batch)] = np.sum(a * b, axis=1) / denominator
    if not np.isfinite(scores).all():
        raise ValueError('Non-finite cosine scores.')
    return scores


def calculate_metrics(pos, neg):
    grid = np.linspace(*OR_LIMITS, OR_BINS)
    dp, dn = gaussian_kde(pos)(grid), gaussian_kde(neg)(grid)
    overlap = np.minimum(dp, dn).sum() / np.maximum(dp, dn).sum()
    scores = np.concatenate([pos, neg])
    labels = np.concatenate([np.ones(len(pos), dtype=int), np.zeros(len(neg), dtype=int)])
    precision, recall, _ = precision_recall_curve(labels, scores)
    # Retain the original 0..1 grid and >= convention for exploratory threshold metrics.
    thresholds = np.arange(0, 1 + THRESHOLD_STEP / 2, THRESHOLD_STEP)
    tp = len(pos) - np.searchsorted(np.sort(pos), thresholds, side='left')
    fp = len(neg) - np.searchsorted(np.sort(neg), thresholds, side='left')
    fn, tn = len(pos) - tp, len(neg) - fp
    prec = np.divide(tp, tp + fp, out=np.zeros_like(tp, dtype=float), where=(tp + fp) != 0)
    rec = tp / len(pos)
    f1 = np.divide(2 * tp, 2 * tp + fp + fn,
                   out=np.zeros_like(tp, dtype=float), where=(2 * tp + fp + fn) != 0)
    best = int(np.argmax(f1))
    denominator = np.sqrt((tp.astype(float) + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = np.divide(tp.astype(float) * tn - fp.astype(float) * fn, denominator,
                    out=np.zeros_like(tp, dtype=float), where=denominator != 0)
    return dict(Mean_Pos=float(pos.mean()), Mean_Neg=float(neg.mean()),
                Std_Pos=float(pos.std()), Std_Neg=float(neg.std()),
                Delta_Mean=float(pos.mean() - neg.mean()),
                Delta_Median=float(np.median(pos) - np.median(neg)), OR=float(overlap),
                AUC=float(roc_auc_score(labels, scores)), AUPR=float(auc(recall, precision)),
                Average_Precision=float(average_precision_score(labels, scores)),
                F1_at_best_F1=float(f1[best]), MCC_at_best_F1=float(mcc[best]),
                Precision_at_best_F1=float(prec[best]), Recall_at_best_F1=float(rec[best]),
                Accuracy_at_best_F1=float((tp[best] + tn[best]) / len(scores)),
                Specificity_at_best_F1=float(tn[best] / len(neg)),
                Threshold_best_F1=float(thresholds[best]))


def evaluate_models(models, seeds, n_samples, output_dir, write_tables=True):
    if n_samples <= 0 or not seeds or len(set(seeds)) != len(seeds):
        raise ValueError('Use positive n_samples and distinct seeds.')
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows, scores_by_model = [], {}
    reference_keys = reference_seqs = None
    pair_sets = {}
    for model in models:
        feature = Path(model['path'])
        print('Reading cached features:', model['name'], flush=True)
        with feature.open('rb') as f:
            data = pickle.load(f)
        keys = list(data['keys'])
        seqs = list(data['seqs']) if 'seqs' in data else None
        emb = np.asarray(data['seq_repr'])
        if len(set(keys)) != len(keys) or emb.ndim != 2 or len(emb) != len(keys):
            raise ValueError('Duplicate keys or invalid feature shape: ' + model['name'])
        if not np.isfinite(emb).all():
            raise ValueError('Non-finite embeddings: ' + model['name'])
        if reference_keys is None:
            reference_keys, reference_seqs = keys, seqs
            fingerprint = hashlib.sha256(json.dumps(keys, ensure_ascii=True).encode()).hexdigest()
            if seqs is not None:
                fingerprint += hashlib.sha256(json.dumps(seqs, ensure_ascii=True).encode()).hexdigest()
            for seed in seeds:
                pair_sets[seed] = sample_pairs(keys, n_samples, seed)
                directory = output_dir / f'seed_{seed}'
                directory.mkdir(exist_ok=True)
                np.savez_compressed(directory / 'sampled_pair_indices.npz',
                                    positive=pair_sets[seed][0], negative=pair_sets[seed][1],
                                    keys=np.asarray(keys), seed=seed, n_samples=n_samples)
        elif keys != reference_keys or (reference_seqs is not None and seqs != reference_seqs):
            raise ValueError('Sequence IDs/order/content differ across models: ' + model['name'])
        for seed in seeds:
            directory = output_dir / f'seed_{seed}'
            cache = directory / (model['tag'] + '_scores.npz')
            metadata = json.dumps(dict(version=1, path=str(feature.resolve()),
                size=feature.stat().st_size, mtime=feature.stat().st_mtime_ns,
                dataset=fingerprint, seed=seed, n_samples=n_samples,
                or_limits=OR_LIMITS, or_bins=OR_BINS, threshold_step=THRESHOLD_STEP), sort_keys=True)
            cached = False
            if cache.exists():
                with np.load(cache, allow_pickle=False) as z:
                    if str(z['metadata']) == metadata:
                        pos, neg = z['positive'], z['negative']
                        metrics = json.loads(str(z['metrics']))
                        cached = True
            if not cached:
                pos, neg = [cosine_scores(emb, pairs) for pairs in pair_sets[seed]]
                metrics = calculate_metrics(pos, neg)
                np.savez_compressed(cache, positive=pos, negative=neg,
                                    overlap=metrics['OR'], metrics=json.dumps(metrics), metadata=metadata)
            rows.append(dict(Model=model['name'], Seed=seed, N_pairs_per_class=n_samples, **metrics))
            scores_by_model[(model['name'], seed)] = (pos, neg)
            print(f"  seed={seed}: delta={metrics['Delta_Mean']:.6f}, OR={metrics['OR']:.6f}, AUC={metrics['AUC']:.6f}", flush=True)
        del data, emb
    df = pd.DataFrame(rows)
    if not write_tables:
        return df, scores_by_model
    df.to_csv(output_dir / 'metrics_per_seed.csv', index=False, encoding='utf-8-sig')
    metrics = [c for c in df if c not in ('Model', 'Seed', 'N_pairs_per_class')]
    summary = []
    for name, group in df.groupby('Model', sort=False):
        for metric in metrics:
            summary.append(dict(Model=name, Metric=metric, N_seeds=len(group),
                Mean=group[metric].mean(), SD=group[metric].std(ddof=1),
                Min=group[metric].min(), Max=group[metric].max()))
    summary = pd.DataFrame(summary)
    summary.to_csv(output_dir / 'metrics_summary.csv', index=False, encoding='utf-8-sig')
    formatted = summary.assign(Value=summary.apply(
        lambda r: f"{r['Mean']:.6f} +/- {r['SD']:.6f}" if r['N_seeds'] > 1 else f"{r['Mean']:.6f} (one seed)", axis=1))
    formatted.pivot(index='Model', columns='Metric', values='Value').to_csv(
        output_dir / 'metrics_mean_sd.csv', encoding='utf-8-sig')
    if {'DistilncRFM', 'DistilncRFM-EW'} <= set(df.Model):
        a = df[df.Model == 'DistilncRFM'].set_index('Seed')
        b = df[df.Model == 'DistilncRFM-EW'].set_index('Seed')
        diff = a[metrics] - b[metrics]
        diff.to_csv(output_dir / 'adaptive_minus_equal_per_seed.csv', encoding='utf-8-sig')
        pd.DataFrame([dict(Metric=m, Mean_difference=diff[m].mean(),
            SD_difference=diff[m].std(ddof=1), Min_difference=diff[m].min(),
            Max_difference=diff[m].max()) for m in metrics]).to_csv(
                output_dir / 'adaptive_minus_equal_summary.csv', index=False, encoding='utf-8-sig')
    (output_dir / 'README.txt').write_text(
        'Fixed models; vary only sampled Rfam pairs. This is sampling sensitivity, not training-seed stability.\n'
        'All models use identical pair indices for a seed; IDs/order and available sequences are checked.\n'
        'Positive family size >2; family-uniform sampling matches the original evaluator. Pairs may repeat.\n'
        f'Seeds={list(seeds)}; pairs per class={n_samples}.\n'
        f'OR: Scott KDE, min/max density sum on {OR_LIMITS}, {OR_BINS} points; stored unrounded.\n'
        'Old OR was rounded to 3 decimals before aggregation; new output retains full precision.\n'
        'AUC: sklearn ROC AUC over full scores; AUPR: trapezoidal PR area; AP is separate.\n'
        'Old metricCal PR/PR1/AUC fields are not reused because their fixed threshold integration is incomplete.\n'
        f'Threshold metrics: maximize F1 on these evaluation pairs over 0..1 step {THRESHOLD_STEP}.\n'
        'F1/MCC/precision/recall/accuracy at that threshold are exploratory, not held-out classification estimates.\n'
        'SD uses ddof=1; across-seed SD is not a confidence interval; draws share sequences/families.\n',
        encoding='utf-8')
    return df, scores_by_model


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--features', action='append', nargs=2, required=True,
                        metavar=('MODEL', 'PICKLE'))
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--seeds', type=int, nargs='+', default=list(DEFAULT_SEEDS))
    parser.add_argument('--n-samples', type=int, default=DEFAULT_N_SAMPLES)
    args = parser.parse_args(argv)
    names = [name for name, _ in args.features]
    if len(set(names)) != len(names):
        parser.error('Model names must be unique')
    models = [dict(tag=f'model_{index}', name=name, path=path)
              for index, (name, path) in enumerate(args.features)]
    evaluate_models(models, args.seeds, args.n_samples, args.output_dir)
    print('DONE:', args.output_dir.resolve())


if __name__ == '__main__':
    main()
