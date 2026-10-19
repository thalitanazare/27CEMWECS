"""Expert risk-rank agreement and wave-cluster uncertainty."""
import itertools
import numpy as np
from scipy.stats import rankdata
from ensemble import risk_labels,softmax
def identity(record):
    return {key: record.get(key) for key in ['layout', 'state', 'policy', 'test_kind', 'file']}


def diagnostics(risk, prediction, record, norm, mask, model):
    starts = np.arange(128, len(record['y']) - 32, 8)
    realised = risk_labels(prediction, record, norm, mask, starts)
    estimated = risk[starts]
    left = rankdata(estimated, axis=1) - (estimated.shape[1] + 1) / 2
    right = rankdata(realised, axis=1) - (realised.shape[1] + 1) / 2
    denominator = np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1)
    correlations = np.divide((left * right).sum(1), denominator,
                             out=np.full(len(starts), np.nan), where=denominator > 0)
    weights = ((1-model.shrinkage) * softmax(-estimated/model.temperature)
               + model.shrinkage * model.prior)
    entropy = -(weights * np.log(np.maximum(weights, 1e-300))).sum(1)
    actual_best = realised.argmin(1)
    predicted_best = estimated.argmin(1)
    row = identity(record)
    row.update({
        'windows': len(starts), 'experts': prediction.shape[1],
        'mean_spearman': float(np.nanmean(correlations)),
        'median_spearman': float(np.nanmedian(correlations)),
        'undefined_correlations': int(np.isnan(correlations).sum()),
        'top_expert_match_fraction': float(np.mean(actual_best == predicted_best)),
        'predicted_argmin_tie_fraction': float(np.mean((estimated == estimated.min(1)[:, None]).sum(1) > 1)),
        'mean_weight_entropy': float(entropy.mean()),
        'mean_normalised_entropy': float(entropy.mean()/np.log(prediction.shape[1])),
        'mean_max_weight': float(weights.max(1).mean()),
        'mean_effective_experts': float(np.exp(entropy).mean()),
    })
    return row

def clustered(rows, seed=0, draws=10000):
    """Clusters: (state, policy); PI uses realisation 1 and UM realisation 2."""
    clusters = {}
    for r in rows:
        clusters.setdefault((r['state'], r['policy']), []).append(float(r['reduction_percent']))
    keys = sorted(clusters)
    groups = [np.array(clusters[k]) for k in keys]
    allv = np.concatenate(groups)
    rng = np.random.default_rng(seed)
    boot = [np.median(np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))]))
            for _ in range(draws)]
    means = np.array([g.mean() for g in groups])
    observed = means.mean()
    flips = np.array([np.mean(means * s) for s in itertools.product([1, -1], repeat=len(means))])
    p = float(np.mean(np.abs(flips) >= abs(observed) - 1e-12))
    return {'records': len(allv), 'clusters': len(groups), 'median_reduction_percent': float(np.median(allv)),
            'cluster_ci_low': float(np.percentile(boot, 2.5)), 'cluster_ci_high': float(np.percentile(boot, 97.5)),
            'clusters_favouring_ensemble': int((means > 0).sum()),
            'mean_of_cluster_means': float(observed), 'signflip_p_two_sided': p,
            'signflip_minimum_attainable_p': 2 / 2 ** len(groups)}
