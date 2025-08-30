""" Get Precision, Recall, Density, and Coverage following
convention from `NoisyTwins: Class-Consistent and Diverse
Image Generation through StyleGANs` by Rangwani et al.
Check also https://github.com/val-iisc/NoisyTwins."""

import numpy as np

from . import metric_utils


def compute_prdc(opts, max_real, num_gen, k=5):
    detector_url = metric_utils.get_feature_detector_url(opts.detector)
    detector_kwargs = dict(return_features=True)

    feats_real = metric_utils.compute_feature_stats_for_dataset(
        opts=opts, detector_url=detector_url, detector_kwargs=detector_kwargs,
        rel_lo=0, rel_hi=0, capture_all=True, max_items=max_real).get_all()

    feats_gen = metric_utils.compute_feature_stats_for_generator(
        opts=opts, detector_url=detector_url, detector_kwargs=detector_kwargs,
        rel_lo=0, rel_hi=1, capture_all=True, max_items=num_gen).get_all()
    
    if opts.rank != 0:
        return float('nan')
    
    dist_real = metric_utils.knn_dist(feats_real, k=k)
    dist_gen = metric_utils.knn_dist(feats_gen, k=k)
    dist_between = metric_utils.pairwise_dist(feats_real, feats_gen)
    i = 0
    dist_gen_expanded = np.expand_dims(dist_gen, 0)
    prec = np.zeros((feats_gen.shape[0],), dtype=np.bool_)
    recall = 0
    density = 0
    coverage = 0

    for chunk in dist_between:
        chunk_size = chunk.shape[0]
        prec = prec | np.any(chunk < np.expand_dims(dist_real[i : i + chunk_size], axis=1), axis=0)
        recall += np.sum(np.any(chunk < dist_gen_expanded, axis=1), axis=0)
        density += np.sum(chunk < np.expand_dims(dist_real[i : i + chunk_size], axis=1), axis=(0, 1))
        coverage += np.sum(np.min(chunk, axis=1) < dist_real[i : i + chunk_size])
        i += chunk_size

    metrics = {}
    metrics["precision"] = np.mean(prec, axis=0)
    metrics["recall"] = recall / dist_real.shape[0]
    metrics["density"] = (1. / float(k)) * (density / dist_gen.shape[0])
    metrics["coverage"] = coverage / dist_real.shape[0]
    return metrics