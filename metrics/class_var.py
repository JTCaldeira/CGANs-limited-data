import numpy as np

from . import metric_utils
import dnnlib


def compute_class_var(opts, num_gen_per_class):
    """Compute the variance of style vectors and of the (e.g., CLIP) features
    of the resulting samples. This is done for a subset of the classes."""
    detector_url = metric_utils.get_feature_detector_url(opts.detector)
    detector_kwargs = dict(return_features=True) if "clip" not in opts.detector else dict()

    dataset = dnnlib.util.construct_class_by_name(**opts.dataset_kwargs)
    assert dataset.has_labels
    classes = dataset.get_interesting_classes()
    n_class_digits = len(str(max(classes)))
    results = {}

    for c in classes:
        gen_features, ws = metric_utils.compute_feature_stats_for_generator(
            opts=opts, detector_url=detector_url, detector_kwargs=detector_kwargs,
            rel_lo=0, rel_hi=1, capture_all=True, return_ws=True, max_items=num_gen_per_class,
            label=c, rand=np.random.RandomState(123))

        gen_features, ws = gen_features.get_all(), ws.cpu().numpy()

        results[f"var_feat_c{str(c).zfill(n_class_digits)}"] = np.trace(np.cov(gen_features.T))
        results[f"var_ws_c{str(c).zfill(n_class_digits)}"] = np.trace(np.cov(ws.T))

    return results
