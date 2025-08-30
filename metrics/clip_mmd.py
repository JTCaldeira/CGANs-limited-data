"""
Implements the CLIP-MMD metric proposed in Jayasumana et al. (2024):
"Rethinking FID: Towards a Better Evaluation Metric for Image Generation"
https://arxiv.org/abs/2401.09603

Code for the `mmd` function taken from the associated GitHub:
https://github.com/google-research/google-research/tree/master/cmmd
"""

import numpy as np
from sklearn.gaussian_process.kernels import RBF

from . import metric_utils


def mmd(x, y, sigma=10.0, scale=1000.0):
	gamma = 1 / (2 * sigma ** 2)

	x_sqnorms = np.diag(np.matmul(x, x.T))
	y_sqnorms = np.diag(np.matmul(y, y.T))

	k_xx = np.mean(
		np.exp(
			-gamma * (
				-2 * np.matmul(x, x.T)
				+ np.expand_dims(x_sqnorms, 1)
				+ np.expand_dims(x_sqnorms, 0)
			)
		)
	)

	k_xy = np.mean(
		np.exp(
			-gamma * (
				-2 * np.matmul(x, y.T)
				+ np.expand_dims(x_sqnorms, 1)
				+ np.expand_dims(y_sqnorms, 0)
			)
		)
	)

	k_yy = np.mean(
		np.exp(
			-gamma * (
				-2 * np.matmul(y, y.T)
				+ np.expand_dims(y_sqnorms, 1)
				+ np.expand_dims(y_sqnorms, 0)
			)
		)
	)

	return scale * (k_xx + k_yy - 2 * k_xy)


def compute_cmmd(opts, max_real, num_gen):
    detector_url = metric_utils.get_feature_detector_url(opts.detector)
    assert "clip" in opts.detector
    detector_kwargs = dict()

    real_features = metric_utils.compute_feature_stats_for_dataset(
        opts=opts, detector_url=detector_url, detector_kwargs=detector_kwargs,
        rel_lo=0, rel_hi=0, capture_all=True, max_items=max_real).get_all()

    gen_features = metric_utils.compute_feature_stats_for_generator(
            opts=opts, detector_url=detector_url, detector_kwargs=detector_kwargs,
            rel_lo=0, rel_hi=1, capture_all=True, max_items=num_gen).get_all()

    cmmd = mmd(real_features, gen_features)

    return cmmd