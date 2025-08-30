# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Frechet Inception Distance (FID) from the paper
"GANs trained by a two time-scale update rule converge to a local Nash
equilibrium". Matches the original implementation by Heusel et al. at
https://github.com/bioinf-jku/TTUR/blob/master/fid.py"""

import numpy as np
import scipy.linalg

from . import metric_utils
import dnnlib

#----------------------------------------------------------------------------

def compute_fid(opts, max_real, num_gen):
    detector_url = metric_utils.get_feature_detector_url(opts.detector)
    detector_kwargs = dict(return_features=True) if "clip" not in opts.detector else dict()

    mu_real, sigma_real = metric_utils.compute_feature_stats_for_dataset(
        opts=opts, detector_url=detector_url, detector_kwargs=detector_kwargs,
        rel_lo=0, rel_hi=0, capture_mean_cov=True, max_items=max_real).get_mean_cov()

    mu_gen, sigma_gen = metric_utils.compute_feature_stats_for_generator(
        opts=opts, detector_url=detector_url, detector_kwargs=detector_kwargs,
        rel_lo=0, rel_hi=1, capture_mean_cov=True, max_items=num_gen).get_mean_cov()

    if opts.rank != 0:
        return float('nan')

    m = np.square(mu_gen - mu_real).sum()
    s, _ = scipy.linalg.sqrtm(np.dot(sigma_gen, sigma_real), disp=False) # pylint: disable=no-member
    fid = np.real(m + np.trace(sigma_gen + sigma_real - s * 2))
    return float(fid)

#----------------------------------------------------------------------------

def compute_fid_by_class(opts, max_real, num_gen):
    """Computes the FID separately for each class. These values are then used
    to calculate the iFID (average FID over all classes)."""
    detector_url = metric_utils.get_feature_detector_url(opts.detector)
    detector_kwargs = dict(return_features=True) if "clip" not in opts.detector else dict()
    dataset = dnnlib.util.construct_class_by_name(**opts.dataset_kwargs)
    assert dataset.has_labels
    classes = list(range(dataset.label_dim))
    n_digits = len(str(classes[-1]))

    fids_by_class = {}

    for c in classes:
        label_mu_real, label_sigma_real = metric_utils.compute_feature_stats_for_dataset(
            opts=opts, detector_url=detector_url, detector_kwargs=detector_kwargs,
            rel_lo=0, rel_hi=0, capture_mean_cov=True, max_items=max_real, label=c).get_mean_cov()
        
        label_mu_gen, label_sigma_gen = metric_utils.compute_feature_stats_for_generator(
            opts=opts, detector_url=detector_url, detector_kwargs=detector_kwargs,
            rel_lo=0, rel_hi=1, capture_mean_cov=True, max_items=num_gen, label=c).get_mean_cov()

        m = np.square(label_mu_gen - label_mu_real).sum()
        s, _ = scipy.linalg.sqrtm(np.dot(label_sigma_gen, label_sigma_real), disp=False)

        fid = np.real(m + np.trace(label_sigma_gen + label_sigma_real - s * 2))
        fids_by_class[f"class_{str(c).zfill(n_digits)}"] = fid

    return fids_by_class

def compute_fid_specific_classes(opts, max_real, num_gen):
    """Computes FID for a subset of the classes. By default, this corresponds
    to 5 head, 5 middle, and 10 tail classes. Useful when there is a large
    number of classes and tracking the FID for all of them during training
    is too expensive."""
    detector_url = metric_utils.get_feature_detector_url(opts.detector)
    detector_kwargs = dict(return_features=True) if "clip" not in opts.detector else dict()
    dataset = dnnlib.util.construct_class_by_name(**opts.dataset_kwargs)
    assert dataset.has_labels

    # Get sample counts by label, sorted in descending order (head classes first)
    labels = dataset.get_interesting_classes()
    n_digits = len(str(max(labels)))

    fids_by_class = {}

    for l in labels:
        label_mu_real, label_sigma_real = metric_utils.compute_feature_stats_for_dataset(
            opts=opts, detector_url=detector_url, detector_kwargs=detector_kwargs,
            rel_lo=0, rel_hi=0, capture_mean_cov=True, max_items=max_real, label=l).get_mean_cov()
        
        label_mu_gen, label_sigma_gen = metric_utils.compute_feature_stats_for_generator(
            opts=opts, detector_url=detector_url, detector_kwargs=detector_kwargs,
            rel_lo=0, rel_hi=1, capture_mean_cov=True, max_items=num_gen, label=l).get_mean_cov()

        m = np.square(label_mu_gen - label_mu_real).sum()
        s, _ = scipy.linalg.sqrtm(np.dot(label_sigma_gen, label_sigma_real), disp=False)

        fid = np.real(m + np.trace(label_sigma_gen + label_sigma_real - s * 2))
        fids_by_class[f"class_{str(l).zfill(n_digits)}"] = fid

    return fids_by_class

#----------------------------------------------------------------------------

def compute_ifid(opts, max_real, num_gen, tail_percentages=None):
    """If `tail_percentages` is not provided, simply computes the iFID.
    Otherwise, computes the iFID of the tail classes necessary to make up the
    given percentage of the daa"""
    detector_url = metric_utils.get_feature_detector_url(opts.detector)
    detector_kwargs = dict(return_features=True) if "clip" not in opts.detector else dict()
    dataset = dnnlib.util.construct_class_by_name(**opts.dataset_kwargs)
    assert dataset.has_labels
    dataset_n_classes = dataset.label_dim

    if not tail_percentages:
        # Compute iFID for all classes
        n_classes = dataset_n_classes
        classes = list(range(n_classes))
    else:
        # Compute iFID for a subset of classes whose samples comprise
        # up to a given percentage of the training data
        assert all(0.0 < p < 1.0 for p in tail_percentages)
        assert tail_percentages == sorted(tail_percentages)
        # Get sample counts by label, sorted in ascending order (tail classes first)
        label_counts = dataset.get_label_counts(sort=True, reverse=False)

        # Add classes until the given sample percentage is reached (exclusive)
        classes = []
        tails = []
        sample_count = 0
        tmp = tail_percentages.copy()
        p = tail_percentages.pop(0)
        for label, count in label_counts.items():
            if len(tails) == len(tmp):
                # Remove extra class added in the last iteration
                classes.pop()
                break

            # Account for the case where two percentages give the same amount of classes
            while sample_count + count > round(len(dataset) * p):
                tails.append(classes.copy())
                if tail_percentages:
                    p = tail_percentages.pop(0)
                else:
                    break

            classes.append(label)
            sample_count += count

        n_classes = len(classes)
        assert classes
        tail_percentages = tmp

    fid_by_class = np.zeros(dataset_n_classes)

    for label in classes:
        label_mu_real, label_sigma_real = metric_utils.compute_feature_stats_for_dataset(
            opts=opts, detector_url=detector_url, detector_kwargs=detector_kwargs,
            rel_lo=0, rel_hi=0, capture_mean_cov=True, max_items=max_real, label=label).get_mean_cov()
        
        label_mu_gen, label_sigma_gen = metric_utils.compute_feature_stats_for_generator(
            opts=opts, detector_url=detector_url, detector_kwargs=detector_kwargs,
            rel_lo=0, rel_hi=1, capture_mean_cov=True, max_items=num_gen, label=label).get_mean_cov()

        m = np.square(label_mu_gen - label_mu_real).sum()
        s, _ = scipy.linalg.sqrtm(np.dot(label_sigma_gen, label_sigma_real), disp=False)

        fid_by_class[label] = np.real(m + np.trace(label_sigma_gen + label_sigma_real - s * 2))

    if tail_percentages:
        results = {}
        for t, p in zip(tails, tail_percentages):
            p_str = str(p)
            key = f"ifidclip{num_gen // 1000}k_tail_{p_str[p_str.index('.') + 1:].ljust(2, '0')}"
            results[key] = fid_by_class[t].sum() / len(t) if opts.rank == 0 else float('nan')
        return results
    else:
        return fid_by_class.sum() / n_classes if opts.rank == 0 else float('nan')

#----------------------------------------------------------------------------
