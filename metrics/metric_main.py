# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import os
import time
import json
import torch
import dnnlib
import numpy as np

from . import (class_var, clip_mmd, d_predictions, frechet_inception_distance,
                inception_score, kernel_inception_distance, metric_utils,
                perceptual_path_length, prdc, precision_recall)

#----------------------------------------------------------------------------

_metric_dict = dict() # name => fn

def register_metric(fn):
    assert callable(fn)
    _metric_dict[fn.__name__] = fn
    return fn

def is_valid_metric(metric):
    return metric in _metric_dict

def list_valid_metrics():
    return list(_metric_dict.keys())

#----------------------------------------------------------------------------

def calc_metric(metric, **kwargs): # See metric_utils.MetricOptions for the full list of arguments.
    assert is_valid_metric(metric)
    opts = metric_utils.MetricOptions(**kwargs)

    # Calculate.
    start_time = time.time()
    results = _metric_dict[metric](opts)
    total_time = time.time() - start_time

    # Broadcast results.
    for key, value in list(results.items()):
        if opts.num_gpus > 1:
            value = torch.as_tensor(value, dtype=torch.float64, device=opts.device)
            torch.distributed.broadcast(tensor=value, src=0)
            value = float(value.cpu())
        results[key] = value

    # Decorate with metadata.
    return dnnlib.EasyDict(
        results         = dnnlib.EasyDict(results),
        metric          = metric,
        total_time      = total_time,
        total_time_str  = dnnlib.util.format_time(total_time),
        num_gpus        = opts.num_gpus,
    )

#----------------------------------------------------------------------------

def report_metric(result_dict, run_dir=None, snapshot_pkl=None):
    metric = result_dict['metric']
    assert is_valid_metric(metric)
    if run_dir is not None and snapshot_pkl is not None:
        snapshot_pkl = os.path.relpath(snapshot_pkl, run_dir)

    jsonl_line = json.dumps(dict(result_dict, snapshot_pkl=snapshot_pkl, timestamp=time.time()))
    print(jsonl_line)
    if run_dir is not None and os.path.isdir(run_dir):
        with open(os.path.join(run_dir, f'metric-{metric}.jsonl'), 'at') as f:
            f.write(jsonl_line + '\n')

#----------------------------------------------------------------------------
# Primary metrics.

@register_metric
def fid50k_full(opts):
    opts.dataset_kwargs.update(max_size=None, xflip=False)
    fid = frechet_inception_distance.compute_fid(opts, max_real=None, num_gen=50000)
    return dict(fid50k_full=fid)

@register_metric
def fidclip_byclass(opts):
    opts.detector = 'clip'
    opts.dataset_kwargs.update(max_size=None, xflip=False)
    fids_by_class = frechet_inception_distance.compute_fid_by_class(opts, max_real=None, num_gen=5000)
    fids_by_class[f"ifidclip5k_full"] = np.array(list(fids_by_class.values())).mean()
    return fids_by_class

@register_metric
def fid_specific_classes(opts):
    opts.detector = 'clip'
    opts.dataset_kwargs.update(max_size=None, xflip=False)
    fids_specific = frechet_inception_distance.compute_fid_specific_classes(opts, max_real=None, num_gen=5000)
    return fids_specific

@register_metric
def fidclip50k_full(opts):
    opts.detector = 'clip'
    opts.dataset_kwargs.update(max_size=None, xflip=False)
    fid_clip = fid50k_full(opts)
    fid_clip['fidclip50k_full'] = fid_clip.pop('fid50k_full')
    return fid_clip

@register_metric
def ifid5k_full(opts):
    opts.dataset_kwargs.update(max_size=None, xflip=False)
    ifid = frechet_inception_distance.compute_ifid(opts, max_real=None, num_gen=5000)
    return dict(ifid5k_full=ifid)

@register_metric
def ifidclip5k_full(opts):
    opts.detector = 'clip'
    opts.dataset_kwargs.update(max_size=None, xflip=False)
    ifid_clip = ifid5k_full(opts)
    ifid_clip['ifidclip5k_full'] = ifid_clip.pop('ifid5k_full')
    return ifid_clip

@register_metric
def ifidclip5k_tail(opts):
    opts.detector = 'clip'
    opts.dataset_kwargs.update(max_size=None, xflip=False)
    # Dict with an entry for each percentage of tail samples
    ifid_clip_tail = frechet_inception_distance.compute_ifid(opts, max_real=None, num_gen=5000, tail_percentages=[0.02, 0.05, 0.10, 0.20])
    return ifid_clip_tail

@register_metric
def kid50k_full(opts):
    opts.dataset_kwargs.update(max_size=None, xflip=False)
    kid = kernel_inception_distance.compute_kid(opts, max_real=1000000, num_gen=50000, num_subsets=100, max_subset_size=1000)
    return dict(kid50k_full=kid)

@register_metric
def prdc50k_full(opts):
    opts.detector = 'vgg16'
    opts.dataset_kwargs.update(max_size=None, xflip=False)
    return prdc.compute_prdc(opts, max_real=None, num_gen=50000, k=5)

@register_metric
def pr50k3_full(opts):
    opts.detector = 'vgg16'
    opts.dataset_kwargs.update(max_size=None, xflip=False)
    precision, recall = precision_recall.compute_pr(opts, max_real=200000, num_gen=50000, nhood_size=3, row_batch_size=10000, col_batch_size=10000)
    return dict(pr50k3_full_precision=precision, pr50k3_full_recall=recall)

@register_metric
def ppl2_wend(opts):
    ppl = perceptual_path_length.compute_ppl(opts, num_samples=50000, epsilon=1e-4, space='w', sampling='end', crop=False, batch_size=2)
    return dict(ppl2_wend=ppl)

@register_metric
def is50k(opts):
    opts.dataset_kwargs.update(max_size=None, xflip=False)
    mean, std = inception_score.compute_is(opts, num_gen=50000, num_splits=10)
    return dict(is50k_mean=mean, is50k_std=std)

@register_metric
def cvar(opts):
    opts.dataset_kwargs.update(max_size=None, xflip=False)
    opts.detector = 'clip'
    return class_var.compute_class_var(opts, num_gen_per_class=1000)

@register_metric
def d_preds(opts):
    opts.dataset_kwargs.update(max_size=None, xflip=False)
    return d_predictions.compute_d_predictions(opts, num_gen_per_class=2000, max_real_per_class=5000)

@register_metric
def cmmd30k_30k(opts):
    opts.detector = 'clip'
    opts.dataset_kwargs.update(max_size=None, xflip=False)
    cmmd = clip_mmd.compute_cmmd(opts, max_real=30000, num_gen=30000)
    return dict(cmmd30k_30k=cmmd)

#----------------------------------------------------------------------------
# Legacy metrics.

@register_metric
def fid50k(opts):
    opts.dataset_kwargs.update(max_size=None)
    fid = frechet_inception_distance.compute_fid(opts, max_real=50000, num_gen=50000)
    return dict(fid50k=fid)

@register_metric
def kid50k(opts):
    opts.dataset_kwargs.update(max_size=None)
    kid = kernel_inception_distance.compute_kid(opts, max_real=50000, num_gen=50000, num_subsets=100, max_subset_size=1000)
    return dict(kid50k=kid)

@register_metric
def pr50k3(opts):
    opts.dataset_kwargs.update(max_size=None)
    precision, recall = precision_recall.compute_pr(opts, max_real=50000, num_gen=50000, nhood_size=3, row_batch_size=10000, col_batch_size=10000)
    return dict(pr50k3_precision=precision, pr50k3_recall=recall)

@register_metric
def ppl_zfull(opts):
    ppl = perceptual_path_length.compute_ppl(opts, num_samples=50000, epsilon=1e-4, space='z', sampling='full', crop=True, batch_size=2)
    return dict(ppl_zfull=ppl)

@register_metric
def ppl_wfull(opts):
    ppl = perceptual_path_length.compute_ppl(opts, num_samples=50000, epsilon=1e-4, space='w', sampling='full', crop=True, batch_size=2)
    return dict(ppl_wfull=ppl)

@register_metric
def ppl_zend(opts):
    ppl = perceptual_path_length.compute_ppl(opts, num_samples=50000, epsilon=1e-4, space='z', sampling='end', crop=True, batch_size=2)
    return dict(ppl_zend=ppl)

@register_metric
def ppl_wend(opts):
    ppl = perceptual_path_length.compute_ppl(opts, num_samples=50000, epsilon=1e-4, space='w', sampling='end', crop=True, batch_size=2)
    return dict(ppl_wend=ppl)

#----------------------------------------------------------------------------
