# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Train a GAN using the techniques described in the paper
"Training Generative Adversarial Networks with Limited Data"."""

import os
import click
import re
import json
import tempfile
import torch
import dnnlib

from training import training_loop
from metrics import metric_main
from torch_utils import training_stats
from torch_utils import custom_ops

#----------------------------------------------------------------------------

class UserError(Exception):
    pass

#----------------------------------------------------------------------------

def setup_training_loop_kwargs(
    # General options (not included in desc).
    gpus       = None, # Number of GPUs: <int>, default = 1 gpu
    snap       = None, # Snapshot interval: <int>, default = 50 ticks
    metrics    = None, # List of metric names: [], ['fid50k_full'] (default), ...
    seed       = None, # Random seed: <int>, default = 0
    wandb_projname  = None, # Name of the project for wandb logging
    wandb_groupname = None, # Name of the wandb group in which to include this run
    description= None, # Custom suffix to add to the run description
    save       = None, # Whether to save model checkpoints: <bool>, default = True

    # Dataset.
    data       = None, # Training dataset (required): <path>
    data_fname = None, # Name of the dataset metadata file: <str>, default = 'dataset.json'
    cond       = None, # Train conditional model based on dataset labels: <bool>, default = False
    subset     = None, # Train with only N images: <int>, default = all
    mirror     = None, # Augment dataset with x-flips: <bool>, default = False

    # Base config.
    cfg        = None, # Base config: 'auto' (default), 'stylegan2', 'paper256', 'paper512', 'paper1024', 'cifar'
    d_arch     = None, # Discriminator architecture: 'orig', 'skip', 'resnet' (default)
    style_mixing_p  = None, # Override style mixing probability: <float>, default = 0.9
    pl_weight  = None, # Override PLR strength: <float>, default = 2.0
    pl_start_kimg   = None, # Only include PLR loss after `INT` kimg steps, default = 0
    fmaps      = None, # Factor that determines the amount of feature maps, multiplied by 32768: <float>
    gamma      = None, # Override R1 gamma: <float>
    ema        = None, # Override EMA kimg: <int>
    ramp       = None, # Override EMA rampup: <float>
    nmap       = None, # Override number of mapping network layers: <int>
    kimg       = None, # Override training duration: <int>
    batch      = None, # Override batch size: <int>

    # Discriminator augmentation.
    aug        = None, # Augmentation mode: 'ada' (default), 'noaug', 'fixed'
    p          = None, # Specify p for 'fixed' (required): <float>
    target     = None, # Override ADA target for 'ada': <float>, default = depends on aug
    augpipe    = None, # Augmentation pipeline: 'blit', 'geom', 'color', 'filter', 'noise', 'cutout', 'bg', 'bgc' (default), ..., 'bgcfnc'

    # LeCam regularization (Tseng et al., 2021)
    lecam      = None, # Whether to use LeCam regularization: <bool>, default = False
    lecam_lambda = None,    # Strength of the LeCam regularizer: <float>, default = 0.01

    # Transitional CGAN (Shahbazi et al., 2022)
    transition = None, # Whether to use Transitional-CGAN: <bool>, default = False
    t_start_kimg    = None, # When to start the transition to conditional training (in kimg): <int>, default = 2000
    t_end_kimg      = None, # At which point training should be fully conditional (in kimg): <int>, default = 4000

    # GSR (Rangwani et al., 2022)
    gsr        = None, # Whether to use group spectral regularization: <bool>, default = False
    gsr_lambda = None, # Strength of GSR: <float>, default = 0.5

    # NoisyTwins (Rangwani et al., 2023)
    noisytwins     = None, # Whether to use NoisyTwins: <bool>, default = False
    nt_alpha       = None, # "Effective samples": <float>, default = 0.0
    nt_sigma       = None, # Noise scaling factor: <float>, default = 0.5
    nt_gamma       = None, # Relative importance of each of the two terms in the NoisyTwins loss: <float>, default = 0.05
    nt_lambda      = None, # Strength of NoisyTwins regularization: <float>, default = 0.001
    nt_start_kimg  = None, # Only include NoisyTwins loss after `INT` kimg steps, default = 0

    # UTLO (Khorram et al., 2024)
    utlo       = None, # Whether to use UTLO: <bool>, default = False
    res_uc     = None, # Up to which resolution to do unconditional training: <int>, default = 8
    utlo_lambda= None, # Unconditional training objective weight: <float>, default = 0.0

    # Transfer learning.
    resume     = None, # Load previous network: 'noresume' (default), 'ffhq256', 'ffhq512', 'ffhq1024', 'celebahq256', 'lsundog256', <file>, <url>
    freezed    = None, # Freeze-D: <int>, default = 0 discriminator layers

    # Performance options (not included in desc).
    fp32       = None, # Disable mixed-precision training: <bool>, default = False
    nhwc       = None, # Use NHWC memory format with FP16: <bool>, default = False
    allow_tf32 = None, # Allow PyTorch to use TF32 for matmul and convolutions: <bool>, default = False
    nobench    = None, # Disable cuDNN benchmarking: <bool>, default = False
    workers    = None, # Override number of DataLoader workers: <int>, default = 3
):
    args = dnnlib.EasyDict()

    # ---------------------------------------------------------------------------
    # General options: gpus, snap, metrics, seed, wandb_projname, wandb_groupname
    # ---------------------------------------------------------------------------

    if gpus is None:
        gpus = 1
    assert isinstance(gpus, int)
    if not (gpus >= 1 and gpus & (gpus - 1) == 0):
        raise UserError('--gpus must be a power of two')
    args.num_gpus = gpus

    if snap is None:
        snap = 50
    assert isinstance(snap, int)
    if snap < 1:
        raise UserError('--snap must be at least 1')
    args.image_snapshot_ticks = snap
    args.network_snapshot_ticks = snap

    if metrics is None:
        metrics = ['fid50k_full']
    assert isinstance(metrics, list)
    if not all(metric_main.is_valid_metric(metric) for metric in metrics):
        raise UserError('\n'.join(['--metrics can only contain the following values:'] + metric_main.list_valid_metrics()))
    args.metrics = metrics

    if seed is None:
        seed = 0
    assert isinstance(seed, int)
    args.random_seed = seed

    # Either both are specified or neither is specified
    assert wandb_projname and wandb_groupname or not wandb_projname and not wandb_groupname
    args.wandb_projname = wandb_projname
    args.wandb_groupname = wandb_groupname

    # -----------------------------------------------
    # Dataset: data, data_fname, cond, subset, mirror
    # -----------------------------------------------

    if data_fname is None:
        data_fname = 'dataset.json'
    assert isinstance(data_fname, str)

    assert data is not None
    assert isinstance(data, str)
    args.training_set_kwargs = dnnlib.EasyDict(class_name='training.dataset.ImageFolderDataset', path=data, fname=data_fname, use_labels=True, max_size=None, xflip=False)
    args.data_loader_kwargs = dnnlib.EasyDict(pin_memory=True, num_workers=3, prefetch_factor=2)
    try:
        training_set = dnnlib.util.construct_class_by_name(**args.training_set_kwargs) # subclass of training.dataset.Dataset
        args.training_set_kwargs.resolution = training_set.resolution # be explicit about resolution
        args.training_set_kwargs.use_labels = training_set.has_labels # be explicit about labels
        args.training_set_kwargs.max_size = len(training_set) # be explicit about dataset size
        desc = training_set.name
        del training_set # conserve memory
    except IOError as err:
        raise UserError(f'--data: {err}')

    if cond is None:
        cond = False
    assert isinstance(cond, bool)
    if cond:
        if not args.training_set_kwargs.use_labels:
            raise UserError('--cond=True requires labels specified in dataset.json')
        desc += '-cond'
    else:
        args.training_set_kwargs.use_labels = False

    if subset is not None:
        assert isinstance(subset, int)
        if not 1 <= subset <= args.training_set_kwargs.max_size:
            raise UserError(f'--subset must be between 1 and {args.training_set_kwargs.max_size}')
        desc += f'-subset{subset}'
        if subset < args.training_set_kwargs.max_size:
            args.training_set_kwargs.max_size = subset
            args.training_set_kwargs.random_seed = args.random_seed

    if mirror is None:
        mirror = False
    assert isinstance(mirror, bool)
    if mirror:
        desc += '-mirror'
        args.training_set_kwargs.xflip = True

    # ------------------------------------
    # Base config: cfg, gamma, kimg, batch
    # ------------------------------------

    if cfg is None:
        cfg = 'auto'
    assert isinstance(cfg, str)
    desc += f'-{cfg}'

    cfg_specs = {
        'auto':      dict(ref_gpus=-1, kimg=25000,  mb=-1, mbstd=-1, fmaps=-1,  lrate=-1,     gamma=-1,   ema=-1,  ramp=0.05, map=2), # Populated dynamically based on resolution and GPU count.
        'stylegan2': dict(ref_gpus=8,  kimg=25000,  mb=32, mbstd=4,  fmaps=1,   lrate=0.002,  gamma=10,   ema=10,  ramp=None, map=8), # Uses mixed-precision, unlike the original StyleGAN2.
        'paper256':  dict(ref_gpus=8,  kimg=25000,  mb=64, mbstd=8,  fmaps=0.5, lrate=0.0025, gamma=1,    ema=20,  ramp=None, map=8),
        'paper512':  dict(ref_gpus=8,  kimg=25000,  mb=64, mbstd=8,  fmaps=1,   lrate=0.0025, gamma=0.5,  ema=20,  ramp=None, map=8),
        'paper1024': dict(ref_gpus=8,  kimg=25000,  mb=32, mbstd=4,  fmaps=1,   lrate=0.002,  gamma=2,    ema=10,  ramp=None, map=8),
        'cifar':     dict(ref_gpus=2,  kimg=100000, mb=64, mbstd=32, fmaps=1,   lrate=0.0025, gamma=0.01, ema=500, ramp=0.05, map=6),
    }

    assert cfg in cfg_specs
    spec = dnnlib.EasyDict(cfg_specs[cfg])
    if cfg == 'auto':
        desc += f'{gpus:d}'
        spec.ref_gpus = gpus
        res = args.training_set_kwargs.resolution
        spec.mb = max(min(gpus * min(4096 // res, 32), 64), gpus) # keep gpu memory consumption at bay
        spec.mbstd = min(spec.mb // gpus, 4) # other hyperparams behave more predictably if mbstd group size remains fixed
        spec.fmaps = 1 if res >= 512 else 0.5
        spec.lrate = 0.002 if res >= 1024 else 0.0025
        spec.gamma = 0.0002 * (res ** 2) / spec.mb # heuristic formula
        spec.ema = spec.mb * 10 / 32

    if fmaps is not None:
        assert isinstance(fmaps, int) or isinstance(fmaps, float)
        if not fmaps > 0:
            raise UserError('--fmaps must be greater than zero')
        desc += f'-fmaps{fmaps}'
        spec.fmaps = fmaps

    args.G_kwargs = dnnlib.EasyDict(class_name='training.networks.Generator', z_dim=512, w_dim=512, mapping_kwargs=dnnlib.EasyDict(), synthesis_kwargs=dnnlib.EasyDict())
    args.D_kwargs = dnnlib.EasyDict(class_name='training.networks.Discriminator', block_kwargs=dnnlib.EasyDict(), mapping_kwargs=dnnlib.EasyDict(), epilogue_kwargs=dnnlib.EasyDict())
    args.G_kwargs.synthesis_kwargs.channel_base = args.D_kwargs.channel_base = int(spec.fmaps * 32768)
    args.G_kwargs.synthesis_kwargs.channel_max = args.D_kwargs.channel_max = 512
    args.G_kwargs.mapping_kwargs.num_layers = spec.map
    args.G_kwargs.synthesis_kwargs.num_fp16_res = args.D_kwargs.num_fp16_res = 4 # enable mixed-precision training
    args.G_kwargs.synthesis_kwargs.conv_clamp = args.D_kwargs.conv_clamp = 256 # clamp activations to avoid float16 overflow
    args.D_kwargs.epilogue_kwargs.mbstd_group_size = spec.mbstd

    args.G_opt_kwargs = dnnlib.EasyDict(class_name='torch.optim.Adam', lr=spec.lrate, betas=[0,0.99], eps=1e-8)
    args.D_opt_kwargs = dnnlib.EasyDict(class_name='torch.optim.Adam', lr=spec.lrate, betas=[0,0.99], eps=1e-8)
    args.loss_kwargs = dnnlib.EasyDict(class_name='training.loss.StyleGAN2Loss', r1_gamma=spec.gamma)

    args.total_kimg = spec.kimg
    args.batch_size = spec.mb
    args.batch_gpu = spec.mb // spec.ref_gpus
    args.ema_kimg = spec.ema
    args.ema_rampup = spec.ramp

    if cfg == 'cifar':
        args.loss_kwargs.pl_weight = 0 # disable path length regularization
        args.loss_kwargs.style_mixing_prob = 0 # disable style mixing
        args.D_kwargs.architecture = 'orig' # disable residual skip connections

    if d_arch is None:
        d_arch = 'resnet'
    else:
        desc += f'-{d_arch}'
    assert isinstance(d_arch, str)
    assert d_arch in ('orig', 'skip', 'resnet')
    args.D_kwargs.architecture = d_arch

    if style_mixing_p is not None:
        assert isinstance(style_mixing_p, float)
        if not 0 <= style_mixing_p <= 1:
            raise UserError('--style_mixing_p must be a probability')
        desc += f'-smp{style_mixing_p:g}'
        args.loss_kwargs.style_mixing_prob = style_mixing_p

    if pl_weight is not None:
        assert isinstance(pl_weight, float)
        desc += f'-plr{pl_weight:g}'
        args.loss_kwargs.pl_weight = pl_weight

    if pl_start_kimg is not None:
        assert isinstance(pl_start_kimg, int)
        desc += f'-plkimg{pl_start_kimg:d}'
        args.pl_start_kimg = pl_start_kimg

    if gamma is not None:
        assert isinstance(gamma, float)
        if not gamma >= 0:
            raise UserError('--gamma must be non-negative')
        desc += f'-gamma{gamma:g}'
        args.loss_kwargs.r1_gamma = gamma

    if ema is not None:
        assert isinstance(ema, int)
        if not ema >= 0:
            raise UserError('--ema must be non-negative')
        desc += f'-ema{ema:d}'
        args.ema_kimg = ema

    if ramp is not None:
        assert isinstance(ramp, float)
        if not ramp >= 0:
            raise UserError('--ramp must be non-negative')
        if ramp == 0:
            ramp = None
            desc += '-rampNone'
        else:
            desc += f'-ramp{ramp:g}'
        args.ema_rampup = ramp

    if nmap is not None:
        assert isinstance(nmap, int)
        if not nmap >= 0:
            raise UserError('--nmap must be non-negative')
        desc += f'-map{nmap:d}'
        args.G_kwargs.mapping_kwargs.num_layers = nmap

    if kimg is not None:
        assert isinstance(kimg, int)
        if not kimg >= 1:
            raise UserError('--kimg must be at least 1')
        desc += f'-kimg{kimg:d}'
        args.total_kimg = kimg

    if batch is not None:
        assert isinstance(batch, int)
        if not (batch >= 1 and batch % gpus == 0):
            raise UserError('--batch must be at least 1 and divisible by --gpus')
        desc += f'-batch{batch}'
        args.batch_size = batch
        args.batch_gpu = batch // gpus

    # ---------------------------------------------------
    # Discriminator augmentation: aug, p, target, augpipe
    # ---------------------------------------------------

    if aug is None:
        aug = 'ada'
    else:
        assert isinstance(aug, str)
        desc += f'-{aug}'

    if aug == 'ada':
        args.ada_target = 0.6

    elif aug == 'noaug':
        pass

    elif aug == 'fixed':
        if p is None:
            raise UserError(f'--aug={aug} requires specifying --p')

    else:
        raise UserError(f'--aug={aug} not supported')

    if p is not None:
        assert isinstance(p, float)
        if aug != 'fixed':
            raise UserError('--p can only be specified with --aug=fixed')
        if not 0 <= p <= 1:
            raise UserError('--p must be between 0 and 1')
        desc += f'-p{p:g}'
        args.augment_p = p

    if target is not None:
        assert isinstance(target, float)
        if aug != 'ada':
            raise UserError('--target can only be specified with --aug=ada')
        if not 0 <= target <= 1:
            raise UserError('--target must be between 0 and 1')
        desc += f'-target{target:g}'
        args.ada_target = target

    assert augpipe is None or isinstance(augpipe, str)
    if augpipe is None:
        augpipe = 'bgc'
    else:
        if aug == 'noaug':
            raise UserError('--augpipe cannot be specified with --aug=noaug')
        desc += f'-{augpipe}'

    augpipe_specs = {
        'blit':   dict(xflip=1, rotate90=1, xint=1),
        'geom':   dict(scale=1, rotate=1, aniso=1, xfrac=1),
        'color':  dict(brightness=1, contrast=1, lumaflip=1, hue=1, saturation=1),
        'filter': dict(imgfilter=1),
        'noise':  dict(noise=1),
        'cutout': dict(cutout=1),
        'bg':     dict(xflip=1, rotate90=1, xint=1, scale=1, rotate=1, aniso=1, xfrac=1),
        'bgc':    dict(xflip=1, rotate90=1, xint=1, scale=1, rotate=1, aniso=1, xfrac=1, brightness=1, contrast=1, lumaflip=1, hue=1, saturation=1),
        'bgcf':   dict(xflip=1, rotate90=1, xint=1, scale=1, rotate=1, aniso=1, xfrac=1, brightness=1, contrast=1, lumaflip=1, hue=1, saturation=1, imgfilter=1),
        'bgcfn':  dict(xflip=1, rotate90=1, xint=1, scale=1, rotate=1, aniso=1, xfrac=1, brightness=1, contrast=1, lumaflip=1, hue=1, saturation=1, imgfilter=1, noise=1),
        'bgcfnc': dict(xflip=1, rotate90=1, xint=1, scale=1, rotate=1, aniso=1, xfrac=1, brightness=1, contrast=1, lumaflip=1, hue=1, saturation=1, imgfilter=1, noise=1, cutout=1),
    }

    assert augpipe in augpipe_specs
    if aug != 'noaug':
        args.augment_kwargs = dnnlib.EasyDict(class_name='training.augment.AugmentPipe', **augpipe_specs[augpipe])

    # --------------------------
    # LeCam: lecam, lecam_lambda
    # --------------------------
    if lecam is None:
        lecam = False
    if lecam_lambda is None:
        lecam_lambda = 0.01
    assert isinstance(lecam, bool)
    assert isinstance(lecam_lambda, float)
    assert lecam_lambda >= 0

    if lecam:
        args.use_lecam = True
        args.lecam_lambda = lecam_lambda
        desc += f'-lecam:lambda{lecam_lambda}'

    # ------------------------------------------------
    # Transition: transition, t_start_kimg, t_end_kimg
    # ------------------------------------------------
    if transition is None:
        transition = False
    if t_start_kimg is None:
        t_start_kimg = 2000
    if t_end_kimg is None:
        t_end_kimg = 4000
    assert isinstance(t_start_kimg, int)
    assert isinstance(t_end_kimg, int)
    assert cond or not transition
    assert (t_start_kimg >= 0 and t_start_kimg <= t_end_kimg)

    if transition:
        args.use_transition = True
        args.t_start_kimg = t_start_kimg
        args.t_end_kimg = t_end_kimg
        args.G_kwargs.mapping_kwargs.embed_class_info = True
        args.D_kwargs.is_transitional = True
        desc += f'-trans:{t_start_kimg}-{t_end_kimg}'

    # --------------------
    # GSR: gsr, gsr_lambda
    # --------------------
    if gsr is None:
        gsr = False
    if gsr_lambda is None:
        gsr_lambda = 0.5
    assert isinstance(gsr, bool)
    assert isinstance(gsr_lambda, float)
    assert gsr_lambda > 0.0

    if gsr:
        args.use_gsr = True
        args.gsr_lambda = gsr_lambda
        desc += f'-gsr:lambda{gsr_lambda}'

    # ---------------------------------------------------------------
    # NoisyTwins: noisytwins, nt_alpha, nt_sigma, nt_gamma, nt_lambda
    # ---------------------------------------------------------------
    if noisytwins is None:
        noisytwins = False
    if nt_alpha is None:
        nt_alpha = 0.0
    if nt_sigma is None:
        nt_sigma = 0.5
    if nt_gamma is None:
        nt_gamma = 0.05
    if nt_lambda is None:
        nt_lambda = 0.001
    if nt_start_kimg is None:
        nt_start_kimg = 0

    assert isinstance(noisytwins, bool)
    assert isinstance(nt_alpha, float)
    assert isinstance(nt_sigma, float)
    assert isinstance(nt_gamma, float)
    assert isinstance(nt_lambda, float)
    assert isinstance(nt_start_kimg, int)
    assert cond or not noisytwins
    assert (nt_gamma > 0 and nt_lambda > 0) or (nt_gamma == 0 and nt_lambda == 0)
    assert nt_start_kimg >= 0

    if noisytwins:
        args.use_noisytwins = True
        args.nt_alpha = nt_alpha
        args.nt_sigma = nt_sigma
        args.nt_gamma = nt_gamma
        args.nt_lambda = nt_lambda
        args.nt_start_kimg = nt_start_kimg
        desc += f'-nt:alp{nt_alpha}-sig{nt_sigma}-gam{nt_gamma}-lam{nt_lambda}-kimg{nt_start_kimg}'

    # -------------------------------
    # UTLO: utlo, res_uc, utlo_lambda
    # -------------------------------
    if utlo is None:
        utlo = False
    if res_uc is None:
        res_uc = 8
    if utlo_lambda is None:
        utlo_lambda = 0.0
    assert isinstance(res_uc, int)
    assert isinstance(utlo_lambda, float)
    assert cond or not utlo

    if utlo:
        args.use_utlo = True
        args.D_kwargs.res_uc = res_uc
        args.G_kwargs.synthesis_kwargs.res_uc = res_uc
        args.loss_kwargs.add_uc = True
        args.loss_kwargs.utlo_lambda = utlo_lambda

        desc += f'-utlo:res_uc{res_uc}-lam{utlo_lambda}'

    # ----------------------------------
    # Transfer learning: resume, freezed
    # ----------------------------------

    resume_specs = {
        'ffhq256':     'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/transfer-learning-source-nets/ffhq-res256-mirror-paper256-noaug.pkl',
        'ffhq512':     'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/transfer-learning-source-nets/ffhq-res512-mirror-stylegan2-noaug.pkl',
        'ffhq1024':    'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/transfer-learning-source-nets/ffhq-res1024-mirror-stylegan2-noaug.pkl',
        'celebahq256': 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/transfer-learning-source-nets/celebahq-res256-mirror-paper256-kimg100000-ada-target0.5.pkl',
        'lsundog256':  'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/transfer-learning-source-nets/lsundog-res256-paper256-kimg100000-noaug.pkl',
    }

    assert resume is None or isinstance(resume, str)
    if resume is None:
        resume = 'noresume'
    elif resume == 'noresume':
        desc += '-noresume'
    elif resume in resume_specs:
        desc += f'-resume{resume}'
        args.resume_pkl = resume_specs[resume] # predefined url
    else:
        desc += '-resumecustom'
        args.resume_pkl = resume # custom path or url

    if resume != 'noresume':
        args.ada_kimg = 100 # make ADA react faster at the beginning
        args.ema_rampup = None # disable EMA rampup

    if freezed is not None:
        assert isinstance(freezed, int)
        if not freezed >= 0:
            raise UserError('--freezed must be non-negative')
        desc += f'-freezed{freezed:d}'
        args.D_kwargs.block_kwargs.freeze_layers = freezed

    # -------------------------------------------------
    # Performance options: fp32, nhwc, nobench, workers
    # -------------------------------------------------

    if fp32 is None:
        fp32 = False
    assert isinstance(fp32, bool)
    if fp32:
        args.G_kwargs.synthesis_kwargs.num_fp16_res = args.D_kwargs.num_fp16_res = 0
        args.G_kwargs.synthesis_kwargs.conv_clamp = args.D_kwargs.conv_clamp = None

    if nhwc is None:
        nhwc = False
    assert isinstance(nhwc, bool)
    if nhwc:
        args.G_kwargs.synthesis_kwargs.fp16_channels_last = args.D_kwargs.block_kwargs.fp16_channels_last = True

    if nobench is None:
        nobench = False
    assert isinstance(nobench, bool)
    if nobench:
        args.cudnn_benchmark = False

    if allow_tf32 is None:
        allow_tf32 = False
    assert isinstance(allow_tf32, bool)
    if allow_tf32:
        args.allow_tf32 = True

    if workers is not None:
        assert isinstance(workers, int)
        if not workers >= 1:
            raise UserError('--workers must be at least 1')
        args.data_loader_kwargs.num_workers = workers

    # Add custom description at the end
    if description is not None:
        desc += f'-{description}'

    if save is None:
        save = True
    assert isinstance(save, bool)
    args.save = save

    return desc, args

#----------------------------------------------------------------------------

def subprocess_fn(rank, args, temp_dir):
    dnnlib.util.Logger(file_name=os.path.join(args.run_dir, 'log.txt'), file_mode='a', should_flush=True)

    # Init torch.distributed.
    if args.num_gpus > 1:
        init_file = os.path.abspath(os.path.join(temp_dir, '.torch_distributed_init'))
        if os.name == 'nt':
            init_method = 'file:///' + init_file.replace('\\', '/')
            torch.distributed.init_process_group(backend='gloo', init_method=init_method, rank=rank, world_size=args.num_gpus)
        else:
            init_method = f'file://{init_file}'
            torch.distributed.init_process_group(backend='nccl', init_method=init_method, rank=rank, world_size=args.num_gpus)

    # Init torch_utils.
    sync_device = torch.device('cuda', rank) if args.num_gpus > 1 else None
    training_stats.init_multiprocessing(rank=rank, sync_device=sync_device)
    if rank != 0:
        custom_ops.verbosity = 'none'

    # Execute training loop.
    training_loop.training_loop(rank=rank, **args)

#----------------------------------------------------------------------------

class CommaSeparatedList(click.ParamType):
    name = 'list'

    def convert(self, value, param, ctx):
        _ = param, ctx
        if value is None or value.lower() == 'none' or value == '':
            return []
        return value.split(',')

#----------------------------------------------------------------------------

@click.command()
@click.pass_context

# General options.
@click.option('--outdir', help='Where to save the results', required=True, metavar='DIR')
@click.option('--gpus', help='Number of GPUs to use [default: 1]', type=int, metavar='INT')
@click.option('--snap', help='Snapshot interval [default: 50 ticks]', type=int, metavar='INT')
@click.option('--metrics', help='Comma-separated list or "none" [default: fid50k_full]', type=CommaSeparatedList())
@click.option('--seed', help='Random seed [default: 0]', type=int, metavar='INT')
@click.option('--wandb_projname', help='Name of the project for wandb logging (remove "required=True" from train.py if you don\'t want wandb logging)', required=True)
@click.option('--wandb_groupname', help='Name of the wandb group in which to include this run (requires --wandb_projname to be specified)')
@click.option('--description', help='Custom suffix to add to the run description')
@click.option('-n', '--dry-run', help='Print training options and exit', is_flag=True)
@click.option('--save', help='Whether to save model checkpoints', type=bool, metavar='BOOL')

# Dataset.
@click.option('--data', help='Training data (directory or zip)', metavar='PATH', required=True)
@click.option('--data_fname', help='Name of the dataset metadata file [default: \'dataset.json\']', metavar='STR')
@click.option('--cond', help='Train conditional model based on dataset labels [default: false]', type=bool, metavar='BOOL')
@click.option('--subset', help='Train with only N images [default: all]', type=int, metavar='INT')
@click.option('--mirror', help='Enable dataset x-flips [default: false]', type=bool, metavar='BOOL')

# Base config.
@click.option('--cfg', help='Base config [default: auto]', type=click.Choice(['auto', 'stylegan2', 'paper256', 'paper512', 'paper1024', 'cifar']))
@click.option('--d_arch', help='Discriminator architecture [default: resnet]', type=click.Choice(['orig', 'skip', 'resnet']))
@click.option('--style_mixing_p', help='Override style mixing probability', type=float)
@click.option('--pl_weight', help='PLR strength [default: 2.0]', type=float)
@click.option('--pl_start_kimg', help='Only include PLR loss after `INT` kimg steps [default: 0]', type=int, metavar='INT')
@click.option('--fmaps', help='Factor that determines the amount of feature maps, multiplied by 32768 [default: 0.5]', type=float)
@click.option('--gamma', help='Override R1 gamma', type=float)
@click.option('--ema', help='Override EMA kimg', type=int)
@click.option('--ramp', help='Override EMA rampup (0 is NO rampup)', type=float)
@click.option('--nmap', help='Override number of mapping network layers', type=int)
@click.option('--kimg', help='Override training duration', type=int, metavar='INT')
@click.option('--batch', help='Override batch size', type=int, metavar='INT')

# Discriminator augmentation.
@click.option('--aug', help='Augmentation mode [default: ada]', type=click.Choice(['noaug', 'ada', 'fixed']))
@click.option('--p', help='Augmentation probability for --aug=fixed', type=float)
@click.option('--target', help='ADA target value for --aug=ada', type=float)
@click.option('--augpipe', help='Augmentation pipeline [default: bgc]', type=click.Choice(['blit', 'geom', 'color', 'filter', 'noise', 'cutout', 'bg', 'bgc', 'bgcf', 'bgcfn', 'bgcfnc']))

# LeCam regularization (Tseng et al., 2021)
@click.option('--lecam', help='Whether to use LeCam regularization [default: false]', type=bool, metavar='BOOL')
@click.option('--lecam_lambda', help='LeCam regularizer strength [default: 0.01]', type=float)

# Transitional CGAN (Shahbazi et al., 2022)
@click.option('--transition', help='Whether to use a transition from unconditional to conditional training [default: false]', type=bool, metavar='BOOL')
@click.option('--t_start_kimg', help='Start kimg for progressive conditioning (requires Transitional-CGAN to be enabled) [default: 2000]', type=int, metavar='INT')
@click.option('--t_end_kimg', help='End kimg for progressive conditioning (requires Transitional-CGAN to be enabled) [default: 4000]', type=int, metavar='INT')

# Group Spectral Regularization (Rangwani et al., 2022)
@click.option('--gsr', help='Whether to use group spectral regularization [default: false]', type=bool, metavar='BOOL')
@click.option('--gsr_lambda', help='GSR strength (requires gsr to be enabled) [default: 0.5]', type=float)

# NoisyTwins (Rangwani et al., 2023)
@click.option('--noisytwins', help='Whether to use NoisyTwins [default: false]', type=bool, metavar='BOOL')
@click.option('--nt_alpha', help='"Effective samples"; used in determining how much noise to add to each class (requires NoisyTwins to be enabled) [default: 0.0]', type=float)
@click.option('--nt_sigma', help='Noise scaling factor (requires NoisyTwins to be enabled) [default: 0.5]', type=float)
@click.option('--nt_gamma', help='Relative importance of each of the two terms in the NoisyTwins loss (requires NoisyTwins to be enabled) [default: 0.05]', type=float)
@click.option('--nt_lambda', help='Strength of NoisyTwins regularization (requires NoisyTwins to be enabled) [default: 0.001]', type=float)
@click.option('--nt_start_kimg', help='Only include NoisyTwins loss after `INT` kimg steps [default: 0]', type=int, metavar='INT')

# UTLO (Khorram et al., 2024)
@click.option('--utlo', help='Whether to use UTLO [default: false]', type=bool, metavar='BOOL')
@click.option('--res_uc', help='Up to which resolution to do unconditional training (requires UTLO to be enabled) [default: 8]', type=int, metavar='INT')
@click.option('--utlo_lambda', help='Unconditional training objective weight (requires UTLO to be enabled) [default: 0.0]', type=float)

# Transfer learning.
@click.option('--resume', help='Resume training [default: noresume]', metavar='PKL')
@click.option('--freezed', help='Freeze-D [default: 0 layers]', type=int, metavar='INT')

# Performance options.
@click.option('--fp32', help='Disable mixed-precision training', type=bool, metavar='BOOL')
@click.option('--nhwc', help='Use NHWC memory format with FP16', type=bool, metavar='BOOL')
@click.option('--nobench', help='Disable cuDNN benchmarking', type=bool, metavar='BOOL')
@click.option('--allow-tf32', help='Allow PyTorch to use TF32 internally', type=bool, metavar='BOOL')
@click.option('--workers', help='Override number of DataLoader workers', type=int, metavar='INT')

def main(ctx, outdir, dry_run, **config_kwargs):
    """Train a GAN using the techniques described in the paper
    "Training Generative Adversarial Networks with Limited Data".

    Examples:

    \b
    # Train with custom dataset using 1 GPU.
    python train.py --outdir=~/training-runs --data=~/mydataset.zip --gpus=1

    \b
    # Train class-conditional CIFAR-10 using 2 GPUs.
    python train.py --outdir=~/training-runs --data=~/datasets/cifar10.zip \\
        --gpus=2 --cfg=cifar --cond=1

    \b
    # Transfer learn MetFaces from FFHQ using 4 GPUs.
    python train.py --outdir=~/training-runs --data=~/datasets/metfaces.zip \\
        --gpus=4 --cfg=paper1024 --mirror=1 --resume=ffhq1024 --snap=10

    \b
    # Reproduce original StyleGAN2 config F.
    python train.py --outdir=~/training-runs --data=~/datasets/ffhq.zip \\
        --gpus=8 --cfg=stylegan2 --mirror=1 --aug=noaug

    \b
    Base configs (--cfg):
      auto       Automatically select reasonable defaults based on resolution
                 and GPU count. Good starting point for new datasets.
      stylegan2  Reproduce results for StyleGAN2 config F at 1024x1024.
      paper256   Reproduce results for FFHQ and LSUN Cat at 256x256.
      paper512   Reproduce results for BreCaHAD and AFHQ at 512x512.
      paper1024  Reproduce results for MetFaces at 1024x1024.
      cifar      Reproduce results for CIFAR-10 at 32x32.

    \b
    Transfer learning source networks (--resume):
      ffhq256        FFHQ trained at 256x256 resolution.
      ffhq512        FFHQ trained at 512x512 resolution.
      ffhq1024       FFHQ trained at 1024x1024 resolution.
      celebahq256    CelebA-HQ trained at 256x256 resolution.
      lsundog256     LSUN Dog trained at 256x256 resolution.
      <PATH or URL>  Custom network pickle.
    """
    dnnlib.util.Logger(should_flush=True)

    # Setup training options.
    try:
        run_desc, args = setup_training_loop_kwargs(**config_kwargs)
    except UserError as err:
        ctx.fail(err)

    # Pick output directory.
    prev_run_dirs = []
    if os.path.isdir(outdir):
        prev_run_dirs = [x for x in os.listdir(outdir) if os.path.isdir(os.path.join(outdir, x))]
    prev_run_ids = [re.match(r'^\d+', x) for x in prev_run_dirs]
    prev_run_ids = [int(x.group()) for x in prev_run_ids if x is not None]
    cur_run_id = max(prev_run_ids, default=-1) + 1
    args.run_dir = os.path.join(outdir, f'{cur_run_id:05d}-{run_desc}')
    assert not os.path.exists(args.run_dir)

    # Print options.
    print()
    print('Training options:')
    print(json.dumps(args, indent=2))
    print()
    print(f'Output directory:   {args.run_dir}')
    print(f'Training data:      {args.training_set_kwargs.path}')
    print(f'Training duration:  {args.total_kimg} kimg')
    print(f'Number of GPUs:     {args.num_gpus}')
    print(f'Number of images:   {args.training_set_kwargs.max_size}')
    print(f'Image resolution:   {args.training_set_kwargs.resolution}')
    print(f'Conditional model:  {args.training_set_kwargs.use_labels}')
    print(f'Dataset x-flips:    {args.training_set_kwargs.xflip}')
    print()

    # Dry run?
    if dry_run:
        print('Dry run; exiting.')
        return

    # Create output directory.
    print('Creating output directory...')
    os.makedirs(args.run_dir)
    with open(os.path.join(args.run_dir, 'training_options.json'), 'wt') as f:
        json.dump(args, f, indent=2)

    # Launch processes.
    print('Launching processes...')
    torch.multiprocessing.set_start_method('spawn')
    with tempfile.TemporaryDirectory() as temp_dir:
        if args.num_gpus == 1:
            subprocess_fn(rank=0, args=args, temp_dir=temp_dir)
        else:
            torch.multiprocessing.spawn(fn=subprocess_fn, args=(args, temp_dir), nprocs=args.num_gpus)

#----------------------------------------------------------------------------

if __name__ == "__main__":
    main() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------
