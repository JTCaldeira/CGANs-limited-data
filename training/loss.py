# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import numpy as np
import torch
from torch_utils import training_stats
from torch_utils import misc
from torch_utils.ops import conv2d_gradfix
from training.barlow import compute_contrastive
from training.gsr import compute_gsr_loss

#----------------------------------------------------------------------------

class Loss:
    def accumulate_gradients(self, phase, real_img, real_c, gen_z, gen_c, sync, gain): # to be overridden by subclass
        raise NotImplementedError()

#----------------------------------------------------------------------------

class StyleGAN2Loss(Loss):
    def __init__(self, device, G_mapping, G_synthesis, D, augment_pipe=None, style_mixing_prob=0.9, r1_gamma=10,
        pl_batch_shrink=2, pl_decay=0.01, pl_weight=2,  # PLR
        use_lecam=False, lecam=None, lecam_lambda=0.0,   # LeCam
        use_gsr=False, gsr_lambda=0.5, effective_num_samples=None,   # GSR
        nt=None, nt_lambda=0.0,         # NoisyTwins
        add_uc=False, utlo_lambda=0.0,  # UTLO
    ):
        super().__init__()
        self.device = device
        self.G_mapping = G_mapping
        self.G_synthesis = G_synthesis
        self.D = D
        self.augment_pipe = augment_pipe
        self.style_mixing_prob = style_mixing_prob
        self.r1_gamma = r1_gamma
        self.pl_batch_shrink = pl_batch_shrink
        self.pl_decay = pl_decay
        self.pl_weight = pl_weight
        self.pl_mean = torch.zeros([], device=device)
        self.use_lecam = use_lecam    # Whether to use LeCam regularization
        self.lecam = lecam          # Tracks EMAs and computes loss
        self.lecam_lambda = lecam_lambda    # LeCam regularization strength
        self.use_gsr = use_gsr      # Whether to use GSR
        self.gsr_lambda = gsr_lambda# GSR strength
        self.effective_num_samples = effective_num_samples
        self.nt = nt    # NoisyTwins functionality; use NoisyTwins if not None
        self.nt_lambda = nt_lambda  # NoisyTwins regularization strength
        self.add_uc = add_uc    # Whether to use UTLO
        self.utlo_lambda = utlo_lambda  # UTLO unconditional loss strength

    def run_G(self, z, c, sync, add_uc=False):
        with misc.ddp_sync(self.G_mapping, sync):
            ws = self.G_mapping(z, c)
            if self.style_mixing_prob > 0:
                with torch.autograd.profiler.record_function('style_mixing'):
                    cutoff = torch.empty([], dtype=torch.int64, device=ws.device).random_(1, ws.shape[1])
                    cutoff = torch.where(torch.rand([], device=ws.device) < self.style_mixing_prob, cutoff, torch.full_like(cutoff, ws.shape[1]))
                    ws[:, cutoff:] = self.G_mapping(torch.randn_like(z), c, skip_w_avg_update=True)[:, cutoff:]
        with misc.ddp_sync(self.G_synthesis, sync):
            img = self.G_synthesis(ws, add_uc=add_uc)
        return img, ws

    def run_D(self, img, c, sync, uc=False):
        if self.augment_pipe is not None:
            img = self.augment_pipe(img)
        with misc.ddp_sync(self.D, sync):
            logits = self.D(img, c, uc=uc)
        return logits

    def accumulate_gradients(self, phase, real_img, real_c, gen_z, gen_c, sync, gain, add_nt=False, add_gpl=True):
        assert phase in ['Gmain', 'Greg', 'Gboth', 'Dmain', 'Dreg', 'Dboth']
        do_Gmain = (phase in ['Gmain', 'Gboth'])
        do_Dmain = (phase in ['Dmain', 'Dboth'])
        do_Gpl   = (phase in ['Greg', 'Gboth']) and (self.pl_weight != 0) and add_gpl
        do_Dr1   = (phase in ['Dreg', 'Dboth']) and (self.r1_gamma != 0)

        # Gmain: Maximize logits for generated images.
        if do_Gmain:
            with torch.autograd.profiler.record_function('Gmain_forward'):
                gen_img, _gen_ws = self.run_G(gen_z, gen_c, sync=(sync and not do_Gpl), add_uc=self.add_uc) # May get synced by Gpl.

                # (UTLO)
                gen_img_uc = None
                if isinstance(gen_img, list):
                    gen_img, gen_img_uc = gen_img

                gen_logits = self.run_D(gen_img, gen_c, sync=False)

                gen_logits_c = None
                if gen_logits.size(1) == 2:
                    gen_logits_c = gen_logits[:, 1]
                    gen_logits = gen_logits[:, 0]

                training_stats.report('Loss/scores/fake', gen_logits)
                training_stats.report('Loss/signs/fake', gen_logits.sign())
                loss_Gmain = torch.nn.functional.softplus(-gen_logits) # -log(sigmoid(gen_logits))

                if gen_logits_c is not None:
                    training_stats.report('Loss/scores/fake_c', gen_logits_c)
                    training_stats.report('Loss/signs/fake_c', gen_logits_c.sign())
                    loss_Gmain_c = torch.nn.functional.softplus(-gen_logits_c)
                    loss_Gmain = loss_Gmain + self.G_mapping.transition * loss_Gmain_c

                training_stats.report('Loss/G/loss', loss_Gmain)

                # (GSR)
                if self.use_gsr:
                    gsr_loss = self.gsr_lambda * compute_gsr_loss(_gen_ws, gen_c, self.effective_num_samples)
                    training_stats.report('GSR/G/loss', gsr_loss)
                    loss_Gmain = loss_Gmain + gsr_loss

                # (UTLO)
                if gen_img_uc is not None:
                    gen_logits_uc = self.run_D(gen_img_uc, gen_c, sync=False, uc=True)
                    training_stats.report('UTLO/scores/fake_uc', gen_logits_uc)
                    training_stats.report('UTLO/signs/fake_uc', gen_logits_uc.sign())
                    loss_Gmain_uc = torch.nn.functional.softplus(-gen_logits_uc)
                    training_stats.report('UTLO/G/loss_uc', loss_Gmain_uc)
                    loss_Gmain = loss_Gmain + self.utlo_lambda * loss_Gmain_uc
                    training_stats.report('UTLO/G/loss', loss_Gmain)
                
                # (NoisyTwins)
                if self.nt is not None:
                    if add_nt:
                        reduced_batchsize = _gen_ws.shape[0] // 2
                        nt_loss = compute_contrastive(_gen_ws[:reduced_batchsize, 0, :], _gen_ws[reduced_batchsize:, 0, :], self.nt, self.nt_lambda)
                        loss_Gmain = loss_Gmain + nt_loss
                    else:
                        nt_loss = 0.0

                    training_stats.report('NoisyTwins/loss_contrastive', nt_loss)
                    training_stats.report('NoisyTwins/loss', loss_Gmain)
            with torch.autograd.profiler.record_function('Gmain_backward'):
                loss_Gmain.mean().mul(gain).backward()

        # Gpl: Apply path length regularization.
        if do_Gpl:
            with torch.autograd.profiler.record_function('Gpl_forward'):
                batch_size = gen_z.shape[0] // self.pl_batch_shrink
                gen_img, gen_ws = self.run_G(gen_z[:batch_size], gen_c[:batch_size], sync=sync)
                pl_noise = torch.randn_like(gen_img) / np.sqrt(gen_img.shape[2] * gen_img.shape[3])
                with torch.autograd.profiler.record_function('pl_grads'), conv2d_gradfix.no_weight_gradients():
                    pl_grads = torch.autograd.grad(outputs=[(gen_img * pl_noise).sum()], inputs=[gen_ws], create_graph=True, only_inputs=True)[0]
                pl_lengths = pl_grads.square().sum(2).mean(1).sqrt()
                pl_mean = self.pl_mean.lerp(pl_lengths.mean(), self.pl_decay)
                self.pl_mean.copy_(pl_mean.detach())
                pl_penalty = (pl_lengths - pl_mean).square()
                training_stats.report('Loss/pl_penalty', pl_penalty)
                loss_Gpl = pl_penalty * self.pl_weight
                training_stats.report('Loss/G/reg', loss_Gpl)
            with torch.autograd.profiler.record_function('Gpl_backward'):
                (gen_img[:, 0, 0, 0] * 0 + loss_Gpl).mean().mul(gain).backward()

        # Dmain: Minimize logits for generated images.
        loss_Dgen = 0
        if do_Dmain:
            with torch.autograd.profiler.record_function('Dgen_forward'):
                gen_img, _gen_ws = self.run_G(gen_z, gen_c, sync=False, add_uc=self.add_uc)
                
                # (UTLO)
                gen_img_uc = None
                if isinstance(gen_img, list):
                    gen_img, gen_img_uc = gen_img

                gen_logits = self.run_D(gen_img, gen_c, sync=False, uc=False) # Gets synced by loss_Dreal.

                gen_logits_c = None
                if gen_logits.size(1) == 2:
                    gen_logits_c = gen_logits[:, 1]
                    gen_logits = gen_logits[:, 0]

                training_stats.report('Loss/scores/fake', gen_logits)
                training_stats.report('Loss/signs/fake', gen_logits.sign())
                loss_Dgen = torch.nn.functional.softplus(gen_logits) # -log(1 - sigmoid(gen_logits))

                if gen_logits_c is not None:
                    training_stats.report('Loss/scores/fake_c', gen_logits_c)
                    training_stats.report('Loss/signs/fake_c', gen_logits_c.sign())
                    loss_Dgen_c = torch.nn.functional.softplus(gen_logits_c) # -log(1 - sigmoid(gen_logits))
                    loss_Dgen = loss_Dgen + self.G_mapping.transition * loss_Dgen_c

                # (LeCam)
                if self.use_lecam:
                    lecam_loss = self.lecam_lambda * self.lecam.loss_fake(gen_logits)
                    training_stats.report('LeCam/fake/loss_lecam', lecam_loss)
                    loss_Dgen = loss_Dgen + lecam_loss
                    training_stats.report('LeCam/fake/loss', loss_Dgen)

                # (UTLO)
                if gen_img_uc is not None:
                    gen_logits_uc = self.run_D(gen_img_uc, gen_c, sync=False, uc=True)
                    training_stats.report('UTLO/scores/fake_uc', gen_logits_uc)
                    training_stats.report('UTLO/signs/fake_uc', gen_logits_uc.sign())
                    loss_Dgen_uc = torch.nn.functional.softplus(gen_logits_uc)
                    loss_Dgen = loss_Dgen + self.utlo_lambda * loss_Dgen_uc

            with torch.autograd.profiler.record_function('Dgen_backward'):
                loss_Dgen.mean().mul(gain).backward()

        # Dmain: Maximize logits for real images.
        # Dr1: Apply R1 regularization.
        if do_Dmain or do_Dr1:
            name = 'Dreal_Dr1' if do_Dmain and do_Dr1 else 'Dreal' if do_Dmain else 'Dr1'
            with torch.autograd.profiler.record_function(name + '_forward'):
                real_img_tmp = real_img.detach().requires_grad_(do_Dr1)
                real_logits = self.run_D(real_img_tmp, real_c, sync=sync, uc=False)

                real_logits_c = None
                if real_logits.size(1) == 2:
                    real_logits_c = real_logits[:, 1]
                    real_logits = real_logits[:, 0]

                training_stats.report('Loss/scores/real', real_logits)
                training_stats.report('Loss/signs/real', real_logits.sign())

                loss_Dreal = 0
                if do_Dmain:
                    loss_Dreal = torch.nn.functional.softplus(-real_logits) # -log(sigmoid(real_logits))

                    if real_logits_c is not None:
                        training_stats.report('Loss/scores/real_c', real_logits_c)
                        training_stats.report('Loss/signs/real_c', real_logits_c.sign())
                        loss_Dreal_c = torch.nn.functional.softplus(-real_logits_c) # -log(sigmoid(real_logits))
                        loss_Dreal = loss_Dreal + self.G_mapping.transition * loss_Dreal_c

                    training_stats.report('Loss/D/loss', loss_Dgen + loss_Dreal)

                    # (LeCam)
                    if self.use_lecam:
                        lecam_loss = self.lecam_lambda * self.lecam.loss_real(real_logits)
                        training_stats.report('LeCam/real/loss_lecam', lecam_loss)
                        loss_Dreal = loss_Dreal + lecam_loss
                        training_stats.report('LeCam/real/loss', loss_Dreal)

                    # (UTLO)
                    if self.add_uc:
                        real_img_tmp_uc = self.D.downsample_uc(real_img_tmp)
                        real_logits_uc = self.run_D(real_img_tmp_uc, real_c, sync=sync, uc=True)
                        training_stats.report('UTLO/scores/real_uc', real_logits_uc)
                        training_stats.report('UTLO/signs/real_uc', real_logits_uc.sign())
                        loss_Dreal_uc = torch.nn.functional.softplus(-real_logits_uc)
                        loss_Dreal = loss_Dreal + self.utlo_lambda * loss_Dreal_uc
                        training_stats.report('UTLO/D/loss', loss_Dgen + loss_Dreal)

                loss_Dr1 = 0
                if do_Dr1:
                    with torch.autograd.profiler.record_function('r1_grads'), conv2d_gradfix.no_weight_gradients():
                        r1_grads = torch.autograd.grad(outputs=[real_logits.sum()], inputs=[real_img_tmp], create_graph=True, only_inputs=True)[0]
                    r1_penalty = r1_grads.square().sum([1,2,3])
                    loss_Dr1 = r1_penalty * (self.r1_gamma / 2)
                    training_stats.report('Loss/r1_penalty', r1_penalty)
                    training_stats.report('Loss/D/reg', loss_Dr1)

            with torch.autograd.profiler.record_function(name + '_backward'):
                (real_logits * 0 + loss_Dreal + loss_Dr1).mean().mul(gain).backward()

#----------------------------------------------------------------------------
