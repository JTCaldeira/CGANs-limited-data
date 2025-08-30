"""
Implements the GSR loss proposed in Rangwani et al. (2022):
"Improving GANs for Long-Tailed Data through Group Spectral Regularization"
https://arxiv.org/abs/2208.09932

Code partly taken from the associated GitHub:
https://github.com/val-iisc/gSRGAN
"""

import torch
import torch.nn.functional as F


_wdim_to_groupshape = {
    256: (16, 16),
    512: (16, 32)
}
_num_iter = 1

def compute_gsr_loss(ws, c, effective_num_samples):
    if ws.dim() == 3:
        ws = ws[:, 0, :].clone()
    assert ws.dim() == 2

    batch_size = ws.shape[0]
    group_shape = _wdim_to_groupshape[ws.shape[1]]

    # Reshape from (batch_size, wdim) to (batchsize, g, c).
    assert ws.dim() == 2
    ws_grouped = ws.reshape(batch_size, *group_shape)

    with torch.no_grad():
        u = torch.randn((batch_size, group_shape[0])).to(ws.device).unsqueeze(2)
        v = torch.randn((batch_size, group_shape[1])).to(ws.device).unsqueeze(2)

        for _ in range(_num_iter):
            v = F.normalize(torch.bmm(ws_grouped.permute(0, 2, 1), u), dim=1, eps=1e-3, out=v)
            u = F.normalize(torch.bmm(ws_grouped, v), dim=1, eps=1e-3, out=u)

        u = u.clone()
        v = v.clone()

    sigma = (u * torch.bmm(ws_grouped, v)).squeeze(2)
    sigma = sigma.sum(-1).unsqueeze(-1)

    for i, l in enumerate(c.argmax(dim=1)):
        sigma[i] = sigma[i] * effective_num_samples[l]

    return sigma