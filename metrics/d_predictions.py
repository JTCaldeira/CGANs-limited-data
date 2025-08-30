import torch
import numpy as np
import copy

import dnnlib


def compute_d_predictions(opts, num_gen_per_class, max_real_per_class, batch_size=None):
    """Compute the discriminator logits for a subset of the classes.
    This is done for both real and fake samples."""
    dataset = dnnlib.util.construct_class_by_name(**opts.dataset_kwargs)
    assert dataset.has_labels
    idxs_per_label = dataset.get_idxs_per_label()
    labels_to_track = dataset.get_interesting_classes()
    max_label_len = len(str(max(labels_to_track)))

    if batch_size is None:
        batch_size = min(4096 // dataset.resolution, 64)

    G = copy.deepcopy(opts.G).eval().requires_grad_(False).to(opts.device)
    D = copy.deepcopy(opts.D).eval().requires_grad_(False).to(opts.device)

    results = {}

    for label in labels_to_track:
        score_real = 0.0
        score_fake = 0.0
        if idxs_per_label[label].shape[0] < max_real_per_class:
            real_img_idxs = idxs_per_label[label]
        else:
            real_img_idxs = np.random.RandomState(123).choice(idxs_per_label[label], max_real_per_class, replace=False)

        img_counter = 0
        n_real = min(max_real_per_class, idxs_per_label[label].shape[0])
        while img_counter < n_real:
            with torch.no_grad():
                real_img = torch.stack([torch.Tensor(dataset[i][0]) for i in real_img_idxs[img_counter:img_counter + batch_size]]).to(opts.device).to(torch.float32) / 127.5 - 1
                c = torch.nn.functional.one_hot(torch.full((batch_size,), label), num_classes=dataset.label_dim).pin_memory().to(opts.device)
                n_real_img = real_img.shape[0]
                # Fill batch completely to not have issues with MinibatchStd grouping.
                if n_real_img < batch_size:
                    n_extend = batch_size - n_real_img
                    padding = real_img[-1].unsqueeze(0).repeat(n_extend, 1, 1, 1)
                    real_img = torch.cat((real_img, padding))

                logits = D(real_img, c)
                # Ignore the padding possibly added above.
                logits = logits[:n_real_img]
                if logits.size(1) == 2:
                    logits = logits[:, 0]

                score_real += logits.cpu().sum()
                img_counter += batch_size

        img_counter = 0
        torch_rand = torch.Generator(device=opts.device)
        torch_rand.manual_seed(123)
        while img_counter < num_gen_per_class:
            with torch.no_grad():
                z = torch.randn([batch_size, G.z_dim], generator=torch_rand, device=opts.device)
                c = torch.nn.functional.one_hot(torch.full((batch_size,), label), num_classes=dataset.label_dim).pin_memory().to(opts.device)

                gen_img = G(z, c)
                logits = D(gen_img, c)
                if logits.size(1) == 2:
                    logits = logits[:, 0]

                score_fake += logits.cpu().sum()
                img_counter += batch_size

        results[f"zreal_{str(label).zfill(max_label_len)}"] = score_real.item() / n_real
        results[f"zfake_{str(label).zfill(max_label_len)}"] = score_fake.item() / img_counter

    return results
