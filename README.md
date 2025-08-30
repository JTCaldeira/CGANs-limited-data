## Training Conditional GANs on Limited and Long-Tailed Data: a Survey and Comparative Analysis

João Tomás Caldeira, Fábio Vital, Francisco Melo<br>
Link: TODO<br>

Abstract: TODO

## Overview

1. [Requirements](#Requirements)
2. [Usage](#Usage)
    1. [Dataset Preparation](#Dataset-Preparation)
    2. [Training](#Training)
    3. [Evaluation](#Evaluation)
3. [Contact](#Contact)
4. [License](#License)
5. [How to Cite](#How-to-Cite)

## Requirements<a name="Requirements"></a>

* Linux and Windows are supported, but we recommend Linux for performance and compatibility reasons.
* For a batch size of 64, we used an NVIDIA GeForce RTX 3090 GPU with 24GB of memory. Memory usage at the beginning of each run is about double of what is required throughout training due to `torch.backends.cudnn.benchmark`.
* 64-bit Python 3.7 and PyTorch 1.7.1. See [https://pytorch.org/](https://pytorch.org/) for PyTorch install instructions.
* CUDA toolkit 11.0 or later. Use at least version 11.1 if running on RTX 3090. (Why is a separate CUDA toolkit installation required?  See comments in [#2](https://github.com/NVlabs/stylegan2-ada-pytorch/issues/2#issuecomment-779457121).)
* Python libraries: `pip install click requests tqdm pyspng ninja imageio-ffmpeg==0.4.3 psutil scipy wandb scikit-learn`. We use the Anaconda3 2020.11 distribution which installs most of these by default.
* This project uses Weights and Biases (W&B) for optional visualization and logging. In addition to installing W&B, you need to create a free account on the [W&B website](https://wandb.ai/site/). Then, you must log in to your W&B account by running `wandb login` in the command line.
* Docker users: use the [provided Dockerfile](./Dockerfile) by StyleGAN2+ADA to build an image with the required library dependencies.

The code relies heavily on custom PyTorch extensions that are compiled on the fly using NVCC. On Windows, the compilation requires Microsoft Visual Studio. We recommend installing [Visual Studio Community Edition](https://visualstudio.microsoft.com/vs/) and adding it into `PATH` using `"C:\Program Files (x86)\Microsoft Visual Studio\<VERSION>\Community\VC\Auxiliary\Build\vcvars64.bat"`.

## Usage<a name="Usage"></a>

This codebase is based on the PyTorch implementation of [StyleGAN2+ADA](https://github.com/NVlabs/stylegan2-ada-pytorch), so we strongly advise first reading its instructions. We also implement and adopt some code the following methods:
- Transitional-CGAN ([paper](https://arxiv.org/abs/2201.06578), [implementation](https://github.com/mshahbazi72/transitional-cGAN))
- LeCam Regularizer ([paper](https://arxiv.org/abs/2104.03310))
- Group Spectral Regularizer ([paper](https://arxiv.org/abs/2208.09932), [implementation](https://github.com/val-iisc/gSRGAN))
- NoisyTwins ([paper](https://arxiv.org/abs/2304.05866), [implementation](https://github.com/val-iisc/NoisyTwins/tree/main))
- UTLO ([paper](https://arxiv.org/abs/2402.17065), [implementation](https://github.com/khorrams/utlo/tree/main))

##### Dataset Preparation<a name="Dataset-Preparation"></a>

All datasets used in the paper can be found at TODO. However, to demonstrate how custom datasets may be created from folders containing images, we reproduce the steps used to create the `AnimalFace` dataset.

Datasets are stored as uncompressed ZIP archives containing uncompressed PNG files and a metadata file `dataset.json` for labels. 

The following command downloads and extracts the `AnimalFace` dataset.
```.bash
wget https://vcla.stat.ucla.edu/people/zhangzhang-si/HiT/AnimalFace.zip
unzip AnimalFace.zip -d ./
rm -rf Image/Natural
rm AnimalFace.zip
```
Custom datasets can be created from a folder containing images (each sub-directory containing images of one class in case of multi-class datasets) using `dataset_tool.py`; Here is an example of how to convert the dataset folder to the desired ZIP file:

```.bash
mkdir datasets
python dataset_tool.py --source=Image/ --dest=datasets/animalface.zip --transform=center-crop --width=64 --height=64
rm -rf Image/
```
The above example reads the images from the image folder provided by `--src`, resizes the images to the sizes provided by `--width` and `--height`, and applys the transform `center-crop` to them. The resulting images along with the metadata (label information) are stored as a ZIP file determined by `--dest`.

Please see [`python dataset_tool.py --help`](./docs/dataset-tool-help.txt) for more information. See [StyleGAN2+ADA instructions](https://github.com/NVlabs/stylegan2-ada-pytorch/blob/main/README.md#preparing-datasets) for more details on specific datasets or Legacy TFRecords datasets .

The created ZIP file can be passed to the training and evaluation code using `--data` argument.

##### Training<a name="Training"></a>

After preparing the dataset, you can run the following command to start a training run.

```.bash
python train.py --outdir=training_runs_af_sg2ada --data=~/datasets/animalface.zip  \
--data_fname=lt_100.json --snap=50 --mirror=True --seed=42 --gpus=1 --batch=64 \
--wandb_proj=af_sg2ada --metrics=fid50k_full,fidclip50k_full,prdc50k_full,cmmd30k_30k \
--description="example-run-af"
```

See [StyleGAN2+ADA instructions](https://github.com/NVlabs/stylegan2-ada-pytorch/blob/main/README.md#training-new-networks) for more details on the arguments, configurations amd hyper-parammeters. Please refer to [`python train.py --help`](./docs/train-help.txt) for the full list of arguments.

##### Evaluation<a name="Evaluation"></a>

By default, `train.py` automatically computes the FID for each network pickle exported during training. More metrics can be added to the argument `--metrics` as a comma-separated list. To monitor the training, you can inspect the log.txt an JSON files (e.g. `metric-fid50k_full.jsonl` for FID) saved in the ouput directory. Alternatively, you can inspect WandB or Tensorboard logs. Specifying the options `wandb_projname` and `wandb_groupname` is required to use WandB.

Metric computation can be disabled with `--metrics=none` to speed up the training (3%&ndash;9%). Additional metrics can also be computed after training from a network pickle:

```.bash
# Previous training run: look up options automatically, save result to JSONL file.
python calc_metrics.py --metrics=pr50k3_full \
    --network=~/training-runs/00000-ffhq10k-res64-auto1/network-snapshot-000000.pkl

# Pre-trained network pickle: specify dataset explicitly, print result to stdout.
python calc_metrics.py --metrics=fid50k_full --data=~/datasets/ffhq.zip --mirror=1 \
    --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ffhq.pkl
```

The first example looks up the training configuration and performs the same operation as if `--metrics=pr50k3_full` had been specified during training. The second example downloads a pre-trained network pickle, in which case the values of `--mirror` and `--data` must be specified explicitly.

Note that in order to use CLIP-based metrics, you need to download the pre-trained [CLIP ViT-B/32](https://arxiv.org/abs/2103.00020) by OpenAI (shared through a GitHub [repository](https://github.com/openai/CLIP) and place it in  `models/clip-vit_b32.pkl`.

See [StyleGAN2+ADA instructions](https://github.com/NVlabs/stylegan2-ada-pytorch/blob/main/README.md#quality-metrics) for more details on the available metrics. 

## Contact<a name="Contact"></a>

For any questions, suggestions, or issues with the code, please contact João Tomás Caldeira at [joao.tomas.brazao.caldeira@tecnico.ulisboa.pt](mailto:joao.tomas.brazao.caldeira@tecnico.ulisboa.pt).

## License<a name="License"></a>
This repository uses code from the methods listed in [Usage](#Usage). Please refer to and acknowledge their license and terms of use.

## How to Cite<a name="How-to-Cite"></a>

TODO
