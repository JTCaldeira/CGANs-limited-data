# Carnivores

## SG2+ADA

```bash
# Replace <SEED> by 111, 222, and 333. (3 runs)
./train.py --outdir=out --wandb_projname=carnivores --wandb_groupname=sg2+ada --data=./datasets/carnivores.zip --snap=50 --kimg=10000 --seed=<SEED> --description=seed<SEED> --cond=1 --cfg=auto --mirror=1 --metrics=fid50k_full,pr50k3_full,cvar,d_preds --style_mixing_p=0 --gamma=0.01 --ema=500 --ramp=0.05 --nmap=2 --batch=64
```

## Transitional

```bash
# Replace <SEED> by 111, 222, and 333. (3 runs)
./train.py --outdir=out --transition=1 --wandb_projname=flowers --wandb_groupname=transitional --data=./datasets/carnivores.zip --snap=50 --kimg=16000 --seed=<SEED> --description=seed<SEED> --cond=1 --cfg=auto --mirror=1 --metrics=fid50k_full,pr50k3_full,cvar,d_preds --style_mixing_p=0 --gamma=0.01 --ema=500 --ramp=0.05 --nmap=2 --batch=64
```

## LeCam

```bash
# Replace <SEED> by 111, 222, and 333. (3 runs)
./train.py --outdir=out --wandb_projname=carnivores --wandb_groupname=lecam --lecam=1 --lecam_lambda=3e-7 --data=./datasets/carnivores.zip --snap=50 --kimg=25000 --seed=<SEED> --description=seed<SEED> --cond=1 --cfg=auto --mirror=1 --metrics=fid50k_full,pr50k3_full,cvar,d_preds --style_mixing_p=0 --gamma=0.01 --ema=500 --ramp=0.05 --nmap=2 --batch=64
```

## GSR

```bash
# Replace <SEED> by 111, 222, and 333. (3 runs)
./train.py --outdir=out --wandb_projname=carnivores --wandb_groupname=gsr --data=./datasets/carnivores.zip --gsr=1 --gsr_lambda=0.5 --snap=50 --kimg=25000 --seed=<SEED> --description=seed<SEED> --cond=1 --cfg=auto --mirror=1 --metrics=fid50k_full,pr50k3_full,cvar,d_preds --style_mixing_p=0 --gamma=0.01 --ema=500 --ramp=0.05 --nmap=2 --batch=64
```

## NoisyTwins

```bash
# Replace <SEED> by 111, 222, and 333. (3 runs)
./train.py --outdir=out --wandb_projname=carnivores --wandb_groupname=noisytwins --noisytwins=1 --nt_alpha=0 --nt_sigma=0.5 --nt_gamma=0.05 --nt_lambda=0.001 --nt_start_kimg=0 --data=./datasets/carnivores.zip --snap=50 --kimg=25000 --seed=<SEED> --description=seed<SEED> --cond=1 --cfg=auto --mirror=1 --metrics=fid50k_full,pr50k3_full,cvar,d_preds --style_mixing_p=0 --gamma=0.01 --ema=500 --ramp=0.05 --nmap=2 --batch=64
```

## UTLO

```bash
# Replace <SEED> by 111, 222, and 333. (3 runs)
./train.py --outdir=out --wandb_projname=carnivores --wandb_groupname=utlo --utlo=1 --res_uc=8 --utlo_lambda=1 --data=./datasets/carnivores.zip --snap=50 --kimg=25000 --seed=<SEED> --description=seed<SEED> --cond=1 --cfg=auto --mirror=1 --metrics=fid50k_full,pr50k3_full,cvar,d_preds --style_mixing_p=0 --gamma=0.01 --ema=500 --ramp=0.05 --nmap=2 --batch=64
```

# AnimalFaces-LT

## SG2+ADA

```bash
# Replace <SEED> by 111, 222, and 333. (3 runs)
./train.py --outdir=out --wandb_projname=animalface --wandb_groupname=sg2+ada --data=./datasets/animalface-lt.zip --snap=50 --kimg=100000 --seed=<SEED> --description=seed<SEED> --cond=1 --cfg=auto --mirror=1 --metrics=fid50k_full,pr50k3_full,cvar,d_preds --style_mixing_p=0.5 --gamma=0.01 --ema=500 --ramp=0.05 --nmap=2 --batch=64
```

## Transitional

```bash
# Replace <SEED> by 111, 222, and 333. (3 runs)
./train.py --outdir=out --transition=1 --wandb_projname=animalface --wandb_groupname=transitional --data=./datasets/animalface-lt.zip --snap=50 --kimg=100000 --seed=<SEED> --description=seed<SEED> --cond=1 --cfg=auto --mirror=1 --metrics=fid50k_full,pr50k3_full,cvar,d_preds --style_mixing_p=0.5 --gamma=0.01 --ema=500 --ramp=0.05 --nmap=2 --batch=64
```

## LeCam

```bash
# Replace <SEED> by 111, 222, and 333. (3 runs)
./train.py --outdir=out --wandb_projname=animalface --wandb_groupname=lecam --lecam=1 --lecam_lambda=3e-7 --data=./datasets/animalface-lt.zip --snap=50 --kimg=100000 --seed=<SEED> --description=seed<SEED> --cond=1 --cfg=auto --mirror=1 --metrics=fid50k_full,pr50k3_full,cvar,d_preds --style_mixing_p=0.5 --gamma=0.01 --ema=500 --ramp=0.05 --nmap=2 --batch=64
```

## GSR

```bash
# Replace <SEED> by 111, 222, and 333. (3 runs)
./train.py --outdir=out --wandb_projname=animalface --wandb_groupname=gsr --data=./datasets/animalface-lt.zip --gsr=1 --gsr_lambda=0.5 --snap=50 --kimg=100000 --seed=<SEED> --description=seed<SEED> --cond=1 --cfg=auto --mirror=1 --metrics=fid50k_full,pr50k3_full,cvar,d_preds --style_mixing_p=0.5 --gamma=0.01 --ema=500 --ramp=0.05 --nmap=2 --batch=64
```

## NoisyTwins

```bash
# Replace <SEED> by 111, 222, and 333. (3 runs)
./train.py --outdir=out --wandb_projname=animalface --wandb_groupname=noisytwins --noisytwins=1 --nt_alpha=0 --nt_sigma=0.5 --nt_gamma=0.05 --nt_lambda=0.001 --nt_start_kimg=0 --data=./datasets/animalface-lt.zip --snap=50 --kimg=150000 --seed=<SEED> --description=seed<SEED> --cond=1 --cfg=auto --mirror=1 --metrics=fid50k_full,pr50k3_full,cvar,d_preds --style_mixing_p=0.5 --gamma=0.01 --ema=500 --ramp=0.05 --nmap=2 --batch=64
```

## UTLO

```bash
# Replace <SEED> by 111, 222, and 333. (3 runs)
./train.py --outdir=out --wandb_projname=animalface --wandb_groupname=utlo --utlo=1 --res_uc=8 --utlo_lambda=1 --data=./datasets/animalface-lt.zip --snap=50 --kimg=100000 --seed=<SEED> --description=seed<SEED> --cond=1 --cfg=auto --mirror=1 --metrics=fid50k_full,pr50k3_full,cvar,d_preds --style_mixing_p=0.5 --gamma=0.01 --ema=500 --ramp=0.05 --nmap=2 --batch=64
```

# Flowers-LT

## SG2+ADA

```bash
# Replace <SEED> by 111, 222, and 333. (3 runs)
./train.py --outdir=out --wandb_projname=flowers --wandb_groupname=sg2+ada --data=./datasets/flowers-lt.zip --snap=50 --kimg=25000 --seed=<SEED> --description=seed<SEED> --cond=1 --cfg=auto --mirror=1 --metrics=fid50k_full,pr50k3_full,cvar,d_preds --batch=64
```

## Transitional

```bash
# Replace <SEED> by 111, 222, and 333. (3 runs)
./train.py --outdir=out --transition=1 --wandb_projname=flowers --wandb_groupname=transitional --data=./datasets/flowers-lt.zip --snap=50 --kimg=16000 --seed=<SEED> --description=seed<SEED> --cond=1 --cfg=auto --mirror=1 --metrics=fid50k_full,pr50k3_full,cvar,d_preds --batch=64
```

## LeCam

```bash
# Replace <SEED> by 111, 222, and 333. (3 runs)
./train.py --outdir=out --wandb_projname=flowers --wandb_groupname=lecam --lecam=1 --lecam_lambda=3e-7 --data=./datasets/flowers-lt.zip --snap=50 --kimg=25000 --seed=<SEED> --description=seed<SEED> --cond=1 --cfg=auto --mirror=1 --metrics=fid50k_full,pr50k3_full,cvar,d_preds --batch=64
```

## GSR

```bash
# Replace <SEED> by 111, 222, and 333. (3 runs)
./train.py --outdir=out --wandb_projname=flowers --wandb_groupname=gsr --data=./datasets/flowers-lt.zip --gsr=1 --gsr_lambda=0.5 --snap=50 --kimg=100000 --seed=<SEED> --description=seed<SEED> --cond=1 --cfg=auto --mirror=1 --metrics=fid50k_full,pr50k3_full,cvar,d_preds --batch=64
```

## NoisyTwins

```bash
# Replace <SEED> by 111, 222, and 333. (3 runs)
./train.py --outdir=out --wandb_projname=flowers --wandb_groupname=noisytwins --noisytwins=1 --nt_alpha=0 --nt_sigma=0.1 --nt_gamma=0.005 --nt_lambda=0.001 --nt_start_kimg=0 --data=./datasets/flowers-lt.zip --snap=50 --kimg=40000 --seed=<SEED> --description=seed<SEED> --cond=1 --cfg=auto --mirror=1 --metrics=fid50k_full,pr50k3_full,cvar,d_preds --batch=64
```

## UTLO

```bash
# Replace <SEED> by 111, 222, and 333. (3 runs)
./train.py --outdir=out --wandb_projname=flowers --wandb_groupname=utlo --utlo=1 --res_uc=16 --utlo_lambda=10 --data=./datasets/flowers-lt.zip --snap=50 --kimg=25000 --seed=<SEED> --description=seed<SEED> --cond=1 --cfg=auto --mirror=1 --metrics=fid50k_full,pr50k3_full,cvar,d_preds --batch=64
```

# CIFAR10-LT

## SG2+ADA

```bash
# Replace <SEED> by 111, 222, and 333. (3 runs)
./train.py --outdir=out --wandb_projname=cifar10 --wandb_groupname=sg2+ada --data=./datasets/cifar-10_train --data_fname=lt_100.json --snap=50 --kimg=150000 --seed=<SEED> --description=seed<SEED> --cond=1 --cfg=cifar --mirror=1 --metrics=fid50k_full,pr50k3_full,cvar,d_preds --batch=64
```

## Transitional

```bash
# Replace <SEED> by 111, 222, and 333. (3 runs)
./train.py --outdir=out --transition=1 --wandb_projname=cifar10 --wandb_groupname=transitional --data=./datasets/cifar-10_train --data_fname=lt_100.json --snap=50 --kimg=150000 --seed=<SEED> --description=seed<SEED> --cond=1 --cfg=cifar --mirror=1 --metrics=fid50k_full,pr50k3_full,cvar,d_preds --batch=64
```

## LeCam

```bash
# Replace <SEED> by 111, 222, and 333. (3 runs)
./train.py --outdir=out --wandb_projname=cifar10 --wandb_groupname=lecam --lecam=1 --lecam_lambda=3e-7 --data=./datasets/cifar-10_train --data_fname=lt_100.json --snap=50 --kimg=150000 --seed=<SEED> --description=seed<SEED> --cond=1 --cfg=cifar --mirror=1 --metrics=fid50k_full,pr50k3_full,cvar,d_preds --batch=64
```

## GSR

```bash
# Replace <SEED> by 111, 222, and 333. (3 runs)
./train.py --outdir=out --wandb_projname=cifar10 --wandb_groupname=gsr --data=./datasets/cifar-10_train --data_fname=lt_100.json --gsr=1 --gsr_lambda=0.5 --snap=50 --kimg=150000 --seed=<SEED> --description=seed<SEED> --cond=1 --cfg=cifar --mirror=1 --metrics=fid50k_full,pr50k3_full,cvar,d_preds --batch=64
```

## NoisyTwins

```bash
# Replace <SEED> by 111, 222, and 333. (3 runs)
./train.py --outdir=out --wandb_projname=cifar10 --wandb_groupname=noisytwins --noisytwins=1 --nt_alpha=0.99 --nt_sigma=0.75 --nt_gamma=0.05 --nt_lambda=0.01 --data=./datasets/cifar-10_train --data_fname=lt_100.json --snap=50 --kimg=150000 --seed=<SEED> --description=seed<SEED> --cond=1 --cfg=cifar --mirror=1 --metrics=fid50k_full,pr50k3_full,cvar,d_preds --batch=64
```

## UTLO

```bash
# Replace <SEED> by 111, 222, and 333. (3 runs)
./train.py --outdir=out --wandb_projname=cifar10 --wandb_groupname=utlo --utlo=1 --res_uc=8 --utlo_lambda=10 --data=./datasets/cifar-10_train --data_fname=lt_100.json --snap=50 --kimg=150000 --seed=<SEED> --description=seed<SEED> --cond=1 --cfg=cifar --mirror=1 --metrics=fid50k_full,pr50k3_full,cvar,d_preds --batch=64
```

# iNaturalist2019

## SG2+ADA

```bash
# Replace <SEED> by 111, 222, and 333. (3 runs)
./train.py --outdir=out --wandb_projname=inaturalist2019 --wandb_groupname=sg2+ada --data=./datasets/inat_lt_train --snap=50 --kimg=160000 --seed=<SEED> --description=seed<SEED> --cond=1 --cfg=auto --mirror=1 --metrics=fid50k_full,pr50k3_full,cvar,d_preds --style_mixing_p=0.9 --gamma=0.2048 --ema=20 --ramp=0 --nmap=2 --batch=64
```

## Transitional

```bash
# Replace <SEED> by 111, 222, and 333. (3 runs)
./train.py --outdir=out --transition=1 --wandb_projname=inaturalist2019 --wandb_groupname=transitional --data=./datasets/inat_lt_train --snap=50 --kimg=160000 --seed=<SEED> --description=seed<SEED> --cond=1 --cfg=auto --mirror=1 --metrics=fid50k_full,pr50k3_full,cvar,d_preds --style_mixing_p=0.9 --gamma=0.2048 --ema=20 --ramp=0 --nmap=2 --batch=64
```

## LeCam

```bash
# Replace <SEED> by 111, 222, and 333. (3 runs)
./train.py --outdir=out --wandb_projname=inaturalist2019 --wandb_groupname=lecam --lecam=1 --lecam_lambda=3e-7 --data=./datasets/inat_lt_train --snap=50 --kimg=160000 --seed=<SEED> --description=seed<SEED> --cond=1 --cfg=auto --mirror=1 --metrics=fid50k_full,pr50k3_full,cvar,d_preds --style_mixing_p=0.9 --gamma=0.2048 --ema=20 --ramp=0 --nmap=2 --batch=64
```

## GSR

```bash
# Replace <SEED> by 111, 222, and 333. (3 runs)
./train.py --outdir=out --wandb_projname=inaturalist2019 --wandb_groupname=gsr --data=./datasets/inat_lt_train --gsr=1 --gsr_lambda=0.5 --snap=50 --kimg=150000 --seed=<SEED> --description=seed<SEED> --cond=1 --cfg=auto --mirror=1 --metrics=fid50k_full,pr50k3_full,cvar,d_preds --style_mixing_p=0.9 --gamma=0.2048 --ema=20 --ramp=0 --nmap=2 --batch=64
```

## NoisyTwins

```bash
# Replace <SEED> by 111, 222, and 333. (3 runs)
./train.py --outdir=out --wandb_projname=inaturalist2019 --wandb_groupname=noisytwins --noisytwins=1 --nt_alpha=0 --nt_sigma=0.1 --nt_gamma=0.005 --nt_lambda=0.001 --nt_start_kimg=6 --data=./datasets/inat_lt_train --snap=50 --kimg=160000 --seed=<SEED> --description=seed<SEED> --cond=1 --cfg=auto --mirror=1 --metrics=fid50k_full,pr50k3_full,cvar,d_preds --style_mixing_p=0.9 --gamma=0.2048 --ema=20 --ramp=0 --nmap=2 --batch=64
```

## UTLO

```bash
# Replace <SEED> by 111, 222, and 333. (3 runs)
./train.py --outdir=out --wandb_projname=inaturalist2019 --wandb_groupname=utlo --utlo=1 --res_uc=8 --utlo_lambda=1 --data=./datasets/inat_lt_train --snap=50 --kimg=160000 --seed=<SEED> --description=seed<SEED> --cond=1 --cfg=auto --mirror=1 --metrics=fid50k_full,pr50k3_full,cvar,d_preds --style_mixing_p=0.9 --gamma=0.2048 --ema=20 --ramp=0 --nmap=2 --batch=64
```

# ImageNet-LT

## SG2+ADA

```bash
# Replace <SEED> by 111, 222, and 333. (3 runs)
./train.py --outdir=out --wandb_projname=imagenet --wandb_groupname=sg2+ada --data=./datasets/imagenet_lt_train --snap=50 --kimg=160000 --seed=<SEED> --description=seed<SEED> --cond=1 --cfg=auto --mirror=1 --metrics=fid50k_full,pr50k3_full,cvar,d_preds --style_mixing_p=0.9 --gamma=0.2048 --ema=20 --ramp=0 --nmap=8 --batch=64
```

## Transitional

```bash
# Replace <SEED> by 111, 222, and 333. (3 runs)
./train.py --outdir=out --transition=1 --wandb_projname=imagenet --wandb_groupname=transitional --data=./datasets/imagenet_lt_train --snap=50 --kimg=160000 --seed=<SEED> --description=seed<SEED> --cond=1 --cfg=auto --mirror=1 --metrics=fid50k_full,pr50k3_full,cvar,d_preds --style_mixing_p=0.9 --gamma=0.2048 --ema=20 --ramp=0 --nmap=8 --batch=64
```

## LeCam

```bash
# Replace <SEED> by 111, 222, and 333. (3 runs)
./train.py --outdir=out --wandb_projname=imagenet --wandb_groupname=lecam --lecam=1 --lecam_lambda=3e-7 --data=./datasets/imagenet_lt_train --snap=50 --kimg=160000 --seed=<SEED> --description=seed<SEED> --cond=1 --cfg=auto --mirror=1 --metrics=fid50k_full,pr50k3_full,cvar,d_preds --style_mixing_p=0.9 --gamma=0.2048 --ema=20 --ramp=0 --nmap=8 --batch=64
```

## GSR

```bash
# Replace <SEED> by 111, 222, and 333. (3 runs)
./train.py --outdir=out --wandb_projname=imagenet --wandb_groupname=gsr --data=./datasets/imagenet_lt_train --gsr=1 --gsr_lambda=0.5 --snap=50 --kimg=150000 --seed=<SEED> --description=seed<SEED> --cond=1 --cfg=auto --mirror=1 --metrics=fid50k_full,pr50k3_full,cvar,d_preds --style_mixing_p=0.9 --gamma=0.2048 --ema=20 --ramp=0 --nmap=8 --batch=64
```

## NoisyTwins

```bash
# Replace <SEED> by 111, 222, and 333. (3 runs)
./train.py --outdir=out --wandb_projname=imagenet --wandb_groupname=noisytwins --noisytwins=1 --nt_alpha=0 --nt_sigma=0.25 --nt_gamma=0.005 --nt_lambda=0.001 --data=./datasets/imagenet_lt_train --snap=50 --kimg=100000 --seed=<SEED> --description=seed<SEED> --cond=1 --cfg=auto --mirror=1 --metrics=fid50k_full,pr50k3_full,cvar,d_preds --style_mixing_p=0.9 --gamma=0.2048 --ema=20 --ramp=0 --nmap=8 --batch=64
```

## UTLO

```bash
# Replace <SEED> by 111, 222, and 333. (3 runs)
./train.py --outdir=out --wandb_projname=imagenet --wandb_groupname=utlo --utlo=1 --res_uc=8 --utlo_lambda=1 --data=./datasets/imagenet_lt_train --snap=50 --kimg=160000 --seed=<SEED> --description=seed<SEED> --cond=1 --cfg=auto --mirror=1 --metrics=fid50k_full,pr50k3_full,cvar,d_preds --style_mixing_p=0.9 --gamma=0.2048 --ema=20 --ramp=0 --nmap=8 --batch=64
```


