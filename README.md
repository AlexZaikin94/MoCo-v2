# a PyTorch implementation of [MoCo-v1](https://arxiv.org/abs/1911.05722) with [MoCo-v2](https://arxiv.org/abs/2003.04297) improvements


## Enviorment
to install requirements run:
```
conda env create -f environment.yml
```


## Downloading Data
simply run:
```
python download_data.py
```


## Unsupervised Pre-Training
Hyperparameters for recreating our results:
<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">backbone</th>
<th valign="bottom">batch size</th>
<th valign="bottom">temperature</th>
<th valign="bottom">queue size</th>
<th valign="bottom">cos</th>
<th valign="bottom">mlp</th>
<th valign="bottom">aug+</th>
<!-- TABLE BODY -->
<tr><td align="left">ResNeXt-50-32x4d</td>
<td align="center">32</td>
<td align="center">0.2</td>
<td align="center">16384</td>
<td align="center">&#x2713</td>
<td align="center">&#x2713</td>
<td align="center">&#x2713</td>
</tr>
</tbody></table>

for recreating our results, adjust the `config.py` file according to our hyperparameters and on a single GPU machine, simply run:
```
python main_moco.py
```

or on a [Slurm](https://slurm.schedmd.com/) cluster, run:
```
sbatch -c 2 --gres=gpu:1 -o out_moco.out -J run_moco run_moco.sh
```

## Linear Classification
Using the pre-trained model with frozen weights, we achieve `92.8%` top-1 accuracy, using Linear classification on the [Imagenette](https://github.com/fastai/imagenette) validation set.

for recreating our results, adjust the `config.py` file according to our hyperparameters and on a single GPU machine, simply run:
```
python main_clf.py
```

or on a [Slurm](https://slurm.schedmd.com/) cluster, run:
```
sbatch -c 2 --gres=gpu:1 -o out_clf.out -J run_clf run_clf.sh
```

## Results
<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom"></th>
<th valign="bottom">epochs</th>
<th valign="bottom">time</th>
<th valign="bottom">top-1 accuracy</th>
<th valign="bottom">checkpoint</th>
<!-- TABLE BODY -->
<tr><td align="left">pre-training phase</td>
<td align="center">600</td>
<td align="center">97 hours</td>
<td align="center">0.92367</td>
<td align="center"><a href="https://1drv.ms/u/s!AtvUxcft_YQ-g-kV29FAUS-f8hWecg?e=OQtPKa">download</a></td>
</tr>
<tr><td align="left">linear classification phase</td>
<td align="center">200</td>
<td align="center">5 hours</td>
<td align="center">0.92777</td>
<td align="center"><a href="https://1drv.ms/u/s!AtvUxcft_YQ-g-kUXSn_CX7o_nn5Yw?e=54u59f">download</a></td>
</tr>
</tbody></table>

for evaluating the results on the Imagenette train and validation dataset, simply run (after downloading the data):
```
python evaluate.py
```
or on a [Slurm](https://slurm.schedmd.com/) cluster, run:
```
sbatch -c 2 --gres=gpu:1 -o out_evaluate.out -J run_evaluate run_evaluate.sh
```

training logs can be found in [`logs`](https://github.com/AlexZaikin94/MoCo-v2/tree/master/logs)

### Pre-Training loss
<p align="center">
  <img src="https://github.com/AlexZaikin94/MoCo-v2/blob/master/logs/moco_loss.png" width="700">
</p>

### Pre-Training top-1 accuracy
<p align="center">
  <img src="https://github.com/AlexZaikin94/MoCo-v2/blob/master/logs/moco_score.png" width="700">
</p>

### linear classification loss
<p align="center">
  <img src="https://github.com/AlexZaikin94/MoCo-v2/blob/master/logs/clf_loss.png" width="700">
</p>

### linear classification top-1 accuracy
<p align="center">
  <img src="https://github.com/AlexZaikin94/MoCo-v2/blob/master/logs/clf_score.png" width="700">
</p>


## Implementation details
* our implementation only supports single-gpu training, so we do not implement batch-shuffle.
* top-1 accuracy reported for training is an approximation, using the encodings (train and val, with no_gran and evaluation mode).
* we run training in a few consecutive sessions on Nvidia GeForce GTX 1080 Ti/GeForce GTX 2080 Ti/Titan Xp.



