# Self-Supervised Learning Examples
## Requriments

To run examples you need catalyst[cv]==22.02 and catalyst[ml]==22.02
```
pip install catalyst[cv]==22.02
pip install catalyst[ml]==22.02
```

## Description

All traing files have common command line parametrs:

    --dataset - Dataset: CIFAR-10, CIFAR-100 or STL10
    --logdir - Logs directory (tensorboard, weights, etc)
    --epochs - Number of sweeps over the dataset to train
    --batch-size - Number of images in each mini-batch
    --num-workers - Number of workers to process a dataloader
    --feature-dim - Feature dim for latent vector
    --temperature - Temperature used in softmax
    --learning-rate - Learning rate for optimizer

### Extra parametrs

Barlow-twins (barlow_twins.py) has an extra parametr ``--offdig-lambda`` - lambda that controls the on- and off-diagonal terms from Barlow twins loss.

## Usage

Implemented algorithms:
- Barlow-Twins: ``barlow_twins.py``
- BYOL: ``byol.py``
- SimClR: ``simCLR.py``
- Supervised contrastive: ``supervised_contrastive.py``

You can run an algorithm with a command:
```
python3 barlow_twins.py --batch-size 32
```
Also, you can use the Docker:
```
docker build . -t train-self-supervised
docker run train-self-supervised python3 simCLR.py --batch-size 32
```

## Test

You can test that all settled with ``run.sh``.

## Results

<details>
<summary>TITAN V JHH Special Edition</summary>
<p>

### % correctly classified samples with sklearn.LogisticRegression on learned representations.

| accuracy01 | Barlow Twins  | BYOL          | simCLR        | Supervised Contrastive |
|------------|---------------|---------------|---------------|------------------------|
| CIFAR-10   | 25.68±2.82    | *33.85±2.71*  | *32.92±3.30*  | **77.78±2.53**          |
| CIFAR-100  | 5.24±1.18     | *11.88±1.83*  | *10.49±1.77*  | **37.56±2.93**          |
| STL10      | 27.77±3.27    | *31.22±2.98*  | *34.37±2.71*  | **63.17±2.78**          |

| accuracy03 | Barlow Twins  | BYOL          | simCLR        | Supervised Contrastive |
|------------|---------------|---------------|---------------|------------------------|
| CIFAR-10   | 56.03±3.47    | *67.74±3.50*  | *65.87±3.06*  | **96.16±1.13**          |
| CIFAR-100  | 12.40±2.10    | *23.70±2.46*  | *22.15±2.45*  | **61.33±2.71**          |
| STL10      | *60.08±3.68*  | 64.50±2.45    | *69.43±2.58*  | **89.57±2.02**          |

- **Bold** - Top1 performance results
- *Italic* - Top2 performance results

</p>
</details>

