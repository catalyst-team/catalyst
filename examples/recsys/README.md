# RecSys Learning Examples

## Requirements

To run examples you need `catalyst[ml]>=21.10`
```bash
pip install catalyst[ml]
```

## Models

### MultiVAE

Implementation based on the article [Variational Autoencoders for Collaborative Filtering](https://arxiv.org/pdf/1802.05814.pdf).
Model train on [MovieLens 100K Dataset](https://grouplens.org/datasets/movielens/100k/). 
To make an implicit feedback dataset from an explicit one, all `ratings > 0` are replaced with `1`.
Train part is 80% of users, valid - 20%.

``collate_fn`` functions split each user interactions tensor on two tensors. ``collate_fn_train``: `inputs` and `targets`
tensors contain all 100% interactions. ``collate_fn_valid``: `inputs` contain 80% interactions per each user, `targets` - all 100%.

Run the example:
```bash
python multivae.py
```