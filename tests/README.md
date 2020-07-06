# Catalyst tests and docs

We train a number of different models for various of tasks:
- [large number of different DL scripts](./_tests_scripts)
- [image classification](./_tests_cv_classification)
- [image classification with config transforms](./_tests_cv_classification_transforms)
- [image segmentation](./_tests_cv_segmentation)
- [text classification](./_tests_nlp_classification)
- [GAN training](../examples/mnist_gans)

During the tests, we compare their convergence metrics in order to verify 
the correctness of the training procedure and its reproducibility.

Our Continuous Integration pipelines
perform tests under various training conditions:
- Python 3.6, 3.7, 3.8
- PyTorch 1.1, 1.2, 1.3, 1.4, 1.5
- Linux, OSX
- CPU only
- 1 GPU
- 2 GPUs
- 1 GPU with fp16 training
- 2 GPUs with fp16 training
- 2 GPUs with fp16 distributed training

This provides testing for most combinations of important settings.
The tests expect the model to perform to a reasonable degree of testing accuracy to pass.

## Running tests

To run all tests do the following:
```bash
git clone https://github.com/catalyst-team/catalyst
cd catalyst

# install develop dependencies
pip install -r requirements/requirements.txt -r requirements/requirements-cv.txt -r requirements/requirements-nlp.txt -r requirements/requirements-ecosystem.txt -r requirements/requirements-dev.txt

# run python tests
pytest .

# run deep learning tests
## CPU
bash ./bin/teamcity/dl_cpu.sh
## GPU
bash ./bin/teamcity/dl_gpu.sh
## GPUs
bash ./bin/teamcity/dl_gpu2.sh
```

To test models that require GPU make sure to run the above command on a GPU machine.
The GPU machine may have:
1. At least 1 GPU.
2. [NVIDIA-apex](https://github.com/NVIDIA/apex#linux) installed.


## Running docs
```bash
git clone https://github.com/catalyst-team/catalyst
cd catalyst

# install develop dependencies
pip install -r requirements/requirements.txt -r requirements/requirements-cv.txt -r requirements/requirements-nlp.txt -r requirements/requirements-ecosystem.txt -r requirements/requirements-dev.txt -r docs/requirements.txt

# run docs
rm -rf ./builds; REMOVE_BUILDS=0 make check-docs
# open docs
open ./builds/index.html
```