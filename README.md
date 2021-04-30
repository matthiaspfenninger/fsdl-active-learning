# fsdl-active-learning

Comparing different active learning strategies for image classification (FSDL course 2021 capstone project)

- [fsdl-active-learning](#fsdl-active-learning)
  - [Introduction](#introduction)
  - [Relevant Changes Compared to Lab Template](#relevant-changes-compared-to-lab-template)
    - [DroughtWatch Data Set](#droughtwatch-data-set)
    - [ResNet Image Classifier](#resnet-image-classifier)
    - [modAL Active Learning Experiment Running Framework](#modal-active-learning-experiment-running-framework)
      - [modAL Sampling Strategy Extensions](#modal-sampling-strategy-extensions)
  - [Quickstart](#quickstart)
    - [Local](#local)
    - [Google Colab](#google-colab)

## Introduction

This repository builds upon the template of **lab 08** of the [Full Stack Deep Learning Spring 2021 labs](https://github.com/full-stack-deep-learning/fsdl-text-recognizer-2021-labs) and extends it with a new dataset, model and active learning strategies.

## Relevant Changes Compared to Lab Template

### DroughtWatch Data Set

[text_recognizer/data/droughtwatch.py](./text_recognizer/data/droughtwatch.py): Downloads data from the [W&B Drought Prediction Benchmark](https://github.com/wandb/droughtwatch) and converts it to HDF5 format which can be used by PyTorch for training and inference.

### ResNet Image Classifier

[text_recognizer/models/resnet_classifier.py](./text_recognizer/models/resnet_classifier.py): Implements a PyTorch ResNet model for image classification, with the following adaptions compared to the regular model:

- preprocessing steps (image resizing and normalization)
- class outputs (4 instead of 1000)
- optional dropout layer at the end

The model can be used for transfer learning on the drought prediction data.

### modAL Active Learning Experiment Running Framework

[training/run_modAL_experiment.py](./training/run_modAL_experiment.py): Script to run experiments for model training with different active learning strategies which are implemented via the [modAL library](https://github.com/modAL-python/modAL).

#### modAL Sampling Strategy Extensions

[text_recognizer/active_learning/modal_extensions.py](./text_recognizer/active_learning/modal_extensions.py): Implementation of different sampling strategies for active learning via [modAL library](https://github.com/modAL-python/modAL).

Note that the strategies `bald` and `max_entropy` only make sense when there is a `dropout` layer in the network.

## Quickstart

### Local

```bash
git pull [repo-url] # clone from git
cd [folder]

make conda-update # creates a conda env with the base packages
conda activate fsdl-active-learning-2021 # activates the conda env
make pip-tools # installs required pip packages inside the conda env

# regular experiment, training a drought watch classifier via pytorch lightning
python training/run_experiment.py --max_epochs=1 --num_workers=4 --data_class=DroughtWatch --model_class=ResnetClassifier

# active learning experiment with modAL
python training/run_modAL_experiment.py --al_epochs_init=10 --al_epochs_incr=10 --al_n_iter=20 --al_samples_per_iter=1000 --al_incr_onlynew=False --al_query_strategy=margin_sampling --data_class=DroughtWatch --model_class=ResnetClassifier --batch_size=64 --n_train_images=20000 --n_validation_images=10778  --pretrained=True --wandb
```

### Google Colab

Refer to the example notebook under [notebooks/clone_repo_and_train.ipynb](./notebooks/clone_repo_and_train.ipynb).
