"""
Implementation of various active learning techniques for the course FSDL Spring 2021 that can be used with modAL.
"""
from typing import Tuple
import torch
import numpy as np
from modAL.models import ActiveLearner

T_DEFAULT = 10
N_INSTANCES_DEFAULT = 100
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32
DEBUG_OUTPUT = True


def max_entropy(learner: ActiveLearner, X: np.array, n_instances: int = N_INSTANCES_DEFAULT, T: int = T_DEFAULT) -> Tuple[torch.Tensor, np.array]:
    """Active learning sampling technique that maximizes the predictive entropy based on the paper
    'Deep Bayesian Active Learning with Image Data' (https://arxiv.org/pdf/1703.02910.pdf).

    Examples
    --------
    >>> classifier = skorch.NeuralNetClassifier(MyModelClass, ...)
    >>> learner = modAL.models.ActiveLearner(estimator = classifier, query_strategy = max_entropy, ...) # set max_entropy strategy here
    >>> query_idx, query_instance = learner.query(sample_pool_X, ...) # strategy is then used here
    >>> learner.teach(X = sample_pool_X[query_idx], y = sample_pool_y[query_idx], ...)

    Parameters
    ----------
    learner : modAL.models.ActiveLearner
      modAL ActiveLearner instance with which the sampling technique should be used

    X : numpy.array 
      Array of instances from which to sample from

    n_instances : int (default = 100)
      Number of instsances that should be sampled

    T : int (default = 10)
      Number of predictions to generate per X instance from which then the entropy is estimated via taking the mean (see paper for details)

    Returns
    -------
    Tuple of indexes and corresponding data instances that were chosen based on the sampling strategy.
    """

    return _batched_pytorch_tensor_loop(learner, X, n_instances, _max_entropy_scoring_function, T=T)


def bald(learner: ActiveLearner, X: np.array, n_instances: int = N_INSTANCES_DEFAULT, T: int = T_DEFAULT) -> Tuple[torch.Tensor, np.array]:
    """Active learning sampling technique that maximizes the information gain via maximising mutual information between predictions 
    and model posterior (Bayesian Active Learning by Disagreement - BALD) as depicted in the papers 'Deep Bayesian Active Learning 
    with Image Data' (https://arxiv.org/pdf/1703.02910.pdf) and 'Bayesian Active Learning for Classification and Preference Learning' 
    (https://arxiv.org/pdf/1112.5745.pdf).

    Examples
    --------
    >>> classifier = skorch.NeuralNetClassifier(MyModelClass, ...)
    >>> learner = modAL.models.ActiveLearner(estimator = classifier, query_strategy = bald, ...) # set bald strategy here
    >>> query_idx, query_instance = learner.query(sample_pool_X, ...) # strategy is then used here
    >>> learner.teach(X = sample_pool_X[query_idx], y = sample_pool_y[query_idx], ...)

    Parameters
    ----------
    learner : modAL.models.ActiveLearner
      modAL ActiveLearner instance with which the sampling technique should be used

    X : numpy.array 
      Array of instances from which to sample from

    n_instances : int (default = 100)
      Number of instsances that should be sampled

    T : int (default = 10)
      Number of predictions to generate per X instance from which then the entropy is estimated via taking the mean (see paper for details)

    Returns
    -------
    Tuple of indexes and corresponding data instances that were chosen based on the sampling strategy.
    """

    return _batched_pytorch_tensor_loop(learner, X, n_instances, _bald_scoring_function, T=T)


def random(learner: ActiveLearner, X: np.array, n_instances: int = N_INSTANCES_DEFAULT) -> Tuple[torch.Tensor, np.array]:
    """Baseline active learning sampling technique that takes random instances from available pool.

    Examples
    --------
    >>> classifier = skorch.NeuralNetClassifier(MyModelClass, ...)
    >>> learner = modAL.models.ActiveLearner(estimator = classifier, query_strategy = random, ...) # set random strategy here
    >>> query_idx, query_instance = learner.query(sample_pool_X, ...) # strategy is then used here
    >>> learner.teach(X = sample_pool_X[query_idx], y = sample_pool_y[query_idx], ...)

    Parameters
    ----------
    learner : modAL.models.ActiveLearner
      modAL ActiveLearner instance with which the sampling technique should be used

    X : numpy.array 
      Array of instances from which to sample from

    n_instances : int (default = 100)
      Number of instsances that should be sampled

    Returns
    -------
    Tuple of indexes and corresponding data instances that were chosen based on the sampling strategy.
    """
    
    query_idx = np.random.choice(range(len(X)), size=n_instances, replace=False)
    return query_idx, X[query_idx]


def _batched_pytorch_tensor_loop(learner, X, n_instances, batch_scoring_function, **kwargs): 

    if DEBUG_OUTPUT:
        print("Processing pool of instances with selected active learning strategy...")
        print("(Note: Based on the pool size this takes a while. Will generate debug output every 10%.)")
        ten_percent = int(len(X)/BATCH_SIZE/10)
        i = 0
        percentage_output = 10

    # initialize pytorch tensor to store acquisition scores
    all_acquisitions = torch.Tensor().to(DEVICE)

    # create pytorch dataloader for batch-wise processing
    all_samples = torch.utils.data.DataLoader(X, batch_size=BATCH_SIZE)

    # process pool of instances batch wise
    for batch in all_samples:

        acquisitions = batch_scoring_function(learner, batch, **kwargs)
        
        all_acquisitions = torch.cat([all_acquisitions, acquisitions])

        if DEBUG_OUTPUT:
            i += 1
            if i > ten_percent:
                print(f"{percentage_output}% of samples in pool processed")
                percentage_output += 10
                i = 0

    # collect first n_instances to cpu and return
    idx = torch.argsort(-all_acquisitions)[:n_instances].cpu()
    return idx, X[idx]


def _bald_scoring_function(learner, batch, T):

    with torch.no_grad():

        outputs = torch.stack([
            torch.softmax( # probabilities from logits
                learner.estimator.forward(batch, training=True, device=DEVICE), dim=-1) # logits
            for t in range(T) # multiple calculations to average over
            ])

    mean_outputs = torch.mean(outputs, dim=0)

    H = torch.sum(-mean_outputs*torch.log(mean_outputs + 1e-10), dim=-1)
    E_H = - torch.mean(torch.sum(outputs * torch.log(outputs + 1e-10), dim=-1), dim=0)
    acquisitions = H - E_H

    return acquisitions


def _max_entropy_scoring_function(learner, batch, T):

    with torch.no_grad():

        outputs = torch.stack([
            torch.softmax( # probabilities from logits
                learner.estimator.forward(batch, training=True, device=DEVICE), dim=-1) # logits
            for t in range(T) # multiple calculations to average over
            ])

    mean_outputs = torch.mean(outputs, dim=0)

    acquisitions = torch.sum((-mean_outputs * torch.log(mean_outputs + 1e-10)), dim=-1)
    return acquisitions
