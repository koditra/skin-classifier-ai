import torch
from torch.utils.data import WeightedRandomSampler


def get_class_weights(dataset):
    counts = torch.zeros(len(dataset.classes))

    for l in dataset.labels:
        counts[l] += 1

    weights = 1.0 / (counts + 1e-6)
    weights = weights / weights.sum() * len(weights)

    return weights


def get_sampler(dataset):
    counts = torch.zeros(len(dataset.classes))

    for l in dataset.labels:
        counts[l] += 1

    weights = 1.0 / (counts + 1e-6)

    sample_weights = [weights[l] for l in dataset.labels]

    return WeightedRandomSampler(
        sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )