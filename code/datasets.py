from torchvision import transforms, datasets
from typing import *
import torch
import os
import sklearn.datasets
import math
from torch.utils.data import Dataset, TensorDataset

# set this environment variable to the location of your imagenet directory if you want to read ImageNet data.
# make sure your val directory is preprocessed to look like the train directory, e.g. by running this script
# https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh
IMAGENET_LOC_ENV = "IMAGENET_DIR"

# list of all datasets
DATASETS = ["imagenet", "cifar10", "mnist", "small_mnist", "small_mnist_10pcs",  "swiss_roll"]


def get_dataset(dataset: str, split: str) -> Dataset:
    """Return the dataset as a PyTorch Dataset object"""
    if dataset == "imagenet":
        return _imagenet(split)
    elif dataset == "cifar10":
        return _cifar10(split)
    elif dataset == "mnist":
        return _mnist(split)
    elif dataset == "small_mnist":
        return _small_mnist(split)
    elif dataset == "small_mnist_10pcs":
        return _small_mnist_10pcs(split)
    elif dataset == "swiss_roll":
        return _swiss_roll(split)

def get_num_classes(dataset: str):
    """Return the number of classes in the dataset. """
    if dataset == "imagenet":
        return 1000
    elif dataset == "cifar10":
        return 10
    elif dataset == "mnist":
        return 10
    elif dataset == "small_mnist":
        return 10
    elif dataset == "small_mnist_10pcs":
        return 10
    elif dataset == "swiss_roll":
        return 2


def get_normalize_layer(dataset: str) -> torch.nn.Module:
    """Return the dataset's normalization layer"""
    if dataset == "imagenet":
        return NormalizeLayer(_IMAGENET_MEAN, _IMAGENET_STDDEV)
    elif dataset == "cifar10":
        return NormalizeLayer(_CIFAR10_MEAN, _CIFAR10_STDDEV)
    elif dataset == "mnist":
        return NormalizeLayer(_MNIST_MEAN, _MNIST_STDDEV)
    elif dataset == "small_mnist":
        return NormalizeLayer(_SMALL_MNIST_MEAN, _SMALL_MNIST_STDDEV)
    elif dataset == "small_mnist_10pcs":
        return torch.nn.Identity() #Do not normalize PCA results
    elif dataset == "swiss_roll":
        return torch.nn.Identity() #Do not normalize swiss roll

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STDDEV = [0.229, 0.224, 0.225]

_CIFAR10_MEAN = [0.4914, 0.4822, 0.4465]
_CIFAR10_STDDEV = [0.2023, 0.1994, 0.2010]

_MNIST_MEAN = [0.1307]
_MNIST_STDDEV = [0.3081]

_SMALL_MNIST_MEAN = [0.1307]
_SMALL_MNIST_STDDEV = [0.2307]


def _cifar10(split: str) -> Dataset:
    if split == "train":
        return datasets.CIFAR10("./dataset_cache", train=True, download=True, transform=transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ]))
    elif split == "test":
        return datasets.CIFAR10("./dataset_cache", train=False, download=True, transform=transforms.ToTensor())

def _mnist(split: str) -> Dataset:
    if split == "train":
        return datasets.MNIST("./dataset_cache", train=True, download=True, transform=transforms.ToTensor())
    elif split == "test":
        return datasets.MNIST("./dataset_cache", train=False, download=True, transform=transforms.ToTensor())

def _small_mnist(split: str) -> Dataset:
    if split == "train":
        return datasets.MNIST("./dataset_cache", train=True, download=True, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x : x.reshape([1,28,7,4]).mean(dim=3).reshape([1,7,4,7]).mean(dim=2))
        ]))
    elif split == "test":
        return datasets.MNIST("./dataset_cache", train=False, download=True, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x : x.reshape([1,28,7,4]).mean(dim=3).reshape([1,7,4,7]).mean(dim=2))
        ]))
def _small_mnist_10pcs(split: str) -> Dataset:
    pca = torch.load('small_mnist_10_pcs.pth')
    if split == "train":
        return datasets.MNIST("./dataset_cache", train=True, download=True, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x : torch.tensor(pca.transform(x.reshape([1,28,7,4]).mean(dim=3).reshape([1,7,4,7]).mean(dim=2).reshape(1,49).numpy())).reshape(1,1,10))
        ]))
    elif split == "test":
        return datasets.MNIST("./dataset_cache", train=False, download=True, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x : torch.tensor(pca.transform(x.reshape([1,28,7,4]).mean(dim=3).reshape([1,7,4,7]).mean(dim=2).reshape(1,49).numpy())).reshape(1,1,10))
        ]))

def _swiss_roll(split: str) -> Dataset:
    if split == "train":
        raw, raw_labels = sklearn.datasets.make_swiss_roll(random_state=1,noise=1.0, n_samples = 1000)
        feats = raw[:,::2]/10.
        labels = (raw_labels > 3*math.pi).astype(int)
        return TensorDataset(torch.tensor(feats).float(),torch.tensor(labels))
    elif split == "test":
        raw, raw_labels = sklearn.datasets.make_swiss_roll(random_state=2,noise=1.0, n_samples = 100)
        feats = raw[:,::2]/10.
        labels = (raw_labels > 3*math.pi).astype(int)
        return TensorDataset(torch.tensor(feats).float(),torch.tensor(labels))

def _imagenet(split: str) -> Dataset:
    if not IMAGENET_LOC_ENV in os.environ:
        raise RuntimeError("environment variable for ImageNet directory not set")

    dir = os.environ[IMAGENET_LOC_ENV]
    if split == "train":
        subdir = os.path.join(dir, "train")
        transform = transforms.Compose([
            transforms.RandomSizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])
    elif split == "test":
        subdir = os.path.join(dir, "val")
        transform = transforms.Compose([
            transforms.Scale(256),
            transforms.CenterCrop(224),
            transforms.ToTensor()
        ])
    return datasets.ImageFolder(subdir, transform)


class NormalizeLayer(torch.nn.Module):
    """Standardize the channels of a batch of images by subtracting the dataset mean
      and dividing by the dataset standard deviation.

      In order to certify radii in original coordinates rather than standardized coordinates, we
      add the Gaussian noise _before_ standardizing, which is why we have standardization be the first
      layer of the classifier rather than as a part of preprocessing as is typical.
      """

    def __init__(self, means: List[float], sds: List[float]):
        """
        :param means: the channel means
        :param sds: the channel standard deviations
        """
        super(NormalizeLayer, self).__init__()
        self.means = torch.tensor(means).cuda()
        self.sds = torch.tensor(sds).cuda()

    def forward(self, input: torch.tensor):
        (batch_size, num_channels, height, width) = input.shape
        means = self.means.repeat((batch_size, height, width, 1)).permute(0, 3, 1, 2)
        sds = self.sds.repeat((batch_size, height, width, 1)).permute(0, 3, 1, 2)
        return (input - means) / sds
