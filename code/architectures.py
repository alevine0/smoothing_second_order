import torch
from torchvision.models.resnet import resnet50
import torch.backends.cudnn as cudnn
from archs.cifar_resnet import resnet as resnet_cifar
from archs.mnist_classifier import MnistNet
from archs.small_mnist_classifier import SmallMnistNet
from archs.small_mnist_pca_classifier import SmallMnistPCANet
from archs.swiss_roll_classifier import SwissRollNet
from datasets import get_normalize_layer
from torch.nn.functional import interpolate

# resnet50 - the classic ResNet-50, sized for ImageNet
# cifar_resnet20 - a 20-layer residual network sized for CIFAR
# cifar_resnet110 - a 110-layer residual network sized for CIFAR
ARCHITECTURES = ["resnet50", "cifar_resnet20", "cifar_resnet110", "mnist_net", "small_mnist_net", "small_mnist_pca_net", "swiss_roll_net"]

def get_architecture(arch: str, dataset: str) -> torch.nn.Module:
    """ Return a neural network (with random weights)

    :param arch: the architecture - should be in the ARCHITECTURES list above
    :param dataset: the dataset - should be in the datasets.DATASETS list
    :return: a Pytorch module
    """
    if arch == "resnet50" and dataset == "imagenet":
        model = torch.nn.DataParallel(resnet50(pretrained=False)).cuda()
        cudnn.benchmark = True
    elif arch == "cifar_resnet20":
        model = resnet_cifar(depth=20, num_classes=10).cuda()
    elif arch == "cifar_resnet110":
        model = resnet_cifar(depth=110, num_classes=10).cuda()
    elif arch == "mnist_net":
        model = MnistNet().cuda()
    elif arch == "small_mnist_net":
        model = SmallMnistNet().cuda()
    elif arch == "small_mnist_pca_net":
        model = SmallMnistPCANet().cuda()
    elif arch == "swiss_roll_net":
        model = SwissRollNet().cuda()
    normalize_layer = get_normalize_layer(dataset)
    return torch.nn.Sequential(normalize_layer, model)
