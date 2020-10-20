import numpy as np
import matplotlib

matplotlib.use("TkAgg")
matplotlib.rc('lines', linewidth=1.0)
import matplotlib.pyplot as plt
from typing import *
import pandas as pd
import seaborn as sns
import math

sns.set()


class Accuracy(object):
    def at_radii(self, radii: np.ndarray):
        raise NotImplementedError()


class ApproximateAccuracy(Accuracy):
    def __init__(self, data_file_path: str):
        self.data_file_path = data_file_path

    def at_radii(self, radii: np.ndarray) -> np.ndarray:
        df = pd.read_csv(self.data_file_path, delimiter="\t")
        return np.array([self.at_radius(df, radius) for radius in radii])

    def at_radius(self, df: pd.DataFrame, radius: float):
        return (df["correct"] & (df["radius"] >= radius)).mean()


class HighProbAccuracy(Accuracy):
    def __init__(self, data_file_path: str, alpha: float, rho: float):
        self.data_file_path = data_file_path
        self.alpha = alpha
        self.rho = rho

    def at_radii(self, radii: np.ndarray) -> np.ndarray:
        df = pd.read_csv(self.data_file_path, delimiter="\t")
        return np.array([self.at_radius(df, radius) for radius in radii])

    def at_radius(self, df: pd.DataFrame, radius: float):
        mean = (df["correct"] & (df["radius"] >= radius)).mean()
        num_examples = len(df)
        return (mean - self.alpha - math.sqrt(self.alpha * (1 - self.alpha) * math.log(1 / self.rho) / num_examples)
                - math.log(1 / self.rho) / (3 * num_examples))


class Line(object):
    def __init__(self, quantity: Accuracy, legend: str, plot_fmt: str = "", scale_x: float = 1):
        self.quantity = quantity
        self.legend = legend
        self.plot_fmt = plot_fmt
        self.scale_x = scale_x


def plot_certified_accuracy(outfile: str, title: str, max_radius: float,
                            lines: List[Line], radius_step: float = 0.01) -> None:
    radii = np.arange(0, max_radius + radius_step, radius_step)
    plt.figure()
    for line in lines:
        plt.plot(radii * line.scale_x, line.quantity.at_radii(radii), linewidth=1.0)

    plt.ylim((0, 1))
    plt.xlim((0, max_radius))
    plt.tick_params(labelsize=14)
    plt.xlabel("radius", fontsize=16)
    plt.ylabel("certified accuracy", fontsize=16)
    plt.legend([method.legend for method in lines], loc='upper right', fontsize=16)
    plt.savefig(outfile + ".pdf")
    plt.tight_layout()
    plt.title(title, fontsize=20)
    plt.tight_layout()
    plt.savefig(outfile + ".png", dpi=300)
    plt.close()


def smallplot_certified_accuracy(outfile: str, title: str, max_radius: float,
                                 methods: List[Line], radius_step: float = 0.01, xticks=0.5) -> None:
    radii = np.arange(0, max_radius + radius_step, radius_step)
    plt.figure()
    for method in methods:
        plt.plot(radii, method.quantity.at_radii(radii),  linewidth=1.0)#method.plot_fmt)

    plt.ylim((0, 1))
    plt.xlim((0, max_radius))
    plt.xlabel("radius", fontsize=22)
    plt.ylabel("certified accuracy", fontsize=22)
    plt.tick_params(labelsize=20)
    plt.gca().xaxis.set_major_locator(plt.MultipleLocator(xticks))
    plt.legend([method.legend for method in methods], loc='upper right', fontsize=20)
    plt.tight_layout()
    plt.savefig(outfile + ".pdf")
    plt.close()


def latex_table_certified_accuracy(outfile: str, radius_start: float, radius_stop: float, radius_step: float,
                                   methods: List[Line]):
    radii = np.arange(radius_start, radius_stop + radius_step, radius_step)
    accuracies = np.zeros((len(methods), len(radii)))
    for i, method in enumerate(methods):
        accuracies[i, :] = method.quantity.at_radii(radii)

    f = open(outfile, 'w')

    for radius in radii:
        f.write("& $r = {:.3}$".format(radius))
    f.write("\\\\\n")

    f.write("\midrule\n")

    for i, method in enumerate(methods):
        f.write(method.legend)
        for j, radius in enumerate(radii):
            if i == accuracies[:, j].argmax():
                txt = r" & \textbf{" + "{:.2f}".format(accuracies[i, j]) + "}"
            else:
                txt = " & {:.2f}".format(accuracies[i, j])
            f.write(txt)
        f.write("\\\\\n")
    f.close()


def markdown_table_certified_accuracy(outfile: str, radius_start: float, radius_stop: float, radius_step: float,
                                      methods: List[Line]):
    radii = np.arange(radius_start, radius_stop + radius_step, radius_step)
    accuracies = np.zeros((len(methods), len(radii)))
    for i, method in enumerate(methods):
        accuracies[i, :] = method.quantity.at_radii(radii)

    f = open(outfile, 'w')
    f.write("|  | ")
    for radius in radii:
        f.write("r = {:.3} |".format(radius))
    f.write("\n")

    f.write("| --- | ")
    for i in range(len(radii)):
        f.write(" --- |")
    f.write("\n")

    for i, method in enumerate(methods):
        f.write("<b> {} </b>| ".format(method.legend))
        for j, radius in enumerate(radii):
            if i == accuracies[:, j].argmax():
                txt = "{:.2f}<b>*</b> |".format(accuracies[i, j])
            else:
                txt = "{:.2f} |".format(accuracies[i, j])
            f.write(txt)
        f.write("\n")
    f.close()


if __name__ == "__main__":
    # latex_table_certified_accuracy(
    #     "analysis/latex/vary_noise_cifar10", 0.25, 1.5, 0.25, [
    #         Line(ApproximateAccuracy("data/certify/cifar10/resnet110/noise_0.12/test/sigma_0.12"), "$\sigma = 0.12$"),
    #         Line(ApproximateAccuracy("data/certify/cifar10/resnet110/noise_0.25/test/sigma_0.25"), "$\sigma = 0.25$"),
    #         Line(ApproximateAccuracy("data/certify/cifar10/resnet110/noise_0.50/test/sigma_0.50"), "$\sigma = 0.50$"),
    #         Line(ApproximateAccuracy("data/certify/cifar10/resnet110/noise_1.00/test/sigma_1.00"), "$\sigma = 1.00$"),
    #     ])
    # markdown_table_certified_accuracy(
    #     "analysis/markdown/vary_noise_cifar10", 0.25, 1.5, 0.25, [
    #         Line(ApproximateAccuracy("data/certify/cifar10/resnet110/noise_0.12/test/sigma_0.12"), "&sigma; = 0.12"),
    #         Line(ApproximateAccuracy("data/certify/cifar10/resnet110/noise_0.25/test/sigma_0.25"), "&sigma; = 0.25"),
    #         Line(ApproximateAccuracy("data/certify/cifar10/resnet110/noise_0.50/test/sigma_0.50"), "&sigma; = 0.50"),
    #         Line(ApproximateAccuracy("data/certify/cifar10/resnet110/noise_1.00/test/sigma_1.00"), "&sigma; = 1.00"),
    #     ])
    # latex_table_certified_accuracy(
    #     "analysis/latex/vary_noise_imagenet", 0.5, 3.0, 0.5, [
    #         Line(ApproximateAccuracy("data/certify/imagenet/resnet50/noise_0.25/test/sigma_0.25"), "$\sigma = 0.25$"),
    #         Line(ApproximateAccuracy("data/certify/imagenet/resnet50/noise_0.50/test/sigma_0.50"), "$\sigma = 0.50$"),
    #         Line(ApproximateAccuracy("data/certify/imagenet/resnet50/noise_1.00/test/sigma_1.00"), "$\sigma = 1.00$"),
    #     ])
    # markdown_table_certified_accuracy(
    #     "analysis/markdown/vary_noise_imagenet", 0.5, 3.0, 0.5, [
    #         Line(ApproximateAccuracy("data/certify/imagenet/resnet50/noise_0.25/test/sigma_0.25"), "&sigma; = 0.25"),
    #         Line(ApproximateAccuracy("data/certify/imagenet/resnet50/noise_0.50/test/sigma_0.50"), "&sigma; = 0.50"),
    #         Line(ApproximateAccuracy("data/certify/imagenet/resnet50/noise_1.00/test/sigma_1.00"), "&sigma; = 1.00"),
    #     ])
    # plot_certified_accuracy(
    #     "analysis/plots/vary_noise_cifar10", "CIFAR-10, vary $\sigma$", 1.5, [
    #         Line(ApproximateAccuracy("data/certify/cifar10/resnet110/noise_0.12/test/sigma_0.12"), "$\sigma = 0.12$"),
    #         Line(ApproximateAccuracy("data/certify/cifar10/resnet110/noise_0.25/test/sigma_0.25"), "$\sigma = 0.25$"),
    #         Line(ApproximateAccuracy("data/certify/cifar10/resnet110/noise_0.50/test/sigma_0.50"), "$\sigma = 0.50$"),
    #         Line(ApproximateAccuracy("data/certify/cifar10/resnet110/noise_1.00/test/sigma_1.00"), "$\sigma = 1.00$"),
      #  ])
    # plot_certified_accuracy(
    #     "analysis/plots/vary_train_noise_cifar_050", "CIFAR-10, vary train noise, $\sigma=0.5$", 1.5, [
    #         Line(ApproximateAccuracy("data/certify/cifar10/resnet110/noise_0.25/test/sigma_0.50"), "train $\sigma = 0.25$"),
    #         Line(ApproximateAccuracy("data/certify/cifar10/resnet110/noise_0.50/test/sigma_0.50"), "train $\sigma = 0.50$"),
    #         Line(ApproximateAccuracy("data/certify/cifar10/resnet110/noise_1.00/test/sigma_0.50"), "train $\sigma = 1.00$"),
    #     ])
    plot_certified_accuracy(
        "analysis/plots/vary_method_cifar_100000_0.25", "CIFAR, vary method, $\sigma=0.25$, N=10^5", 2., [
            Line(ApproximateAccuracy("data/certify/cifar10/resnet110/noise_0.25/test/sigma_0.25"), "Baseline"),
            Line(ApproximateAccuracy("data/certify/cifar10/resnet110/noise_0.25/test/dipole_sigma_0.25"), "Dipole"),
        ])

    plot_certified_accuracy(
        "analysis/plots/vary_method_cifar_100000_0.12", "CIFAR, vary method, $\sigma=0.12$, N=10^5", 2., [
            Line(ApproximateAccuracy("data/certify/cifar10/resnet110/noise_0.12/test/sigma_0.12"), "Baseline"),
            Line(ApproximateAccuracy("data/certify/cifar10/resnet110/noise_0.12/test/dipole_sigma_0.12"), "Dipole"),
        ])

    plot_certified_accuracy(
        "analysis/plots/vary_method_cifar_100000_0.50", "CIFAR, vary method, $\sigma=0.50$, N=10^5", 2., [
            Line(ApproximateAccuracy("data/certify/cifar10/resnet110/noise_0.50/test/sigma_0.50"), "Baseline"),
            Line(ApproximateAccuracy("data/certify/cifar10/resnet110/noise_0.50/test/dipole_sigma_0.50"), "Dipole"),
        ])

    plot_certified_accuracy(
        "analysis/plots/vary_method_cifar_100000_1.00", "CIFAR, vary method, $\sigma=1.00$, N=10^5", 2., [
            Line(ApproximateAccuracy("data/certify/cifar10/resnet110/noise_1.00/test/sigma_1.00"), "Baseline"),
            Line(ApproximateAccuracy("data/certify/cifar10/resnet110/noise_1.00/test/dipole_sigma_1.00"), "Dipole"),
        ])
    plot_certified_accuracy(
        "analysis/plots/vary_method_cifar_1000000_0.25", "CIFAR, vary method, $\sigma=0.25$, N=10^6", 2., [
            Line(ApproximateAccuracy("data/certify/cifar10/resnet110/noise_0.25/test/sigma_0.25_N_1000000"), "Baseline"),
            Line(ApproximateAccuracy("data/certify/cifar10/resnet110/noise_0.25/test/dipole_sigma_0.25_N_1000000"), "Dipole"),
        ])

    plot_certified_accuracy(
        "analysis/plots/vary_method_cifar_1000000_0.12", "CIFAR, vary method, $\sigma=0.12$, N=10^6", 2., [
            Line(ApproximateAccuracy("data/certify/cifar10/resnet110/noise_0.12/test/sigma_0.12_N_1000000"), "Baseline"),
            Line(ApproximateAccuracy("data/certify/cifar10/resnet110/noise_0.12/test/dipole_sigma_0.12_N_1000000"), "Dipole"),
        ])

    plot_certified_accuracy(
        "analysis/plots/vary_method_cifar_1000000_0.50", "CIFAR, vary method, $\sigma=0.50$, N=10^6", 2., [
            Line(ApproximateAccuracy("data/certify/cifar10/resnet110/noise_0.50/test/sigma_0.50_N_1000000"), "Baseline"),
            Line(ApproximateAccuracy("data/certify/cifar10/resnet110/noise_0.50/test/dipole_sigma_0.50_N_1000000"), "Dipole"),
        ])

    plot_certified_accuracy(
        "analysis/plots/vary_method_cifar_1000000_1.00", "CIFAR, vary method, $\sigma=1.00$, N=10^6", 2., [
            Line(ApproximateAccuracy("data/certify/cifar10/resnet110/noise_1.00/test/sigma_1.00_N_1000000"), "Baseline"),
            Line(ApproximateAccuracy("data/certify/cifar10/resnet110/noise_1.00/test/dipole_sigma_1.00_N_1000000"), "Dipole"),
        ])

    plot_certified_accuracy(
        "analysis/plots/vary_method_imagenet_100000_0.25", "ImageNet, vary method, $\sigma=0.25$, N=10^5", 2., [
            Line(ApproximateAccuracy("data/certify/imagenet/resnet50/noise_0.25/test/sigma_0.25"), "Baseline"),
            Line(ApproximateAccuracy("data/certify/imagenet/resnet50/noise_0.25/test/dipole_sigma_0.25"), "Dipole"),
        ])


    plot_certified_accuracy(
        "analysis/plots/vary_method_imagenet_100000_0.50", "ImageNet, vary method, $\sigma=0.50$, N=10^5", 2., [
            Line(ApproximateAccuracy("data/certify/imagenet/resnet50/noise_0.50/test/sigma_0.50"), "Baseline"),
            Line(ApproximateAccuracy("data/certify/imagenet/resnet50/noise_0.50/test/dipole_sigma_0.50"), "Dipole"),
        ])

    plot_certified_accuracy(
        "analysis/plots/vary_method_imagenet_100000_1.00", "ImageNet, vary method, $\sigma=1.00$, N=10^5", 2., [
            Line(ApproximateAccuracy("data/certify/imagenet/resnet50/noise_1.00/test/sigma_1.00"), "Baseline"),
            Line(ApproximateAccuracy("data/certify/imagenet/resnet50/noise_1.00/test/dipole_sigma_1.00"), "Dipole"),
        ])

    plot_certified_accuracy(
        "analysis/plots/vary_method_small_mnist_100000_0.25", "Small MNIST, vary method, $\sigma=0.25$, N=10^5", 1.5, [
            Line(ApproximateAccuracy("data/certify/small_mnist_10pcs/small_mnist_pca_net/noise_0.25/test/second_order_N_100000"), "Second Order"),
            Line(ApproximateAccuracy("data/certify/small_mnist_10pcs/small_mnist_pca_net/noise_0.25/test/N_100000"), "Baseline"),
            Line(ApproximateAccuracy("data/certify/small_mnist_10pcs/small_mnist_pca_net/noise_0.25/test/dipole_N_100000"), "Dipole"),
        ])
    plot_certified_accuracy(
        "analysis/plots/vary_method_small_mnist_1000000_0.25", "Small MNIST, vary method, $\sigma=0.25$, N=10^6", 1.5, [
            Line(ApproximateAccuracy("data/certify/small_mnist_10pcs/small_mnist_pca_net/noise_0.25/test/second_order_N_1000000"), "Second Order"),
            Line(ApproximateAccuracy("data/certify/small_mnist_10pcs/small_mnist_pca_net/noise_0.25/test/N_1000000"), "Baseline"),
            Line(ApproximateAccuracy("data/certify/small_mnist_10pcs/small_mnist_pca_net/noise_0.25/test/dipole_N_1000000"), "Dipole"),
        ])
    plot_certified_accuracy(
        "analysis/plots/vary_method_small_mnist_10000000_0.25", "Small MNIST, vary method, $\sigma=0.25$, N=10^7", 1.5, [
            Line(ApproximateAccuracy("data/certify/small_mnist_10pcs/small_mnist_pca_net/noise_0.25/test/second_order_N_10000000"), "Second Order"),
            Line(ApproximateAccuracy("data/certify/small_mnist_10pcs/small_mnist_pca_net/noise_0.25/test/N_10000000"), "Baseline"),
            Line(ApproximateAccuracy("data/certify/small_mnist_10pcs/small_mnist_pca_net/noise_0.25/test/dipole_N_10000000"), "Dipole"),
        ])
    plot_certified_accuracy(
        "analysis/plots/vary_method_small_mnist_100000000_0.25", "Small MNIST, vary method, $\sigma=0.25$, N=10^8", 1.5, [
            Line(ApproximateAccuracy("data/certify/small_mnist_10pcs/small_mnist_pca_net/noise_0.25/test/second_order_N_100000000"), "Second Order"),
            Line(ApproximateAccuracy("data/certify/small_mnist_10pcs/small_mnist_pca_net/noise_0.25/test/N_100000000"), "Baseline"),
            Line(ApproximateAccuracy("data/certify/small_mnist_10pcs/small_mnist_pca_net/noise_0.25/test/dipole_N_100000000"), "Dipole"),
        ])
    plot_certified_accuracy(
        "analysis/plots/vary_method_mnist_100000_0.25", "MNIST, vary method, $\sigma=0.25$, N=10^5", 1.5, [
            Line(ApproximateAccuracy("data/certify/small_mnist/small_mnist_net/noise_0.25/test/second_order_N_100000"), "Second Order"),
            Line(ApproximateAccuracy("data/certify/small_mnist/small_mnist_net/noise_0.25/test/N_100000"), "Baseline"),
            Line(ApproximateAccuracy("data/certify/small_mnist/small_mnist_net/noise_0.25/test/dipole_N_100000"), "Dipole"),
        ])
    plot_certified_accuracy(
        "analysis/plots/vary_method_mnist_1000000_0.25", "MNIST, vary method, $\sigma=0.25$, N=10^6", 1.5, [
            Line(ApproximateAccuracy("data/certify/small_mnist/small_mnist_net/noise_0.25/test/second_order_N_1000000"), "Second Order"),
            Line(ApproximateAccuracy("data/certify/small_mnist/small_mnist_net/noise_0.25/test/N_1000000"), "Baseline"),
            Line(ApproximateAccuracy("data/certify/small_mnist/small_mnist_net/noise_0.25/test/dipole_N_1000000"), "Dipole"),
        ])
    plot_certified_accuracy(
        "analysis/plots/vary_method_mnist_10000000_0.25", "MNIST, vary method, $\sigma=0.25$, N=10^7", 1.5, [
            Line(ApproximateAccuracy("data/certify/small_mnist/small_mnist_net/noise_0.25/test/second_order_N_10000000"), "Second Order"),
            Line(ApproximateAccuracy("data/certify/small_mnist/small_mnist_net/noise_0.25/test/N_10000000"), "Baseline"),
            Line(ApproximateAccuracy("data/certify/small_mnist/small_mnist_net/noise_0.25/test/dipole_N_10000000"), "Dipole"),
        ])
    plot_certified_accuracy(
        "analysis/plots/vary_method_mnist_100000000_0.25", "MNIST, vary method, $\sigma=0.25$, N=10^8", 1.5, [
            Line(ApproximateAccuracy("data/certify/small_mnist/small_mnist_net/noise_0.25/test/second_order_N_100000000"), "Second Order"),
            Line(ApproximateAccuracy("data/certify/small_mnist/small_mnist_net/noise_0.25/test/N_100000000"), "Baseline"),
            Line(ApproximateAccuracy("data/certify/small_mnist/small_mnist_net/noise_0.25/test/dipole_N_100000000"), "Dipole"),
        ])


    plot_certified_accuracy(
        "analysis/plots/vary_method_mnist_adv_100000_0.25", "ADV MNIST, vary method, $\sigma=0.25$, N=10^5", 1.5, [
            Line(ApproximateAccuracy("data/certify/small_mnist_adv/small_mnist_net/noise_0.25/test/second_order_N_100000"), "Second Order"),
            Line(ApproximateAccuracy("data/certify/small_mnist_adv/small_mnist_net/noise_0.25/test/N_100000"), "Baseline"),
            Line(ApproximateAccuracy("data/certify/small_mnist_adv/small_mnist_net/noise_0.25/test/dipole_N_100000"), "Dipole"),
        ])
    plot_certified_accuracy(
        "analysis/plots/vary_method_mnist_adv_1000000_0.25", "ADV MNIST, vary method, $\sigma=0.25$, N=10^6", 1.5, [
            Line(ApproximateAccuracy("data/certify/small_mnist_adv/small_mnist_net/noise_0.25/test/second_order_N_1000000"), "Second Order"),
            Line(ApproximateAccuracy("data/certify/small_mnist_adv/small_mnist_net/noise_0.25/test/N_1000000"), "Baseline"),
            Line(ApproximateAccuracy("data/certify/small_mnist_adv/small_mnist_net/noise_0.25/test/dipole_N_1000000"), "Dipole"),
        ])
    plot_certified_accuracy(
        "analysis/plots/vary_method_mnist_adv_10000000_0.25", "ADV MNIST, vary method, $\sigma=0.25$, N=10^7", 1.5, [
            Line(ApproximateAccuracy("data/certify/small_mnist_adv/small_mnist_net/noise_0.25/test/second_order_N_10000000"), "Second Order"),
            Line(ApproximateAccuracy("data/certify/small_mnist_adv/small_mnist_net/noise_0.25/test/N_10000000"), "Baseline"),
            Line(ApproximateAccuracy("data/certify/small_mnist_adv/small_mnist_net/noise_0.25/test/dipole_N_10000000"), "Dipole"),
        ])
    # plot_certified_accuracy(
    #     "analysis/plots/vary_method_mnist_adv_100000000_0.25", "ADV MNIST, vary method, $\sigma=0.25$, N=10^8", 1.5, [
    #         Line(ApproximateAccuracy("data/certify/small_mnist_adv/small_mnist_net/noise_0.25/test/second_order_N_100000000"), "Second Order"),
    #         Line(ApproximateAccuracy("data/certify/small_mnist_adv/small_mnist_net/noise_0.25/test/N_100000000"), "Baseline"),
    #         Line(ApproximateAccuracy("data/certify/small_mnist_adv/small_mnist_net/noise_0.25/test/dipole_N_100000000"), "Dipole"),
    #     ])



    plot_certified_accuracy(
        "analysis/plots/vary_method_small_mnist_100000_0.50", "Small MNIST, vary method, $\sigma=0.50$, N=10^5", 1.5, [
            Line(ApproximateAccuracy("data/certify/small_mnist_10pcs/small_mnist_pca_net/noise_0.50/test/second_order_N_100000"), "Second Order"),
            Line(ApproximateAccuracy("data/certify/small_mnist_10pcs/small_mnist_pca_net/noise_0.50/test/N_100000"), "Baseline"),
            Line(ApproximateAccuracy("data/certify/small_mnist_10pcs/small_mnist_pca_net/noise_0.50/test/dipole_N_100000"), "Dipole"),
        ])
    plot_certified_accuracy(
        "analysis/plots/vary_method_small_mnist_1000000_0.50", "Small MNIST, vary method, $\sigma=0.50$, N=10^6", 1.5, [
            Line(ApproximateAccuracy("data/certify/small_mnist_10pcs/small_mnist_pca_net/noise_0.50/test/second_order_N_1000000"), "Second Order"),
            Line(ApproximateAccuracy("data/certify/small_mnist_10pcs/small_mnist_pca_net/noise_0.50/test/N_1000000"), "Baseline"),
            Line(ApproximateAccuracy("data/certify/small_mnist_10pcs/small_mnist_pca_net/noise_0.50/test/dipole_N_1000000"), "Dipole"),
        ])
    plot_certified_accuracy(
        "analysis/plots/vary_method_small_mnist_10000000_0.50", "Small MNIST, vary method, $\sigma=0.50$, N=10^7", 1.5, [
            Line(ApproximateAccuracy("data/certify/small_mnist_10pcs/small_mnist_pca_net/noise_0.50/test/second_order_N_10000000"), "Second Order"),
            Line(ApproximateAccuracy("data/certify/small_mnist_10pcs/small_mnist_pca_net/noise_0.50/test/N_10000000"), "Baseline"),
            Line(ApproximateAccuracy("data/certify/small_mnist_10pcs/small_mnist_pca_net/noise_0.50/test/dipole_N_10000000"), "Dipole"),
        ])
    plot_certified_accuracy(
        "analysis/plots/vary_method_small_mnist_100000000_0.50", "Small MNIST, vary method, $\sigma=0.50$, N=10^8", 1.5, [
            Line(ApproximateAccuracy("data/certify/small_mnist_10pcs/small_mnist_pca_net/noise_0.50/test/second_order_N_100000000"), "Second Order"),
            Line(ApproximateAccuracy("data/certify/small_mnist_10pcs/small_mnist_pca_net/noise_0.50/test/N_100000000"), "Baseline"),
            Line(ApproximateAccuracy("data/certify/small_mnist_10pcs/small_mnist_pca_net/noise_0.50/test/dipole_N_100000000"), "Dipole"),
        ])
    plot_certified_accuracy(
        "analysis/plots/vary_method_mnist_100000_0.50", "MNIST, vary method, $\sigma=0.50$, N=10^5", 1.5, [
            Line(ApproximateAccuracy("data/certify/small_mnist/small_mnist_net/noise_0.50/test/second_order_N_100000"), "Second Order"),
            Line(ApproximateAccuracy("data/certify/small_mnist/small_mnist_net/noise_0.50/test/N_100000"), "Baseline"),
            Line(ApproximateAccuracy("data/certify/small_mnist/small_mnist_net/noise_0.50/test/dipole_N_100000"), "Dipole"),
        ])
    plot_certified_accuracy(
        "analysis/plots/vary_method_mnist_1000000_0.50", "MNIST, vary method, $\sigma=0.50$, N=10^6", 1.5, [
            Line(ApproximateAccuracy("data/certify/small_mnist/small_mnist_net/noise_0.50/test/second_order_N_1000000"), "Second Order"),
            Line(ApproximateAccuracy("data/certify/small_mnist/small_mnist_net/noise_0.50/test/N_1000000"), "Baseline"),
            Line(ApproximateAccuracy("data/certify/small_mnist/small_mnist_net/noise_0.50/test/dipole_N_1000000"), "Dipole"),
        ])
    plot_certified_accuracy(
        "analysis/plots/vary_method_mnist_10000000_0.50", "MNIST, vary method, $\sigma=0.50$, N=10^7", 1.5, [
            Line(ApproximateAccuracy("data/certify/small_mnist/small_mnist_net/noise_0.50/test/second_order_N_10000000"), "Second Order"),
            Line(ApproximateAccuracy("data/certify/small_mnist/small_mnist_net/noise_0.50/test/N_10000000"), "Baseline"),
            Line(ApproximateAccuracy("data/certify/small_mnist/small_mnist_net/noise_0.50/test/dipole_N_10000000"), "Dipole"),
        ])
    plot_certified_accuracy(
        "analysis/plots/vary_method_mnist_100000000_0.50", "MNIST, vary method, $\sigma=0.50$, N=10^8", 1.5, [
            Line(ApproximateAccuracy("data/certify/small_mnist/small_mnist_net/noise_0.50/test/second_order_N_100000000"), "Second Order"),
            Line(ApproximateAccuracy("data/certify/small_mnist/small_mnist_net/noise_0.50/test/N_100000000"), "Baseline"),
            Line(ApproximateAccuracy("data/certify/small_mnist/small_mnist_net/noise_0.50/test/dipole_N_100000000"), "Dipole"),
        ])

    plot_certified_accuracy(
        "analysis/plots/vary_method_small_mnist_100000_0.12", "Small MNIST, vary method, $\sigma=0.12$, N=10^5", 1.5, [
            Line(ApproximateAccuracy("data/certify/small_mnist_10pcs/small_mnist_pca_net/noise_0.12/test/second_order_N_100000"), "Second Order"),
            Line(ApproximateAccuracy("data/certify/small_mnist_10pcs/small_mnist_pca_net/noise_0.12/test/N_100000"), "Baseline"),
            Line(ApproximateAccuracy("data/certify/small_mnist_10pcs/small_mnist_pca_net/noise_0.12/test/dipole_N_100000"), "Dipole"),
        ])
    plot_certified_accuracy(
        "analysis/plots/vary_method_small_mnist_1000000_0.12", "Small MNIST, vary method, $\sigma=0.12$, N=10^6", 1.5, [
            Line(ApproximateAccuracy("data/certify/small_mnist_10pcs/small_mnist_pca_net/noise_0.12/test/second_order_N_1000000"), "Second Order"),
            Line(ApproximateAccuracy("data/certify/small_mnist_10pcs/small_mnist_pca_net/noise_0.12/test/N_1000000"), "Baseline"),
            Line(ApproximateAccuracy("data/certify/small_mnist_10pcs/small_mnist_pca_net/noise_0.12/test/dipole_N_1000000"), "Dipole"),
        ])
    plot_certified_accuracy(
        "analysis/plots/vary_method_small_mnist_10000000_0.12", "Small MNIST, vary method, $\sigma=0.12$, N=10^7", 1.5, [
            Line(ApproximateAccuracy("data/certify/small_mnist_10pcs/small_mnist_pca_net/noise_0.12/test/second_order_N_10000000"), "Second Order"),
            Line(ApproximateAccuracy("data/certify/small_mnist_10pcs/small_mnist_pca_net/noise_0.12/test/N_10000000"), "Baseline"),
            Line(ApproximateAccuracy("data/certify/small_mnist_10pcs/small_mnist_pca_net/noise_0.12/test/dipole_N_10000000"), "Dipole"),
        ])
    plot_certified_accuracy(
        "analysis/plots/vary_method_small_mnist_100000000_0.12", "Small MNIST, vary method, $\sigma=0.12$, N=10^8", 1.5, [
            Line(ApproximateAccuracy("data/certify/small_mnist_10pcs/small_mnist_pca_net/noise_0.12/test/second_order_N_100000000"), "Second Order"),
            Line(ApproximateAccuracy("data/certify/small_mnist_10pcs/small_mnist_pca_net/noise_0.12/test/N_100000000"), "Baseline"),
            Line(ApproximateAccuracy("data/certify/small_mnist_10pcs/small_mnist_pca_net/noise_0.12/test/dipole_N_100000000"), "Dipole"),
        ])
    plot_certified_accuracy(
        "analysis/plots/vary_method_mnist_100000_0.12", "MNIST, vary method, $\sigma=0.12$, N=10^5", 1.5, [
            Line(ApproximateAccuracy("data/certify/small_mnist/small_mnist_net/noise_0.12/test/second_order_N_100000"), "Second Order"),
            Line(ApproximateAccuracy("data/certify/small_mnist/small_mnist_net/noise_0.12/test/N_100000"), "Baseline"),
            Line(ApproximateAccuracy("data/certify/small_mnist/small_mnist_net/noise_0.12/test/dipole_N_100000"), "Dipole"),
        ])
    plot_certified_accuracy(
        "analysis/plots/vary_method_mnist_1000000_0.12", "MNIST, vary method, $\sigma=0.12$, N=10^6", 1.5, [
            Line(ApproximateAccuracy("data/certify/small_mnist/small_mnist_net/noise_0.12/test/second_order_N_1000000"), "Second Order"),
            Line(ApproximateAccuracy("data/certify/small_mnist/small_mnist_net/noise_0.12/test/N_1000000"), "Baseline"),
            Line(ApproximateAccuracy("data/certify/small_mnist/small_mnist_net/noise_0.12/test/dipole_N_1000000"), "Dipole"),
        ])
    plot_certified_accuracy(
        "analysis/plots/vary_method_mnist_10000000_0.12", "MNIST, vary method, $\sigma=0.12$, N=10^7", 1.5, [
            Line(ApproximateAccuracy("data/certify/small_mnist/small_mnist_net/noise_0.12/test/second_order_N_10000000"), "Second Order"),
            Line(ApproximateAccuracy("data/certify/small_mnist/small_mnist_net/noise_0.12/test/N_10000000"), "Baseline"),
            Line(ApproximateAccuracy("data/certify/small_mnist/small_mnist_net/noise_0.12/test/dipole_N_10000000"), "Dipole"),
        ])
    plot_certified_accuracy(
        "analysis/plots/vary_method_mnist_100000000_0.12", "MNIST, vary method, $\sigma=0.12$, N=10^8", 1.5, [
            Line(ApproximateAccuracy("data/certify/small_mnist/small_mnist_net/noise_0.12/test/second_order_N_100000000"), "Second Order"),
            Line(ApproximateAccuracy("data/certify/small_mnist/small_mnist_net/noise_0.12/test/N_100000000"), "Baseline"),
            Line(ApproximateAccuracy("data/certify/small_mnist/small_mnist_net/noise_0.12/test/dipole_N_100000000"), "Dipole"),
        ])
    # plot_certified_accuracy(
    #     "analysis/plots/vary_train_noise_imagenet_050", "ImageNet, vary train noise, $\sigma=0.5$", 1.5, [
    #         Line(ApproximateAccuracy("data/certify/imagenet/resnet50/noise_0.25/test/sigma_0.50"), "train $\sigma = 0.25$"),
    #         Line(ApproximateAccuracy("data/certify/imagenet/resnet50/noise_0.50/test/sigma_0.50"), "train $\sigma = 0.50$"),
    #         Line(ApproximateAccuracy("data/certify/imagenet/resnet50/noise_1.00/test/sigma_0.50"), "train $\sigma = 1.00$"),
    #     ])
    # plot_certified_accuracy(
    #     "analysis/plots/vary_noise_imagenet", "ImageNet, vary $\sigma$", 4, [
    #         Line(ApproximateAccuracy("data/certify/imagenet/resnet50/noise_0.25/test/sigma_0.25"), "$\sigma = 0.25$"),
    #         Line(ApproximateAccuracy("data/certify/imagenet/resnet50/noise_0.50/test/sigma_0.50"), "$\sigma = 0.50$"),
    #         Line(ApproximateAccuracy("data/certify/imagenet/resnet50/noise_1.00/test/sigma_1.00"), "$\sigma = 1.00$"),
    #     ])
    # plot_certified_accuracy(
    #     "analysis/plots/high_prob", "Approximate vs. High-Probability", 2.0, [
    #         Line(ApproximateAccuracy("data/certify/imagenet/resnet50/noise_0.50/test/sigma_0.50"), "Approximate"),
    #         Line(HighProbAccuracy("data/certify/imagenet/resnet50/noise_0.50/test/sigma_0.50", 0.001, 0.001), "High-Prob"),
    #     ])
