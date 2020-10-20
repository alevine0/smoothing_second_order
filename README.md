Code for the paper "Tight Second-Order Certificates for Randomized Smoothing" by Alexander Levine, Aounon Kumar, Tom Goldstein, and Soheil Feizi. Forked from (Cohen et al.) code at https://github.com/locuslab/smoothing. Functions almost exactly the same as that repository, except there is a new '--method' flag on certify.py which takes 'baseline', 'second_order', 'dipole' for new certification methods.

ImageNet and Cifar experiments are using checkpoints provided by Cohen et al; see download links at https://github.com/locuslab/smoothing

checkpoints for MNIST (7*7 and 10 PCs) are provided. They are trained using default options, except batch_size=400. The classifier architectures (simple CNN and 3-layer network, respectively) are small_mnist_classifier and small_mnist_pca_classifier.