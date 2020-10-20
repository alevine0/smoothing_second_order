from torchvision import transforms, datasets
import torch
import sklearn.decomposition
small_mnist_dataset = datasets.MNIST("./dataset_cache", train=True, download=True, transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x : x.reshape([1,28,7,4]).mean(dim=3).reshape([1,7,4,7]).mean(dim=2).reshape(49))
]))

small_mnist_np = torch.stack(list([a[0] for a in small_mnist_dataset])).numpy()
pca = sklearn.decomposition.PCA(n_components=10)
pca.fit(small_mnist_np)
torch.save(pca, 'small_mnist_10_pcs.pth')

