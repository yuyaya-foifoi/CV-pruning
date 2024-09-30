from src.data.CelebA.celeba import get_celeba_data_loaders
from src.data.cifar.cifar10 import get_cifar10_data_loaders
from src.data.cifar.cifar100 import get_cifar100_data_loaders
from src.data.fashionMNIST.fashion_mnist import get_fashionmnist_data_loaders
from src.data.stl10.stl10 import get_stl10_data_loaders
from src.data.SVHN.SVHN import get_svhn_data_loaders


def get_data_loaders(batch_size: int, dataset_name: str):
    if dataset_name == "CIFAR10":
        train_loader, test_loader = get_cifar10_data_loaders(batch_size)
    elif dataset_name == "CIFAR100":
        train_loader, test_loader = get_cifar100_data_loaders(batch_size)
    elif dataset_name == "CelebA":
        train_loader, test_loader = get_celeba_data_loaders(batch_size)
    elif dataset_name == "fashionMNIST":
        train_loader, test_loader = get_fashionmnist_data_loaders(batch_size)
    elif dataset_name == "STL10":
        train_loader, test_loader = get_stl10_data_loaders(batch_size)
    elif dataset_name == "SVHN":
        train_loader, test_loader = get_svhn_data_loaders(batch_size)
    return train_loader, test_loader
