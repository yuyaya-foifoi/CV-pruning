import torch
import torchvision
import torchvision.transforms as transforms


def get_svhn_data_loaders(batch_size):
    transform_train = transforms.Compose(
        [
            transforms.Pad(4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32),
            transforms.ToTensor(),
        ]
    )

    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )

    # SVHN dataset
    train_dataset = torchvision.datasets.SVHN(
        root="./datasets",
        split="train",  # 'train' for the training data
        transform=transform_train,
        download=True,
    )

    test_dataset = torchvision.datasets.SVHN(
        root="./datasets",
        split="test",  # 'test' for the test data
        transform=transform_test,
        download=True,
    )

    # Data loaders
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True
    )

    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset, batch_size=batch_size, shuffle=False
    )

    return train_loader, test_loader
