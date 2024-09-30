import torch
import torchvision
import torchvision.transforms as transforms


def get_stl10_data_loaders(batch_size):
    """
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(96, padding=4),
        transforms.ToTensor(),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])
    """
    transform_train = transforms.Compose(
        [
            transforms.Resize((32, 32)),
            transforms.Pad(4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32),
            transforms.ToTensor(),
        ]
    )

    transform_test = transforms.Compose(
        [
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
        ]
    )

    # STL10データセット
    train_dataset = torchvision.datasets.STL10(
        root="./datasets",
        split="train",
        transform=transform_train,
        download=True,
    )

    test_dataset = torchvision.datasets.STL10(
        root="./datasets",
        split="test",
        transform=transform_test,
    )

    # データローダー
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    return train_loader, test_loader
