import torch
import torchvision
import torchvision.transforms as transforms


def get_fashionmnist_data_loaders(batch_size):
    # FashionMNISTの寸法と特徴に合わせて変換を調整
    transform_train = transforms.Compose(
        [
            transforms.Pad(2),  # 画像サイズに合わせてパディングを変更
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(28),  # FashionMNIST用にクロップサイズを調整
            transforms.ToTensor(),
        ]
    )

    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )

    # FashionMNISTデータセット
    train_dataset = torchvision.datasets.FashionMNIST(
        root="./datasets",
        train=True,
        transform=transform_train,
        download=True,
    )

    test_dataset = torchvision.datasets.FashionMNIST(
        root="./datasets", train=False, transform=transform_test
    )

    # データローダー
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True
    )

    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset, batch_size=batch_size, shuffle=False
    )

    return train_loader, test_loader
