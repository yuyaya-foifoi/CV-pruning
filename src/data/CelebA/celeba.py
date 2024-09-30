import torch
import torchvision
import torchvision.transforms as transforms


def get_celeba_data_loaders(batch_size):
    # 訓練データ用のトランスフォーム
    transform_train = transforms.Compose(
        [
            transforms.CenterCrop(178),
            transforms.Resize(128),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    )

    # テストデータ用のトランスフォーム
    transform_test = transforms.Compose(
        [
            transforms.CenterCrop(178),
            transforms.Resize(128),
            transforms.ToTensor(),
        ]
    )

    # CelebA訓練データセット
    train_dataset = torchvision.datasets.CelebA(
        root="./datasets",
        split="train",
        transform=transform_train,
        download=True,
    )

    # CelebAテストデータセット
    test_dataset = torchvision.datasets.CelebA(
        root="./datasets",
        split="valid",  # CelebAの検証セットをテスト用として使用
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
