import copy
import os

import click
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from dotenv import load_dotenv
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.data import get_data_loaders
from src.models.resnet.resnet import ResNet18
from src.pruning.slth.edgepopup import modify_module_for_slth
from src.utils.date import get_current_datetime_for_path
from src.utils.email import send_email
from src.utils.logger import setup_logger
from src.utils.seed import torch_fix_seed

load_dotenv()


@click.command()
@click.option("--learning_rate", default=0.1, help="Initial learning rate.")
@click.option("--num_epochs", default=100, help="Number of epochs to train.")
@click.option(
    "--weight_decay", default=0.0001, help="Weight decay (L2 penalty)."
)
@click.option("--momentum", default=0.9, help="Momentum.")
@click.option("--remain_rate", default=0.3, help="remain_rate")
@click.option("--seeds", default=5, help="Number of seeds")
@click.option("--batch_size", default=128, help="Batch size for training.")
@click.option("--dataset_name", default="CIFAR10", help="name of dataset")
@click.option("--source_path", help="trained_model")

def train_model(
    learning_rate,
    num_epochs,
    weight_decay,
    momentum,
    seeds,
    remain_rate,
    batch_size,
    dataset_name,
    source_path
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    current_date = get_current_datetime_for_path()
    for seed in np.arange(seeds):
        save_dir = "./logs/{}/is_transfer/{}/{}/{}".format(
            dataset_name,
            "remain_rate_" + str(int(remain_rate * 100)),
            "seed_" + str(int(seed)),
            current_date,
        )
        os.makedirs(save_dir, exist_ok=True)
        logger = setup_logger(save_dir)
        logger.info("save_dir : {}".format(save_dir))
        logger.info(
            "the model will be pruned and remain rate is {}".format(
                str(remain_rate * 100) + "%"
            )
        )

        torch_fix_seed(seed)
        resnet = ResNet18(100).to(device)
        resnet_slth = modify_module_for_slth(
            resnet, remain_rate=remain_rate
        ).to(device)

        source_weight = torch.load(source_path.format(str(seed)))
        source_weight["fc.weight"] = resnet_slth.state_dict()["fc.weight"]
        source_weight["fc.scores"] = resnet_slth.state_dict()["fc.scores"]

        resnet_slth.load_state_dict(source_weight)
        resnet_slth_init = copy.deepcopy(resnet_slth).to(device)

        train_loader, test_loader = get_data_loaders(dataset_name=dataset_name, batch_size=batch_size)
        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(
            resnet_slth.parameters(),
            lr=learning_rate,
            momentum=momentum,
            weight_decay=weight_decay,
        )

        # 学習率のスケジューラ（コサインアニーリング）
        scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)

        # Train the model
        losses = []
        val_accuracies = []

        total_step = len(train_loader)
        for epoch in range(num_epochs):
            resnet_slth.train()
            epoch_losses = []
            for i, (images, labels) in enumerate(train_loader):
                images = images.to(device)
                labels = labels.to(device)

                # Forward pass
                outputs = resnet_slth(images)
                loss = criterion(outputs, labels)

                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_losses.append(loss.item())

                if (i + 1) % 100 == 0:
                    logger.info(
                        "Epoch [{}/{}], Step [{}/{}] Loss: {:.4f}".format(
                            epoch + 1,
                            num_epochs,
                            i + 1,
                            total_step,
                            loss.item(),
                        )
                    )

            # 学習率の更新
            scheduler.step()
            epoch_loss = sum(epoch_losses) / len(epoch_losses)
            losses.append(epoch_loss)

            # Optional: 学習率のログ出力
            logger.info(
                "Epoch [{}/{}], Current learning rate: {:.5f}".format(
                    epoch + 1, num_epochs, scheduler.get_last_lr()[0]
                )
            )
            # エポックごとにテストデータでモデルを評価
            resnet_slth.eval()  # 評価モードに設定
            with torch.no_grad():
                correct = 0
                total = 0
                for images, labels in test_loader:
                    images = images.to(device)
                    labels = labels.to(device)
                    outputs = resnet_slth(images)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

                acc = 100 * correct / total
                val_accuracies.append(acc)
                logger.info(
                    "Epoch [{}/{}], Accuracy of the model on the test images: {:.2f} %".format(
                        epoch + 1, num_epochs, acc
                    )
                )

            for name, param in resnet_slth.named_parameters():
                if (
                    "weight" in name
                ):  # 'weight'を含む名前のパラメータのみチェック
                    # 初期状態のモデルから同じ名前のパラメータを取得
                    init_param = resnet_slth_init.state_dict()[name]
                    # 現在のパラメータと初期パラメータを比較
                    assert torch.equal(
                        param.data, init_param
                    ), f"Weight mismatch found in {name} after epoch {epoch+1}"

        df = pd.DataFrame(
            {
                "Epoch": range(1, num_epochs + 1),
                "Loss": losses,
                "Validation Accuracy": val_accuracies,
            }
        )
        torch.save(resnet_slth.state_dict(), os.path.join(save_dir, "resnet_slth_state.pkl"))
        df.to_csv(os.path.join(save_dir, "training_results.csv"), index=False)
        send_email(
            os.environ.get("SENDER_ADDRESS"),
            os.environ.get("RECEIVER_ADDRESS"),
            os.environ.get("PASS"),
            os.path.join(save_dir, "training_results.csv"),
            logger,
        )


if __name__ == "__main__":
    train_model()
