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

from src.data.cifar.cifar10 import get_data_loaders
from src.models.resnet.resnet import ResNet18
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
@click.option("--seeds", default=5, help="Number of seeds")
@click.option("--batch_size", default=128, help="Batch size for training.")
def train_model(
    learning_rate, num_epochs, weight_decay, momentum, seeds, batch_size
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    current_date = get_current_datetime_for_path()
    for seed in np.arange(seeds):
        save_dir = "./logs/CIFAR10/{}/{}/{}".format(
            "no_prune",
            "seed_" + str(int(seed)),
            current_date,
        )
        os.makedirs(save_dir, exist_ok=True)
        logger = setup_logger(save_dir)
        logger.info("save_dir : {}".format(save_dir))
        logger.info("the model will be NOT pruned")

        torch_fix_seed(seed)
        resnet = ResNet18().to(device)

        train_loader, test_loader = get_data_loaders(batch_size)
        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(
            resnet.parameters(),
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
            resnet.train()
            epoch_losses = []
            for i, (images, labels) in enumerate(train_loader):
                images = images.to(device)
                labels = labels.to(device)

                # Forward pass
                outputs = resnet(images)
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
            resnet.eval()  # 評価モードに設定
            with torch.no_grad():
                correct = 0
                total = 0
                for images, labels in test_loader:
                    images = images.to(device)
                    labels = labels.to(device)
                    outputs = resnet(images)
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

        df = pd.DataFrame(
            {
                "Epoch": range(1, num_epochs + 1),
                "Loss": losses,
                "Validation Accuracy": val_accuracies,
            }
        )
        torch.save(resnet.state_dict(), os.path.join(save_dir, "model_state.pkl"))
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
