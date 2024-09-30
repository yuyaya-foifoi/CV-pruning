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
from src.pruning.slth.edgepopup_ensemble_output import modify_module_for_slth
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
def train_model(
    learning_rate,
    num_epochs,
    weight_decay,
    momentum,
    remain_rate,
    seeds,
    batch_size,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    current_date = get_current_datetime_for_path()
    for seed in np.arange(seeds):
        save_dir = "./logs/CIFAR10/is_prune/{}/{}/{}".format(
            "ensemble_output",
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
        resnet = ResNet18().to(device)
        resnet_slth1 = modify_module_for_slth(
            resnet, remain_rate, init_scores_mode="kaiming_uniform"
        ).to(device)
        resnet_slth2 = modify_module_for_slth(
            resnet, remain_rate, init_scores_mode="kaiming_normal"
        ).to(device)
        resnet_slth3 = modify_module_for_slth(
            resnet, remain_rate, init_scores_mode="xavier_uniform"
        ).to(device)
        resnet_slth4 = modify_module_for_slth(
            resnet, remain_rate, init_scores_mode="xavier_normal"
        ).to(device)
        resnet_slth5 = modify_module_for_slth(
            resnet, remain_rate, init_scores_mode="uniform"
        ).to(device)
        resnet_slth6 = modify_module_for_slth(
            resnet, remain_rate, init_scores_mode="normal"
        ).to(device)
        resnet_slth_init = copy.deepcopy(resnet_slth1).to(device)

        train_loader, test_loader = get_data_loaders(batch_size)
        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer1 = optim.SGD(
            resnet_slth1.parameters(),
            lr=learning_rate,
            momentum=momentum,
            weight_decay=weight_decay,
        )
        optimizer2 = optim.SGD(
            resnet_slth2.parameters(),
            lr=learning_rate,
            momentum=momentum,
            weight_decay=weight_decay,
        )
        optimizer3 = optim.SGD(
            resnet_slth3.parameters(),
            lr=learning_rate,
            momentum=momentum,
            weight_decay=weight_decay,
        )
        optimizer4 = optim.SGD(
            resnet_slth4.parameters(),
            lr=learning_rate,
            momentum=momentum,
            weight_decay=weight_decay,
        )
        optimizer5 = optim.SGD(
            resnet_slth5.parameters(),
            lr=learning_rate,
            momentum=momentum,
            weight_decay=weight_decay,
        )
        optimizer6 = optim.SGD(
            resnet_slth6.parameters(),
            lr=learning_rate,
            momentum=momentum,
            weight_decay=weight_decay,
        )

        # 学習率のスケジューラ（コサインアニーリング）
        scheduler1 = CosineAnnealingLR(optimizer1, T_max=num_epochs)
        scheduler2 = CosineAnnealingLR(optimizer2, T_max=num_epochs)
        scheduler3 = CosineAnnealingLR(optimizer3, T_max=num_epochs)
        scheduler4 = CosineAnnealingLR(optimizer4, T_max=num_epochs)
        scheduler5 = CosineAnnealingLR(optimizer5, T_max=num_epochs)
        scheduler6 = CosineAnnealingLR(optimizer6, T_max=num_epochs)

        # Train the model
        losses = []
        val_accuracies = []

        total_step = len(train_loader)
        for epoch in range(num_epochs):
            resnet_slth1.train()
            resnet_slth2.train()
            resnet_slth3.train()
            resnet_slth4.train()
            resnet_slth5.train()
            resnet_slth6.train()

            epoch_losses = []
            for i, (images, labels) in enumerate(train_loader):
                images = images.to(device)
                labels = labels.to(device)

                # Forward pass
                outputs1 = resnet_slth1(images)
                outputs2 = resnet_slth2(images)
                outputs3 = resnet_slth3(images)
                outputs4 = resnet_slth4(images)
                outputs5 = resnet_slth5(images)
                outputs6 = resnet_slth6(images)

                loss1 = criterion(outputs1, labels)
                loss2 = criterion(outputs2, labels)
                loss3 = criterion(outputs3, labels)
                loss4 = criterion(outputs4, labels)
                loss5 = criterion(outputs5, labels)
                loss6 = criterion(outputs6, labels)

                # Backward and optimize
                optimizer1.zero_grad()
                loss1.backward()
                optimizer1.step()

                optimizer2.zero_grad()
                loss2.backward()
                optimizer2.step()

                optimizer3.zero_grad()
                loss3.backward()
                optimizer3.step()

                optimizer4.zero_grad()
                loss4.backward()
                optimizer4.step()

                optimizer5.zero_grad()
                loss5.backward()
                optimizer5.step()

                optimizer6.zero_grad()
                loss6.backward()
                optimizer6.step()

                total_loss = (
                    loss1.item()
                    + loss2.item()
                    + loss3.item()
                    + loss4.item()
                    + loss5.item()
                    + loss6.item()
                )
                epoch_losses.append(total_loss)

                if (i + 1) % 100 == 0:
                    logger.info(
                        "Epoch [{}/{}], Step [{}/{}] Loss: {:.4f}".format(
                            epoch + 1,
                            num_epochs,
                            i + 1,
                            total_step,
                            total_loss,
                        )
                    )

            # 学習率の更新
            scheduler1.step()
            scheduler2.step()
            scheduler3.step()
            scheduler4.step()
            scheduler5.step()
            scheduler6.step()

            epoch_loss = sum(epoch_losses) / len(epoch_losses)
            losses.append(epoch_loss)

            # Optional: 学習率のログ出力
            logger.info(
                "Epoch [{}/{}], Current learning rate: {:.5f}".format(
                    epoch + 1, num_epochs, scheduler1.get_last_lr()[0]
                )
            )
            # エポックごとにテストデータでモデルを評価
            resnet_slth1.eval()  # 評価モードに設定
            resnet_slth2.eval()
            resnet_slth3.eval()
            resnet_slth4.eval()
            resnet_slth5.eval()
            resnet_slth6.eval()

            with torch.no_grad():
                correct = 0
                total = 0
                for images, labels in test_loader:
                    images = images.to(device)
                    labels = labels.to(device)
                    outputs1 = resnet_slth1(images)
                    outputs2 = resnet_slth2(images)
                    outputs3 = resnet_slth3(images)
                    outputs4 = resnet_slth4(images)
                    outputs5 = resnet_slth5(images)
                    outputs6 = resnet_slth6(images)

                    ensemble_outputs = (
                        outputs1
                        + outputs2
                        + outputs3
                        + outputs4
                        + outputs5
                        + outputs6
                    ) / 6
                    _, predicted = torch.max(ensemble_outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

                acc = 100 * correct / total
                val_accuracies.append(acc)
                logger.info(
                    "Epoch [{}/{}], Accuracy of the model on the test images: {:.2f} %".format(
                        epoch + 1, num_epochs, acc
                    )
                )
            for resnet_slth in [
                resnet_slth1,
                resnet_slth2,
                resnet_slth3,
                resnet_slth4,
                resnet_slth5,
                resnet_slth6,
            ]:
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

        df.to_csv(os.path.join(save_dir, "training_results.csv"), index=False)
        torch.save(
            resnet_slth1.state_dict(),
            os.path.join(save_dir, "resnet_slth1_state.pkl"),
        )
        torch.save(
            resnet_slth2.state_dict(),
            os.path.join(save_dir, "resnet_slth2_state.pkl"),
        )
        torch.save(
            resnet_slth3.state_dict(),
            os.path.join(save_dir, "resnet_slth3_state.pkl"),
        )
        torch.save(
            resnet_slth4.state_dict(),
            os.path.join(save_dir, "resnet_slth4_state.pkl"),
        )
        torch.save(
            resnet_slth5.state_dict(),
            os.path.join(save_dir, "resnet_slth5_state.pkl"),
        )
        torch.save(
            resnet_slth6.state_dict(),
            os.path.join(save_dir, "resnet_slth6_state.pkl"),
        )

        send_email(
            os.environ.get("SENDER_ADDRESS"),
            os.environ.get("RECEIVER_ADDRESS"),
            os.environ.get("PASS"),
            os.path.join(save_dir, "training_results.csv"),
            logger,
        )


if __name__ == "__main__":
    train_model()
