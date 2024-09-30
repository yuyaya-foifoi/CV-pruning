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
@click.option("--dataset_name", default="CIFAR10", help="name of dataset")
@click.option("--n_class", default=10, help="number of cls")
@click.option("--dir_name", help="name of dir")
def train_model(
    learning_rate,
    num_epochs,
    weight_decay,
    momentum,
    remain_rate,
    seeds,
    batch_size,
    dataset_name,
    n_class,
    dir_name,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    current_date = get_current_datetime_for_path()
    for seed in np.arange(seeds):
        save_dir = "./logs/CIFAR10/is_prune/{}/{}/{}/{}".format(
            "ensemble_output_diff_score",
            dir_name,
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
        resnet = ResNet18(n_cls=n_class).to(device)
        init_scores_modes = [
            "kaiming_uniform",
            "kaiming_uniform_wide",
            "kaiming_uniform_narrow",
            #"kaiming_uniform",
            "kaiming_normal",
            "xavier_uniform",
            "xavier_normal",
            "uniform",
            "normal",
        ]
        resnet_slths = [
            modify_module_for_slth(
                resnet, remain_rate, init_scores_mode=mode
            ).to(device)
            for mode in init_scores_modes
        ]
        resnet_slth_init = copy.deepcopy(resnet_slths[0]).to(device)

        train_loader, test_loader = get_data_loaders(
            dataset_name=dataset_name, batch_size=batch_size
        )
        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizers = [
            optim.SGD(
                model.parameters(),
                lr=learning_rate,
                momentum=momentum,
                weight_decay=weight_decay,
            )
            for model in resnet_slths
        ]
        schedulers = [
            CosineAnnealingLR(optimizer, T_max=num_epochs)
            for optimizer in optimizers
        ]

        # Train the model
        losses = []
        val_accuracies = []

        total_step = len(train_loader)
        for epoch in range(num_epochs):
            for resnet_slth in resnet_slths:
                resnet_slth.train()

            epoch_losses = []
            for i, (images, labels) in enumerate(train_loader):
                images = images.to(device)
                labels = labels.to(device)

                for resnet_slth, optimizer in zip(resnet_slths, optimizers):
                    outputs = resnet_slth(images)
                    loss = criterion(outputs, labels)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    epoch_losses.append(loss.item())

                total_loss = sum(epoch_losses[-len(resnet_slths) :])

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

            for scheduler in schedulers:
                scheduler.step()

            epoch_loss = sum(epoch_losses) / len(epoch_losses)
            losses.append(epoch_loss)

            logger.info(
                "Epoch [{}/{}], Current learning rate: {:.5f}".format(
                    epoch + 1, num_epochs, schedulers[0].get_last_lr()[0]
                )
            )

            for resnet_slth in resnet_slths:
                resnet_slth.eval()

            with torch.no_grad():
                correct = 0
                total = 0
                for images, labels in test_loader:
                    images = images.to(device)
                    labels = labels.to(device)
                    outputs_list = [model(images) for model in resnet_slths]
                    ensemble_outputs = torch.stack(outputs_list).mean(dim=0)

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
            if num_epochs > 0:  # 少なくとも1エポック学習した場合のみチェック
                for resnet_slth in resnet_slths:
                    for name, param in resnet_slth.named_parameters():
                        if "weight" in name:
                            init_param = resnet_slth_init.state_dict()[name]
                            assert torch.equal(
                                param.data, init_param
                            ), f"Weight mismatch found in {name} after training"

        df = pd.DataFrame(
            {
                "Epoch": range(1, num_epochs + 1),
                "Loss": losses,
                "Validation Accuracy": val_accuracies,
            }
        )

        df.to_csv(os.path.join(save_dir, "training_results.csv"), index=False)
        for i, resnet_slth in enumerate(resnet_slths):
            torch.save(
                resnet_slth.state_dict(),
                os.path.join(save_dir, f"resnet_slth{i+1}_state.pkl"),
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
