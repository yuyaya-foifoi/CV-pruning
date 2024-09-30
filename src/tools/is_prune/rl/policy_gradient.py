import copy
import os

import click
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributions as dist
from dotenv import load_dotenv
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.data import get_data_loaders
from src.models.resnet import get_resnet
from src.pruning.slth.edgepopup import modify_module_for_slth
from src.utils.date import get_current_datetime_for_path
from src.utils.email import send_email
from src.utils.logger import setup_logger
from src.utils.seed import torch_fix_seed

load_dotenv()

class RLAgent(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(RLAgent, self).__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.ln1 = nn.LayerNorm(256)
        self.fc2 = nn.Linear(256, 256)
        self.ln2 = nn.LayerNorm(256)
        self.fc3 = nn.Linear(256, action_dim)
        
    def forward(self, x):
        x = F.relu(self.ln1(self.fc1(x)))
        x = F.relu(self.ln2(self.fc2(x)))
        return self.fc3(x)

class PolicyGradientAgent:
    def __init__(self, state_dim, action_dim, lr=0.001, device='cuda', gamma=0.99):
        self.device = device
        self.policy = RLAgent(state_dim, action_dim).to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.gamma = gamma
        self.max_grad_norm = 1.0
        
    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        action_mean = self.policy(state)
        action_dist = dist.Normal(action_mean, torch.ones_like(action_mean))
        action = action_dist.sample()
        log_prob = action_dist.log_prob(action).sum(dim=-1)
        return action.squeeze().detach(), log_prob.squeeze().detach()

    def update(self, states, actions, rewards):
        # statesがすでにテンソルのリストであることを前提とする
        states = torch.stack(states).to(self.device)
        actions = torch.stack(actions).to(self.device)
        
        # 累積報酬和の計算
        returns = []
        R = 0
        for r in rewards[::-1]:
            R = r + self.gamma * R
            returns.insert(0, R)
        returns = torch.FloatTensor(returns).to(self.device)
        
        # 報酬の正規化
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        action_means = self.policy(states)
        action_dist = dist.Normal(action_means, torch.ones_like(action_means))
        log_probs = action_dist.log_prob(actions).sum(dim=-1)
        
        loss = -(log_probs * returns).mean()
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)  # 勾配クリッピング
        self.optimizer.step()

        return loss.item()

def is_target_layer(name):
    target_layers = ['fc.scores']
    return any(layer in name for layer in target_layers)

def get_all_scores(model):
    scores = []
    for name, param in model.named_parameters():
        if 'scores' in name and is_target_layer(name):
            scores.append(param.data.view(-1).cpu())
    return torch.cat(scores)

def update_scores(model, score_changes, learning_rate=0.01):
    idx = 0
    for name, param in model.named_parameters():
        if 'scores' in name and is_target_layer(name):
            num_params = param.numel()
            param.data += learning_rate * score_changes[idx:idx+num_params].to(param.device).view(param.shape)
            idx += num_params

def get_target_params(model):
    return sum(p.numel() for n, p in model.named_parameters() if 'scores' in n and is_target_layer(n))

def freeze_non_target_scores(model):
    for name, param in model.named_parameters():
        if 'scores' in name and not is_target_layer(name):
            param.requires_grad = False

def check_non_target_scores_unchanged(model, model_init):
    for name, param in model.named_parameters():
        if 'scores' in name and not is_target_layer(name):
            init_param = model_init.state_dict()[name]
            assert torch.equal(param.data, init_param), f"Non-target score mismatch found in {name}"

def compute_accuracy(outputs, labels):
    _, predicted = torch.max(outputs.data, 1)
    correct = (predicted == labels).sum().item()
    total = labels.size(0)
    return correct / total

@click.command()
@click.option("--learning_rate", default=0.01, help="Initial learning rate.")
@click.option("--num_epochs", default=100, help="Number of epochs to train.")
@click.option("--weight_decay", default=0.0001, help="Weight decay (L2 penalty).")
@click.option("--momentum", default=0.9, help="Momentum.")
@click.option("--remain_rate", default=0.3, help="remain_rate")
@click.option("--seeds", default=5, help="Number of seeds")
@click.option("--batch_size", default=128, help="Batch size for training.")
@click.option("--dataset_name", default="CIFAR10", help="name of dataset")
@click.option("--n_class", default=10, help="number of cls")
@click.option("--dir_name", help="name of dir")
@click.option("--resnet_name", default="ResNet18", help="name of resnet")
def train_model(
    learning_rate,
    num_epochs,
    weight_decay,
    momentum,
    seeds,
    remain_rate,
    batch_size,
    dataset_name,
    n_class,
    dir_name,
    resnet_name
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    current_date = get_current_datetime_for_path()
    for seed in np.arange(seeds):
        save_dir = "./logs/{}/is_prune/RL/{}/{}/{}/{}/{}".format(
            dataset_name,
            resnet_name,
            dir_name,
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
        resnet = get_resnet(resnet_name, n_class).to(device)
        resnet_slth = modify_module_for_slth(
            resnet, remain_rate=remain_rate
        ).to(device)
        resnet_slth_init = copy.deepcopy(resnet_slth).to(device)

        train_loader, test_loader = get_data_loaders(
            dataset_name=dataset_name, batch_size=batch_size
        )
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(
            resnet_slth.parameters(),
            lr=learning_rate,
            momentum=momentum,
            weight_decay=weight_decay,
        )

        scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)

        freeze_non_target_scores(resnet_slth)
        total_params = get_target_params(resnet_slth)
        rl_agent = PolicyGradientAgent(state_dim=total_params, action_dim=total_params, device=device)

        losses = []
        val_accuracies = []

        total_step = len(train_loader)
        for epoch in range(num_epochs):
            resnet_slth.train()
            epoch_losses = []
            states, actions, rewards, log_probs = [], [], [], []
            
            for i, (images, labels) in enumerate(train_loader):
                images = images.to(device)
                labels = labels.to(device)

                state = get_all_scores(resnet_slth)
                action, log_prob = rl_agent.select_action(state)
                update_scores(resnet_slth, action)
                
                outputs = resnet_slth(images)
                loss = criterion(outputs, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_losses.append(loss.item())
                accuracy = compute_accuracy(outputs, labels)
                
                states.append(state)
                actions.append(action)
                rewards.append(accuracy)
                log_probs.append(log_prob)

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
            
            # エポックの終わりにRLエージェントを更新
            rl_loss = rl_agent.update(states, actions, rewards)
            logger.info(f"Epoch [{epoch+1}/{num_epochs}], RL Loss: {rl_loss:.4f}")
            
            check_non_target_scores_unchanged(resnet_slth, resnet_slth_init)
            scheduler.step()
            epoch_loss = sum(epoch_losses) / len(epoch_losses)
            losses.append(epoch_loss)

            logger.info(
                "Epoch [{}/{}], Current learning rate: {:.5f}".format(
                    epoch + 1, num_epochs, scheduler.get_last_lr()[0]
                )
            )
            
            resnet_slth.eval()
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
                if "weight" in name:
                    init_param = resnet_slth_init.state_dict()[name]
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
        torch.save(
            resnet_slth.state_dict(),
            os.path.join(save_dir, "resnet_slth_state.pkl"),
        )
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