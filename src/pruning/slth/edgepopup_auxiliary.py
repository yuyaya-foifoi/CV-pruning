import copy
import math

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F


class AddAuxiliaryLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, loss):
        assert loss.numel() == 1
        ctx.dtype = loss.dtype
        ctx.required_aux_loss = loss.requires_grad
        return x

    @staticmethod
    def backward(ctx, grad_output):
        grad_loss = None
        if ctx.required_aux_loss:
            grad_loss = torch.ones(
                1, dtype=ctx.dtype, device=grad_output.device
            )
        return grad_output, grad_loss


class GetSubnet(autograd.Function):
    @staticmethod
    def forward(ctx, scores, k):
        # Get the subnetwork by sorting the scores and using the top k%
        out = scores.clone()
        _, idx = scores.flatten().sort()
        j = int((1 - k) * scores.numel())

        # flat_out and out access the same memory.
        flat_out = out.flatten()
        flat_out[idx[:j]] = 0
        flat_out[idx[j:]] = 1

        return out

    """
    @staticmethod
    def backward(ctx, g):
        return g, None
    """

    @staticmethod
    def backward(ctx, g):
        return g, None


class SubnetLinear(nn.Linear):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # initialize the scores
        self.scores = nn.Parameter(torch.Tensor(self.weight.size()))
        nn.init.kaiming_uniform_(self.scores, a=math.sqrt(5))
        self.weight.requires_grad = False
        self.bias = None
        self.abs_score = True
        self.subnet_func = GetSubnet
        self.initial_aux_loss_weight = 0.001
        # 補助ロスが極小になるepoch数
        self.min_aux_loss_epoch = 100
        
        # 現在のepoch数
        self.current_epoch = 0

    def set_current_epoch(self, epoch):
        self.current_epoch = epoch

    def get_aux_loss_weight(self):
        # 補助ロスの重みを計算
        return self.initial_aux_loss_weight * (1 - self.current_epoch / self.min_aux_loss_epoch)

    @property
    def clamped_scores(self):
        if self.abs_score:
            return self.scores.abs()
        else:
            self.scores.data = F.relu(self.scores.data)
            return self.scores

    def init_weight(self, name=None):
        if name is None:
            name = "signed_constant"
        self._init_weight(self.weight, name=name)

    def _init_weight(self, weight, name="signed_constant"):
        if name == "signed_constant":
            fan = nn.init._calculate_correct_fan(weight, mode="fan_in")
            gain = nn.init.calculate_gain("relu")
            std = gain / math.sqrt(fan)
            weight.data = weight.data.sign() * std

        elif name == "scaled_signed_constant":
            fan = nn.init._calculate_correct_fan(weight, mode="fan_in")
            fan = fan * (1 - self.remain_rate)
            gain = nn.init.calculate_gain("relu")
            std = gain / math.sqrt(fan)
            weight.data = weight.data.sign() * std

    def set_remain_rate(self, remain_rate):
        self.remain_rate = remain_rate

    def forward(self, x):
        subnet = self.subnet_func.apply(self.clamped_scores, self.remain_rate)
        w = self.weight * subnet
        score_reg_loss = self.get_aux_loss_weight() * torch.norm(self.clamped_scores, p=2)
        return AddAuxiliaryLoss.apply(
            F.linear(x, w, self.bias), score_reg_loss
        )


# Not learning weights, finding subnet
class SubnetConv(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.scores = nn.Parameter(torch.Tensor(self.weight.size()))
        nn.init.kaiming_uniform_(self.scores, a=math.sqrt(5))
        self.weight.requires_grad = False
        self.abs_score = True
        self.subnet_func = GetSubnet
        self.initial_aux_loss_weight = 0.001
        self.min_aux_loss_epoch = 100
        # 現在のepoch数
        self.current_epoch = 0

    def set_current_epoch(self, epoch):
        self.current_epoch = epoch

    def get_aux_loss_weight(self):
        # 補助ロスの重みを計算
        return self.initial_aux_loss_weight * (1 - self.current_epoch / self.min_aux_loss_epoch)


    @property
    def clamped_scores(self):
        if self.abs_score:
            return self.scores.abs()
        else:
            self.scores.data = F.relu(self.scores.data)
            return self.scores

    def set_remain_rate(self, remain_rate):
        self.remain_rate = remain_rate

    def init_weight(self, name=None):
        if name is None:
            name = "signed_constant"
        self._init_weight(self.weight, name=name)

    def _init_weight(self, weight, name="signed_constant"):
        if name == "signed_constant":
            fan = nn.init._calculate_correct_fan(weight, mode="fan_in")
            gain = nn.init.calculate_gain("relu")
            std = gain / math.sqrt(fan)
            weight.data = weight.data.sign() * std

        elif name == "scaled_signed_constant":
            fan = nn.init._calculate_correct_fan(weight, mode="fan_in")
            fan = fan * (1 - self.remain_rate)
            gain = nn.init.calculate_gain("relu")
            std = gain / math.sqrt(fan)
            weight.data = weight.data.sign() * std

    def forward(self, x):
        subnet = self.subnet_func.apply(self.clamped_scores, self.remain_rate)
        w = self.weight * subnet
        score_reg_loss = self.get_aux_loss_weight() * torch.norm(self.clamped_scores, p=2)
        return AddAuxiliaryLoss.apply(
            F.conv2d(
                x,
                w,
                self.bias,
                self.stride,
                self.padding,
                self.dilation,
                self.groups,
            ),
            score_reg_loss,
        )


# 同様にBNもSLTH仕様に変更（アフィンを削除）
class NonAffineBatchNorm1d(nn.BatchNorm1d):
    def __init__(self, dim):
        super(NonAffineBatchNorm1d, self).__init__(dim, affine=False)


class NonAffineBatchNorm2d(nn.BatchNorm2d):
    def __init__(self, dim):
        super(NonAffineBatchNorm2d, self).__init__(dim, affine=False)


# ResNet18の各ConvレイヤーをSubnetConvに置き換える．
def recursive_setattr(obj, name, value):
    if "." in name:
        parent, child = name.split(".")[0], ".".join(name.split(".")[1:])
        recursive_setattr(getattr(obj, parent), child, value)
    else:
        setattr(obj, name, value)


def modify_module_for_slth(net, remain_rate, init_mode=None):
    net_cp = copy.deepcopy(net)
    named_modules = [(n, m) for n, m in net_cp.named_modules()]
    print("#Modules: {}".format(len(named_modules)))
    for n, m in named_modules:
        if isinstance(m, nn.Conv2d):
            print("Replace nn.Conv2d with SubnetConv: {}".format(n))
            m2 = SubnetConv(
                m.in_channels,
                m.out_channels,
                stride=m.stride,
                kernel_size=m.kernel_size,
                padding=m.padding,
                # bias=m.bias,
                bias=(m.bias is not None),
            )
            m2.weight = nn.Parameter(m.weight.detach().clone())
            m2.weight.requires_grad = False
            m2.set_remain_rate(remain_rate)
            m2.init_weight(init_mode)
            recursive_setattr(net_cp, n, m2)
        # you can write other SLTH modules here
        elif isinstance(m, nn.Linear):
            print("Replace nn.Linear with SubnetLinear: {}".format(n))
            m2 = SubnetLinear(m.in_features, m.out_features, bias=False)
            # memo 2023/09/07
            # 重みをloadしてSubnetLinearを適用した時点では重みはランダムになっている
            m2.weight = nn.Parameter(m.weight.detach().clone())
            m2.weight.requires_grad = False
            m2.set_remain_rate(remain_rate)
            m2.init_weight(init_mode)
            recursive_setattr(net_cp, n, m2)
        elif isinstance(m, nn.BatchNorm1d):
            print(
                "Replace nn.BatchNorm1d with NonAffineBatchNorm1d: {}".format(
                    n
                )
            )
            m2 = NonAffineBatchNorm1d(dim=m.num_features)
            recursive_setattr(net_cp, n, m2)
        elif isinstance(m, nn.BatchNorm2d):
            print(
                "Replace nn.BatchNorm2d with NonAffineBatchNorm2d: {}".format(
                    n
                )
            )
            m2 = NonAffineBatchNorm2d(dim=m.num_features)
            recursive_setattr(net_cp, n, m2)
        else:
            print("No modification", n, type(m))
    return net_cp
