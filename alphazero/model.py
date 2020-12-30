import torch.nn as nn
import torch.nn.functional as F

from torch.distributions.categorical import Categorical


class ConvBlock(nn.Module):
    def __init__(self, in_planes, out_planes=256, kernel=3, stride=1):
        super(ConvBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel, stride, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU()
        )

    def forward(self, x):
        return self.block(x)


class ResBlock(nn.Module):
    def __init__(self, in_planes, out_planes=256, kernel=3, stride=1):
        super(ResBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel, stride, padding=1, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(),
            nn.Conv2d(out_planes, out_planes, kernel, stride, padding=1, bias=False),
            nn.BatchNorm2d(out_planes)
        )
        self.skip = nn.Identity()
        if stride != 1 or in_planes != out_planes:
            self.skip = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(out_planes)
            )

    def forward(self, x):
        out = self.block(x)
        out += self.skip(x)
        out = F.relu(out)
        return out


class PolicyHead(nn.Module):
    def __init__(self, n_moves, in_planes, out_planes=2, kernel=1, stride=1, board_size=8):
        super(PolicyHead, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel, stride, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU()
        )
        self.linear = nn.Linear(board_size ** 2 * 2, n_moves)

    def forward(self, x):
        out = self.block(x).view(1, -1)
        out = self.linear(out)
        return F.softmax(out.squeeze())


class ValueHead(nn.Module):
    def __init__(self, in_planes, out_planes=1, kernel=1, stride=1, board_size=8):
        super(ValueHead, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel, stride, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU()
        )
        self.block2 = nn.Sequential(
            nn.Linear(board_size ** 2, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Tanh()
        )

    def forward(self, x):
        out = self.block1(x).view(1, -1)
        out = self.block2(out)
        return out


class AlphaZero(nn.Module):
    def __init__(self, in_planes, out_planes=256, arch='dual-res', height=20, n_moves=4672):
        super(AlphaZero, self).__init__()
        self.arch = arch
        self.height = height
        self.n_moves = n_moves
        self.in_planes = in_planes
        self.policy_head = PolicyHead(n_moves, out_planes)
        self.value_head = ValueHead(out_planes)
        self.tower = self.make_tower()

    def make_tower(self):
        tower = []
        if self.arch == 'dual-res':
            tower.append(ResBlock(self.in_planes))
            for i in range(1, self.height):
                res_block = ResBlock(in_planes=256)
                tower.append(res_block)
        tower = nn.Sequential(*tower)
        return tower

    def forward(self, x):
        x = self.tower(x)
        actions = self.policy_head(x)
        value = self.value_head(x)
        return actions, value

    @staticmethod
    def get_action(act_probs):
        return Categorical(act_probs).sample()
