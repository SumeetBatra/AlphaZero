import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_planes, out_planes=256, kernel=3, stride=1):
        super(ConvBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel, stride),
            nn.BatchNorm2d(out_planes),
            nn.ReLU()
        )

    def forward(self, x):
        return self.block(x)


class ResBlock(nn.Module):
    def __init__(self, in_planes, out_planes=256, kernel=3, stride=1):
        super(ResBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel, stride),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(),
            nn.Conv2d(out_planes, out_planes, kernel, stride),
            nn.BatchNorm2d(out_planes)
        )

    def forward(self, x):
        out = self.block(x)
        out += x
        out = nn.ReLU(out)
        return out


class PolicyHead(nn.Module):
    def __init__(self, n_moves, in_planes, out_planes=2, kernel=1, stride=1, board_size=8):
        super(PolicyHead, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel, stride),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(),
            nn.Linear(board_size ** 2 * 2, n_moves)
        )

    def forward(self, x):
        return self.block(x)


class ValueHead(nn.Module):
    def __init__(self, in_planes, out_planes=1, kernel=1, stride=1, board_size=8):
        super(ValueHead, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel, stride),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(),
            nn.Linear(board_size ** 2, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Tanh()
        )

    def forward(self, x):
        return self.block(x)


class AlphaZero(nn.Module):
    def __init__(self, in_planes, arch='dual-res', height=20, n_moves=4672):
        super(AlphaZero, self).__init__()
        self.arch = arch
        self.height = height
        self.n_moves = n_moves
        self.in_planes = in_planes
        self.policy_head = PolicyHead(n_moves, in_planes)
        self.value_head = ValueHead(in_planes)
        self.tower = self.make_tower()

    def make_tower(self):
        tower = []
        if self.arch == 'dual-res':
            for i in range(self.height):
                res_block = ResBlock(self.in_planes)
                tower.append(res_block)
        tower = nn.Sequential(*tower)
        return tower

    def forward(self, x):
        x = self.tower(x)
        actions = self.policy_head(x)
        value = self.value_head(x)
        return actions, value
