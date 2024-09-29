'''
Third Draft of (-Spatial-) Contextual Adaptive Convolution - Conditional.
Drafted by Kwan Leong Chit Jeff, with aid from Github Copilot & ChatGPT4o.
'''

import torch
import torch.nn as nn

class CAConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1, bias=True, b=1, mlp_hidden=16, c_len=8, condition=False):
        super(CAConv, self).__init__()
        # Base Conv2d
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=bias)

        # Parameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding)
        self.b = b

        # Adjustment MLP
        self.mlp = nn.Sequential(
            nn.Linear(c_len, mlp_hidden),
            nn.ReLU(),
            nn.Linear(mlp_hidden, self.kernel_size[0] * self.kernel_size[1] * self.in_channels * self.out_channels)
        )

        # Unconditional seed
        self.condition = condition
        if not condition:
            self.c = nn.Parameter(torch.randn(1, c_len))

    def forward(self, x, c=None):
        # Get the kernel adjustment
        if self.condition:
            adj = self.mlp(c)
        else:
            adj = self.mlp(self.c)

        adj = adj.view(x.shape[0], self.out_channels, self.in_channels, self.kernel_size[0], self.kernel_size[1])
        for i in range(x.shape[0]):
            self.conv.weight = self.conv.weight + self.b * adj[i]
            conv = self.conv(x[i])
            self.conv.weight = self.conv.weight - self.b * adj[i]

        return conv