'''
Third Draft of Spatial-Contextual Adaptive Convolution - Conditional.
Drafted by Kwan Leong Chit Jeff, with aid from Github Copilot & ChatGPT4o.
Reimplement in depthwise-separable style to mitigate potential high computational load.

Optimized for calculation but incompatible with torch.jit.save for unknown reasons 
    (likely due self.c and self.location_embed).
'''

import torch
import torch.nn as nn

class SCAConv_DS(nn.Module):
    '''Optimized Depthwise-Separable Spatial-Contextual Adaptive Convolution - Conditional.'''
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 bias=True, b=1, mlp_hidden=16, c_len=8, condition=False):
        super(SCAConv_DS, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding)
        self.b = b  # Hyperparameter for adaptivity of kernel
        self.a = nn.Parameter(torch.tensor([1.0, 1.0]))  # Hyperparameter for adaptivity of kernel
        self.condition = condition

        # Unfold will be used to extract sliding local blocks from the input tensor
        self.unfold = nn.Unfold(kernel_size=self.kernel_size, stride=self.stride, padding=self.padding)

        # Kernel (weights) initialized using Xavier
        self.kernel_D = nn.Parameter(torch.empty(self.kernel_size[0], self.kernel_size[1]))
        nn.init.xavier_uniform_(self.kernel_D)
        self.kernel_P = nn.Parameter(torch.empty(in_channels, out_channels))
        nn.init.xavier_uniform_(self.kernel_P)

        # Bias term
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter('bias', None)

        # Unconditional seed
        if not condition:
            self.c = nn.Parameter(torch.randn(1, c_len))

        # MLP to generate kernel adjustments
        self.mlp = nn.Sequential(
            nn.Linear(4 + c_len, mlp_hidden),
            nn.SiLU(),
            nn.Linear(mlp_hidden, self.kernel_size[0] * self.kernel_size[1] + in_channels * out_channels)
            )

    def forward(self, x, c=None):
        batch_size, _, height, width = x.size()
        output_height = (height + 2 * self.padding[0] - self.kernel_size[0]) // self.stride[0] + 1
        output_width = (width + 2 * self.padding[1] - self.kernel_size[1]) // self.stride[1] + 1

        # Unfold the input to extract patches
        x_unfolded = self.unfold(x)  # Shape: (batch_size, in_channels * K, L)
        K = self.kernel_size[0] * self.kernel_size[1]
        L = x_unfolded.size(-1)
        x_unfolded = x_unfolded.view(batch_size, self.in_channels, K, L)  # Shape: (batch_size, in_channels, K, L)

        # Generate location embeddings (calculate once)
        if not hasattr(self, 'location_embed') or self.location_embed.size(1) != L:
            with torch.no_grad():
                x_coords = torch.linspace(0, 1, steps=output_width, device=x.device)
                y_coords = torch.linspace(0, 1, steps=output_height, device=x.device)
                x_norm, y_norm = torch.meshgrid(x_coords, y_coords, indexing='ij')
                x_norm = x_norm.reshape(-1)
                y_norm = y_norm.reshape(-1)
                location_embed = torch.stack([x_norm, 1 - x_norm, y_norm, 1 - y_norm], dim=-1)
                self.location_embed = location_embed.unsqueeze(0)  # Shape: (1, L, 4)
        location_embed = self.location_embed.expand(batch_size, -1, -1)  # Shape: (batch_size, L, 4)

        if self.condition:
            c = c.unsqueeze(1).expand(-1, L, -1)  # Shape: (batch_size, L, c_len)
        else:
            c = self.c.expand(batch_size, L, -1)  # Shape: (batch_size, L, c_len)

        # Generate kernel adjustments using MLP
        adjustments = self.mlp(torch.cat([location_embed, c], dim=-1))  # Shape: (batch_size, L, 4 + c_len)

        '''Depthwise Convolution'''
        # Adaptive kernel adjustment injection
        adjusted_kernel_D = self.kernel_D.flatten().view(1, 1, K) + self.b * self.a[0] * adjustments[:, :, :K]  # Shape: (batch_size, L, K)
        adjusted_kernel_D = adjusted_kernel_D.transpose(1, 2).unsqueeze(1)  # Shape: (batch_size, 1, K, L)
        z = x_unfolded * adjusted_kernel_D  # Shape: (batch_size, in_channels, K, L)

        '''Pointwise Convolution'''
        # Adaptive kernel adjustment injection
        adjusted_kernel_P = self.kernel_P.view(1, 1, self.in_channels, self.out_channels) + self.b * self.a[1] * adjustments[:, :, K:].view(batch_size, L, self.in_channels, self.out_channels) # Shape: (batch_size, L, in_channels, out_channels)

        # Compute output using batch matrix multiplication
        z_sum = z.sum(dim=2).permute(0, 2, 1).unsqueeze(2)
        z = torch.matmul(z_sum, adjusted_kernel_P)  # Shape: (batch_size, L, 1, out_channels)
        z = z.squeeze(2).permute(0, 2, 1)  # Shape: (batch_size, out_channels, L)

        # Add bias
        if self.bias is not None:
            z += self.bias.view(1, -1, 1)

        # Reshape the output 
        return z.view(batch_size, self.out_channels, output_height, output_width)