'''
First Draft of Spatial Contextual Adaptive Convolution - Conditional.
Drafted by Kwan Leong Chit Jeff, with aid from Github copilot & ChatGPT4o.
File darfted on 26-9-2024.
'''

import torch
import torch.nn as nn

class PatchConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True, b=1, hidden_units=16):
        super(PatchConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding)
        self.b = b  # Hyperparameter for patch-specific kernel adjustment

        # Unfold will be used to extract sliding local blocks from the input tensor
        self.unfold = nn.Unfold(kernel_size=self.kernel_size, stride=self.stride, padding=self.padding)

        # Kernel (weights) initialized using Xavier
        self.kernel = nn.Parameter(torch.empty(out_channels, in_channels * self.kernel_size[0] * self.kernel_size[1]))
        nn.init.xavier_uniform_(self.kernel)

        # Bias term
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter('bias', None)

        # Seed vector for adjustment (trainable)
        self.seed_vector = nn.Parameter(torch.randn(hidden_units))

        # MLP to generate kernel adjustments
        self.mlp = nn.Sequential(
            nn.Linear(2 + hidden_units, hidden_units * 2),  # 2 for patch location (x, y) and hidden_units from seed
            nn.SiLU(),
            nn.Linear(hidden_units * 2, out_channels * in_channels * self.kernel_size[0] * self.kernel_size[1])
        )

        # Initialize MLP with small weights
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight, gain=1e-1) # 10x smaller weights because it is an injection
                nn.init.zeros_(layer.bias)
    
    def forward(self, x):
        # Input dimensions
        batch_size, _, height, width = x.size()

        # Unfold the input to extract patches
        x_unfolded = self.unfold(x)  # Shape: (batch_size, in_channels * kernel_h * kernel_w, L), L = number of patches

        # Calculate the number of patches in the output
        output_height = (height + 2 * self.padding[0] - self.kernel_size[0]) // self.stride[0] + 1
        output_width = (width + 2 * self.padding[1] - self.kernel_size[1]) // self.stride[1] + 1

        # Generate patch locations
        patch_grid_x, patch_grid_y = torch.meshgrid(torch.arange(output_width), torch.arange(output_height), indexing='ij')
        patch_locations = torch.stack([patch_grid_x/output_width, patch_grid_y/output_height], dim=-1).view(-1, 2).float().to(x.device)  # Shape: (L, 2)

        # Repeat seed vector to concatenate with patch locations for each patch
        seed_repeated = self.seed_vector.unsqueeze(0).repeat(patch_locations.size(0), 1)  # Shape: (L, hidden_units)

        # Patch-adjustments
        patch_adjustments = self.mlp(torch.cat([patch_locations, seed_repeated], dim=-1)).view(
                            self.out_channels, output_width * output_height, -1).transpose(1,2)  # Shape: (out_channels, L, in_channels * kernel_h * kernel_w)

        # Add patch-specific adjustment to the kernel
        adjusted_kernel = self.kernel[:,:,None] + self.b * patch_adjustments  # Shape: (out_channels, in_channels * kernel_h * kernel_w, L)

        # Matrix multiplication with einsum for batch-wise computation
        out_unf = torch.einsum('bci,oic->bco', x_unfolded.transpose(1, 2), adjusted_kernel)  # Shape: (batch_size, L, out_channels)
        out_unf = out_unf.transpose(1, 2)  # Shape: (batch_size, out_channels, L)

        # Add bias
        if self.bias is not None:
            out_unf += self.bias.view(1, -1, 1)

        # Reshape the output to match expected dimensions
        return out_unf.view(batch_size, self.out_channels, output_height, output_width)