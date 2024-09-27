'''
Second Draft of Spatial-Contextual Adaptive Convolution - Conditional.
Drafted by Kwan Leong Chit Jeff, with aid from Github Copilot & ChatGPT4o.
Reimplement in depthwise-separable style to mitigate potential high computational load.
'''

import torch
import torch.nn as nn

class SCAConv_DS(nn.Module):
    '''Depthwise-separable version of Spatial-Contextual Adaptive Convolution - Conditional.'''
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True, b=1, mlp_hidden=16, c_len=8, condition=False):
        super(SCAConv_DS, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding)
        self.b = b  # Hyperparameter for adaptivity of kernel

        # Unfold will be used to extract sliding local blocks from the input tensor
        self.unfold_D = nn.Unfold(kernel_size=self.kernel_size, stride=self.stride, padding=self.padding)

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
        self.condition = condition
        if not condition:
            self.c = nn.Parameter(torch.randn(1, c_len))

        # MLP to generate kernel adjustments
        self.mlp_D = nn.Sequential(
            nn.Linear(4 + c_len, mlp_hidden),  # 4 for location embeding
            nn.ReLU(),
            nn.Linear(mlp_hidden, self.kernel_size[0] * self.kernel_size[1])
        )

        self.mlp_P = nn.Sequential(
            nn.Linear(4 + c_len, mlp_hidden),  # 4 for location embeding
            nn.ReLU(),
            nn.Linear(mlp_hidden, in_channels * out_channels)
        )

        # Initialize MLPs
        for layer in self.mlp_D:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight, gain=1e-1)    # 10x smaller weights
                nn.init.zeros_(layer.bias)
        for layer in self.mlp_P:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight, gain=1e-1)    # 10x smaller weights
                nn.init.zeros_(layer.bias)
    

    def forward(self, x, c=None):
        # Input dimensions
        batch_size, _, height, width = x.size()

        '''Depthwise Convolution'''
        # Unfold the input to extract patches
        x_unfolded = self.unfold_D(x).view(batch_size, self.in_channels, self.kernel_size[0]*self.kernel_size[1], -1)

        # Calculate the number of patches in the output
        output_height = (height + 2 * self.padding[0] - self.kernel_size[0]) // self.stride[0] + 1
        output_width = (width + 2 * self.padding[1] - self.kernel_size[1]) // self.stride[1] + 1

        # Generate location embeddings
        patch_grid_x, patch_grid_y = torch.meshgrid(torch.arange(output_width).to(x.device), torch.arange(output_height).to(x.device), indexing='ij')
        x_norm = patch_grid_x.float() / output_width
        y_norm = patch_grid_y.float() / output_height
        location_embed = torch.stack([x_norm, 1-x_norm, y_norm, 1-y_norm], dim=-1).view(-1, 4).float()  # Shape: (L, 4)
        location_embed = location_embed.unsqueeze(0).expand(batch_size, -1, -1)  # Shape: (batch, L, 4)

        if self.condition:
            # Repeat context input vector to concatenate with location embeddings for each patch
            c = c.unsqueeze(1).expand(-1, location_embed.size(1), -1)  # Shape: (batch, L, hidden_units)
        else:
            c = self.c.unsqueeze(1).expand(batch_size, location_embed.size(1), -1)
            
        # Generate kernel adjustments using MLP
        adjustments = self.mlp_D(torch.cat([location_embed, c], dim=-1).view(batch_size*c.size(1), -1)).view(batch_size, -1, self.kernel_size[0] * self.kernel_size[1])

        # Adaptive kernel adjustment injection
        adjusted_kernel = self.kernel_D.flatten()[None,None,:] + self.b * adjustments  # Shape: (batch, kernel_h * kernel_w, L)

        # Matrix multiplication with einsum
        x = torch.einsum('bikl,blk->bikl', x_unfolded, adjusted_kernel)


        '''Pointwise Convolution'''
        # Generate kernel adjustments using MLP
        adjustments = self.mlp_P(torch.cat([location_embed, c], dim=-1).view(batch_size*output_height*output_width, -1)).view(batch_size,output_height*output_width,self.in_channels,self.out_channels)

        # Adaptive kernel adjustment injection
        adjusted_kernel = self.kernel_P[None,:,:] + self.b * adjustments  # Shape: (batch, kernel_h * kernel_w, L)
        
        # Matrix multiplication with einsum
        x = torch.einsum('bikl,blio->bol', x, adjusted_kernel)  # Shape: (batch_size, in_channels, L, out_channels)

        # Add bias
        if self.bias is not None:
            x += self.bias.view(1, -1, 1)

        # Reshape the output to match expected dimensions
        return x.view(batch_size, self.out_channels, output_height, output_width)