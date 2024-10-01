'''
Second Draft of Spatial-Contextual Adaptive Convolution - Conditional.
Drafted by Kwan Leong Chit Jeff, with aid from Github Copilot & ChatGPT4o.
Reimplement in depthwise-separable style to mitigate potential high computational load.
'''

import torch
import torch.nn as nn

class KernelAttention(nn.Module):
    def __init__(self, input_dim, base_kernel_size):
        super().__init__()
        kernel_size = base_kernel_size[0] * base_kernel_size[1]
        self.key_proj = nn.Linear(kernel_size, kernel_size, bias=False)
        self.value_proj = nn.Linear(kernel_size, kernel_size, bias=False)
        self.query_proj = nn.Linear(input_dim, kernel_size, bias=False)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, base_kernel, condition):
        # Flatten and project the base kernel
        base_kernel_flat = base_kernel.view(-1)
        key = self.key_proj(base_kernel_flat)
        value = self.value_proj(base_kernel_flat)
        # Project the condition
        query = self.query_proj(condition)
        # Compute attention weights
        attn_weights = self.softmax(query * key)
        # Generate kernel adjustments
        adjustments = attn_weights * value
        return adjustments.view(condition.size(0), *base_kernel.size())


class SCAC_DS_Attn(nn.Module):
    '''Depthwise-separable version of Spatial-Contextual Adaptive Convolution - Conditional.'''
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 bias=True, b=1, c_len=8, condition=False):
        super(SCAC_DS_Attn, self).__init__()
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

        # Attention-based kernel adjustment
        self.c_len = c_len
        self.adj_D = KernelAttention(input_dim=4+c_len, base_kernel_size=self.kernel_D.size())
        self.adj_P = KernelAttention(input_dim=4+c_len, base_kernel_size=self.kernel_P.size())



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
        c = torch.cat([location_embed, c], dim=-1).view(-1, 4+self.c_len)

        '''Depthwise Convolution'''
        # Adaptive kernel adjustment injection
        adj_D = self.adj_D(self.kernel_D, c).view(batch_size, L, -1)
        adjusted_kernel_D = self.kernel_D.flatten().view(1, 1, K) + self.b * self.a[0] * adj_D  # Shape: (batch_size, L, K)
        adjusted_kernel_D = adjusted_kernel_D.transpose(1, 2).unsqueeze(1)  # Shape: (batch_size, 1, K, L)
        x = x_unfolded * adjusted_kernel_D  # Shape: (batch_size, in_channels, K, L)

        '''Pointwise Convolution'''
        # Adaptive kernel adjustment injection
        adj_P = self.adj_P(self.kernel_P, c).view(batch_size, L, self.in_channels, self.out_channels)  # Shape: (batch_size, in_channels, out_channels)
        adjusted_kernel_P = self.kernel_P.view(1, 1, self.in_channels, self.out_channels) + self.b * self.a[1] * adj_P # Shape: (batch_size, L, in_channels, out_channels)

        # Compute output using batch matrix multiplication
        x = x.sum(dim=2)  # Shape: (batch_size, in_channels, L)
        x = torch.matmul(x.permute(0, 2, 1).unsqueeze(2), adjusted_kernel_P)  # Shape: (batch_size, L, 1, out_channels)
        x = x.squeeze(2).permute(0, 2, 1)  # Shape: (batch_size, out_channels, L)

        # Add bias
        if self.bias is not None:
            x += self.bias.view(1, -1, 1)

        # Reshape the output 
        return x.view(batch_size, self.out_channels, output_height, output_width)

    


'''Testing function for development'''
# Test function to compare SCAConv with nn.Conv2d
def test_patch_conv2d_equivalence():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Random seed for reproducibility
    torch.manual_seed(0)

    # Set parameters
    batch_size = 7
    in_channels = 13
    out_channels = 11
    kernel_size = (3, 3)
    stride = (2, 2)
    padding = (1, 1)
    height, width = 14,14
    c_len=8

    # Create input tensor
    input_tensor = torch.randn(batch_size, in_channels, height, width).to(device)

    # Initialize both SCAConv and nn.Conv2d with the same kernel and parameters
    patch_conv = SCAC_DS_Attn(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False, c_len=c_len, condition=False).to(device)
    conv2d = nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False, groups=in_channels),
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=False)
        ).to(device)

    # Set all weights to 1.0
    patch_conv.kernel_D.data.fill_(1.0)
    patch_conv.kernel_P.data.fill_(1.0)
    conv2d[0].weight.data.fill_(1.0)
    conv2d[1].weight.data.fill_(1.0)

    # Get outputs from both layers
    patch_conv.b = 0
    patch_conv_output = patch_conv(input_tensor)
    conv2d_output = conv2d(input_tensor)

    # Assert that both outputs are close around order of 1e-4
    diff = torch.abs(patch_conv_output - conv2d_output)
    assert torch.allclose(patch_conv_output, conv2d_output, atol=1e-5), f"Diff max {diff.max()}"
    assert diff.mean() < 1e-6, f"Diff mean {diff.mean()}"
    print("Diff stats: max {:.4e}, mean {:.4e}".format(diff.max(), diff.mean()))
    print("Test passed! SCAConv matches nn.Conv2d.")

# Run the test
if __name__ == "__main__":
    test_patch_conv2d_equivalence()
