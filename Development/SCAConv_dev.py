'''
First Draft of Spatial-Contextual Adaptive Convolution - Conditional.
Drafted by Kwan Leong Chit Jeff, with aid from Github Copilot & ChatGPT4o.
'''

import torch
import torch.nn as nn

class SCAConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True, b=1, mlp_hidden=16, c_len=8, condition=False):
        super(SCAConv, self).__init__()
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

        # Unconditional seed
        self.condition = condition
        if not condition:
            self.c = nn.Parameter(torch.randn(1, c_len))

        # MLP to generate kernel adjustments
        self.mlp = nn.Sequential(
            nn.Linear(4 + c_len, mlp_hidden * 2),  # 4 for location embeding
            nn.ReLU(),
            nn.Linear(mlp_hidden * 2, out_channels * in_channels * self.kernel_size[0] * self.kernel_size[1])
        )

        # Initialize MLP
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight, gain=1e-1)    # 10x smaller weights
                nn.init.zeros_(layer.bias)
    
    def forward(self, x, c=None):
        # Input dimensions
        batch_size, _, height, width = x.size()

        # Unfold the input to extract patches
        x_unfolded = self.unfold(x)  # Shape: (batch_size, in_channels * kernel_h * kernel_w, L), L = number of patches

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
        adjustments = self.mlp(torch.cat([location_embed, c], dim=-1).view(batch_size*c.size(1), -1)
                                    ).view(batch_size, self.out_channels, output_width * output_height, -1).transpose(2,3)  # Shape: (out_channels, L, in_channels * kernel_h * kernel_w)

        # Adaptive kernel adjustment injection
        adjusted_kernel = self.kernel[None,:,:,None] + self.b * adjustments  # Shape: (batch, out_channels, in_channels * kernel_h * kernel_w, L)

        # Matrix multiplication with einsum
        out = torch.einsum('bic,boic->boc', x_unfolded, adjusted_kernel)  # Shape: (batch_size, L, out_channels)

        # Add bias
        if self.bias is not None:
            out += self.bias.view(1, -1, 1)

        # Reshape the output to match expected dimensions
        return out.view(batch_size, self.out_channels, output_height, output_width)
    



'''Testing function for development'''
# Test function to compare SCAConv with nn.Conv2d
def test_patch_conv2d_equivalence():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Random seed for reproducibility
    torch.manual_seed(0)

    # Set parameters
    batch_size = 9
    in_channels = 1
    out_channels = 1
    kernel_size = (3, 3)
    stride = (2, 2)
    padding = (1, 1)
    height, width = 14,14
    c_len=8

    # Create input tensor
    input_tensor = torch.randn(batch_size, in_channels, height, width).to(device)

    # Initialize both SCAConv and nn.Conv2d with the same kernel and parameters
    patch_conv = SCAConv(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=True, c_len=c_len).to(device)
    conv2d = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=True).to(device)

    # Copy the kernel weights and bias from SCAConv to nn.Conv2d
    with torch.no_grad():
        conv2d.weight.copy_(patch_conv.kernel.view(out_channels, in_channels, kernel_size[0], kernel_size[1]))
        if patch_conv.bias is not None:
            conv2d.bias.copy_(patch_conv.bias)

    # Get outputs from both layers
    c = torch.zeros(batch_size, c_len).to(device)
    patch_conv.b = 0
    patch_conv_output = patch_conv(input_tensor)
    conv2d_output = conv2d(input_tensor)

    # Assert that both outputs are close around order of 1e-4
    diff = torch.abs(patch_conv_output - conv2d_output)
    assert torch.allclose(patch_conv_output, conv2d_output, atol=1e-5), f"Diff max {diff.max()}"
    assert diff.mean() < 1e-6, f"Diff mean {diff.mean()}"

    print("Test passed! SCAConv matches nn.Conv2d.")

# Run the test
if __name__ == "__main__":
    test_patch_conv2d_equivalence()
