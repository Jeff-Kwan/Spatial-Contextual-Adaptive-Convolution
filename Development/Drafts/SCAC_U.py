'''
First Draft of Spatial Contextual Adaptive Convolution.
Drafted by Kwan Leong Chit Jeff, with aid from Github copilot & ChatGPT4o.
File darfted on 26-9-2024.

Unconditional spatial varying adaptive convolution with mlp-powered kernel adjustment, 
and a trainable seed vector. Conceptually similar to applying LoRA on convolutional
kernels, but during the training stage.

Note the use of training schedule to add the mlp patch adjustment, as well as the small
weight initialisation as the adjustment should be a small weight injection.
'''

import torch
import torch.nn as nn

class PatchConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True, b=0.1):
        super(PatchConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding)
        self.b = b  # Beta value for patch-specific kernel adjustment

        # Unfold will be used to extract sliding local blocks from the input tensor
        self.unfold = nn.Unfold(kernel_size=self.kernel_size, stride=self.stride, padding=self.padding)
        
        # Kernel (weights)
        self.kernel = nn.Parameter(torch.randn(out_channels, in_channels, *self.kernel_size))

        # Bias term
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter('bias', None)
    
    def forward(self, x):
        # Input dimensions
        batch_size, _, height, width = x.size()

        # Unfold the input to extract patches
        x_unfolded = self.unfold(x)  # Shape: (batch_size, in_channels * kernel_h * kernel_w, L), L = number of patches

        # Flatten the kernel
        kernel_flat = self.kernel.view(self.out_channels, -1)  # Shape: (out_channels, in_channels * kernel_h * kernel_w)

        # Calculate the number of patches in the output
        output_height = (height + 2 * self.padding[0] - self.kernel_size[0]) // self.stride[0] + 1
        output_width = (width + 2 * self.padding[1] - self.kernel_size[1]) // self.stride[1] + 1

        # Modify the kernel for each patch with the patch-specific adjustment using b * (x + y)
        patch_grid_x, patch_grid_y = torch.meshgrid(torch.arange(output_width), torch.arange(output_height), indexing='ij')
        patch_grid = patch_grid_x + patch_grid_y  # Shape: (output_width, output_height)
        patch_grid = patch_grid.view(1, 1, -1).to(x.device)  # Reshape to (1, 1, L) where L is the number of patches
        patch_adjustment = self.b * patch_grid  # Shape: (1, 1, L)

        # Add patch-specific adjustment to the kernel
        adjusted_kernel_flat = kernel_flat.unsqueeze(-1) + patch_adjustment  # Shape: (out_channels, in_channels * kernel_h * kernel_w, L)

        # Matrix multiplication with einsum for batch-wise computation
        out_unf = torch.einsum('bci,oic->bco', x_unfolded.transpose(1, 2), adjusted_kernel_flat)  # Shape: (batch_size, L, out_channels)
        out_unf = out_unf.transpose(1, 2)  # Shape: (batch_size, out_channels, L)

        # Add bias
        if self.bias is not None:
            out_unf += self.bias.view(1, -1, 1)

        # Reshape the output to match expected dimensions
        out = out_unf.view(batch_size, self.out_channels, output_height, output_width)

        return out


# Test function to compare PatchConv2d with nn.Conv2d
def test_patch_conv2d_with_beta():
    # Random seed for reproducibility
    torch.manual_seed(0)

    # Set parameters
    batch_size = 7
    in_channels = 32
    out_channels = 23
    kernel_size = (3, 7)
    stride = (3, 4)
    padding = (1, 2)
    height, width = 128, 99

    # Create input tensor
    input_tensor = torch.randn(batch_size, in_channels, height, width)

    # Initialize PatchConv2d with the beta modification
    patch_conv = PatchConv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=True, b=0.1)

    # Get output from PatchConv2d
    patch_conv_output = patch_conv(input_tensor)

    # Just print the output dimensions to confirm it works
    print("PatchConv2d Output Shape:", patch_conv_output.shape)



# Test function to compare PatchConv2d with nn.Conv2d
def test_patch_conv2d_equivalence():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Random seed for reproducibility
    torch.manual_seed(0)

    # Set parameters
    batch_size = 9
    in_channels = 32
    out_channels = 16
    kernel_size = (3, 3)
    stride = (2, 2)
    padding = (2, 2)
    height, width = 64,64

    # Create input tensor
    input_tensor = torch.randn(batch_size, in_channels, height, width).to(device)

    # Initialize both PatchConv2d and nn.Conv2d with the same kernel and parameters
    patch_conv = PatchConv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=True, b=0).to(device)
    conv2d = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=True).to(device)

    # Copy the kernel weights and bias from PatchConv2d to nn.Conv2d
    with torch.no_grad():
        conv2d.weight.copy_(patch_conv.kernel)
        if patch_conv.bias is not None:
            conv2d.bias.copy_(patch_conv.bias)

    # Get outputs from both layers
    patch_conv_output = patch_conv(input_tensor)
    conv2d_output = conv2d(input_tensor)

    # Assert that both outputs are close around order of 1e-4
    diff = torch.abs(patch_conv_output - conv2d_output)
    assert torch.allclose(patch_conv_output, conv2d_output, atol=2e-4), f"Diff max {diff.max()}"
    assert diff.mean() < 1e-5, f"Diff mean {diff.mean()}"

    print("Test passed! PatchConv2d matches nn.Conv2d.")

# Run the test
if __name__ == "__main__":
    test_patch_conv2d_with_beta()
    test_patch_conv2d_equivalence()
