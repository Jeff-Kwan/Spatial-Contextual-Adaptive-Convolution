import torch
import torch.nn as nn
from SCAConv_dev import SCAConv
from time import time



class conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(conv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding)

    def forward(self, x):
        return self.conv(x)
    
class SCAC(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, b=1, c_len=8):
        super(SCAC, self).__init__()
        self.conv = SCAConv(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=True, b=b, c_len=c_len)
    
    def forward(self, x):
        c = torch.ones(x.shape[0], 8).to(x.device)
        return self.conv(x, c)

def test_cases():
    # batch, in_channels, out_channels, height, width, kernel_size, stride, padding
    cases = [
        (1, 3, 16, 64, 64, 3, 1, 1),
        (8, 3, 32, 128, 128, 5, 2, 2),
        (16, 3, 64, 256, 256, 7, 1, 3),
        (32, 3, 128, 512, 512, 3, 2, 1),
        (32, 3, 256, 1024, 1024, 5, 7, 2)
    ]
    
    results = []

    for i ,case in enumerate(cases):
        batch_size, in_channels, out_channels, height, width, kernel_size, stride, padding = case
        # init models
        conv2d_model = conv2d(in_channels, out_channels, kernel_size, stride, padding)
        scaconv_model = SCAC(in_channels, out_channels, kernel_size, stride, padding)

        # create random input
        input = torch.randn(batch_size, in_channels, height, width)

        # Time the model runs
        start = time()
        _ = conv2d_model(input)
        conv_2d_time = time() - start
        
        start = time()
        _ = scaconv_model(input)
        scaconv_time = time() - start

        results.append((i, conv_2d_time, scaconv_time))

    # Print results in a pretty table
    print(f"{'Case':<10}{'Conv2D Time':<20}{'SCAConv Time':<20}")
    for i, conv_2d_time, scaconv_time in results:
        print(f"{i:<10}{conv_2d_time:<20}{scaconv_time:<20}")



if __name__ == "__main__":
    test_cases()