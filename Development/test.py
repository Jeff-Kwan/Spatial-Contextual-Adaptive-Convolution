import torch
import torch.nn as nn
from SCAConv_dev import SCAConv
import time

batch = 1
in_c = 16
out_c = 32
a = torch.ones(batch, in_c, 512, 512)
conv1 = SCAConv(in_c, out_c, kernel_size=3, padding=1, stride=1)
conv2 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, stride=1)

start_time = time.time()
output = conv1(a)
end_time = time.time()
print(f"Time taken for SCA-Conv: {end_time - start_time} seconds")

start_time = time.time()
output = conv2(a)
end_time = time.time()
print(f"Time taken for nn.Conv2d: {end_time - start_time} seconds")