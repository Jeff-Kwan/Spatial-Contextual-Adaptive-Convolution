import torch
import torch.nn as nn
from SCAConv_dev import SCAConv
from SCAC_DS import SCAConv_DS
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# batch = 64
# in_c = 32
# out_c = 64
# a = torch.ones(batch, in_c, 32, 32).to(device)

# conv1 = SCAConv(in_c, out_c, kernel_size=3, padding=1, stride=1).to(device)
# conv2 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, stride=1).to(device)
# conv3 = SCAConv_DS(in_c, out_c, kernel_size=3, padding=1, stride=1).to(device)

# # warm up
# output = conv2(a)
# torch.cuda.empty_cache() if torch.cuda.is_available() else None
# torch.cuda.reset_peak_memory_stats(device)

# time.sleep(0.5)

# start_time = time.time()
# output = conv2(a)
# end_time = time.time()
# print(f"Time taken for nn.Conv2d: {(end_time-start_time):.5f} seconds")
# print(f"Max VRAM usage for nn.Conv2d: {torch.cuda.max_memory_allocated(device) / 1024**2:.2f} MB")
# print()
# torch.cuda.empty_cache() if torch.cuda.is_available() else None

# time.sleep(0.5)
# torch.cuda.reset_peak_memory_stats(device)

# start_time = time.time()
# output = conv3(a)
# end_time = time.time()
# print(f"Time taken for Depthwise Separable SCA-Conv: {(end_time-start_time):.5f} seconds")
# print(f"Max VRAM usage for Depthwise Separable SCA-Conv: {torch.cuda.max_memory_allocated(device) / 1024**2:.2f} MB")
# print()
# torch.cuda.empty_cache() if torch.cuda.is_available() else None

# time.sleep(0.5)
# torch.cuda.reset_peak_memory_stats(device)

# start_time = time.time()
# output = conv1(a)
# end_time = time.time()
# print(f"Time taken for SCA-Conv: {(end_time-start_time):.5f} seconds")
# print(f"Max VRAM usage for SCA-Conv: {torch.cuda.max_memory_allocated(device) / 1024**2:.2f} MB")
# print()
# torch.cuda.empty_cache() if torch.cuda.is_available() else None



a = torch.ones([3, 32*32])
conv = nn.Conv1d(3,1,stride=52,kernel_size=256,padding=6)
print(conv(a).shape)