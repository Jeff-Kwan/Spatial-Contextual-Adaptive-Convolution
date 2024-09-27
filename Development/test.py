import torch
import torch.nn as nn
from SCAConv_dev import SCAConv
from Depthwise_Separable_SCAC import SCAConv_DS
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

batch = 4
in_c = 3
out_c = 8
a = torch.ones(batch, in_c, 1024,1024).to(device)

conv1 = SCAConv(in_c, out_c, kernel_size=3, padding=1, stride=1).to(device)
conv2 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, stride=1).to(device)
conv3 = SCAConv_DS(in_c, out_c, kernel_size=3, padding=1, stride=1).to(device)

torch.cuda.reset_peak_memory_stats(device)

start_time = time.time()
output = conv2(a)
end_time = time.time()
print(f"Time taken for nn.Conv2d: {(end_time-start_time):.5f} seconds")
print(f"Max VRAM usage for nn.Conv2d: {torch.cuda.max_memory_allocated(device) / 1024**2:.2f} MB")
print()
torch.cuda.empty_cache() if torch.cuda.is_available() else None

time.sleep(0.5)
torch.cuda.reset_peak_memory_stats(device)

start_time = time.time()
output = conv3(a)
end_time = time.time()
print(f"Time taken for Depthwise Separable SCA-Conv: {(end_time-start_time):.5f} seconds")
print(f"Max VRAM usage for Depthwise Separable SCA-Conv: {torch.cuda.max_memory_allocated(device) / 1024**2:.2f} MB")
print()
torch.cuda.empty_cache() if torch.cuda.is_available() else None

time.sleep(0.5)
torch.cuda.reset_peak_memory_stats(device)

start_time = time.time()
output = conv1(a)
end_time = time.time()
print(f"Time taken for SCA-Conv: {(end_time-start_time):.5f} seconds")
print(f"Max VRAM usage for SCA-Conv: {torch.cuda.max_memory_allocated(device) / 1024**2:.2f} MB")
print()
torch.cuda.empty_cache() if torch.cuda.is_available() else None





# a = torch.ones(batch, in_c, 1024,1024)
# print(f"Memory usage of tensor 'a': {a.element_size() * a.nelement()/1024**2} MB")
# unfold = nn.Unfold(kernel_size=(3, 3), padding=1, stride=1)
# b = unfold(a)
# print(f"Memory usage of tensor 'b': {b.element_size() * b.nelement()/1024**2} MB")
# print(f"Memory ratio b/a: {b.element_size() * b.nelement() / (a.element_size() * a.nelement())}")
# print(f"Length of unfolded tensor: {b.size(2)}")