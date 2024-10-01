import torch
import torch.nn as nn
from SCAC_DS import SCAConv_DS, SCAConv_DS_OLD
from SCAC_DS_Attn import SCAC_DS_Attn
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

batch = 16
in_c = 8
out_c = 16
a = torch.ones(batch, in_c, 512, 512).to(device)
b = torch.ones(batch, in_c, 256, 256).to(device)

conv1 = SCAC_DS_Attn(in_c, out_c, kernel_size=3, padding=1, stride=1).to(device)
conv2 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, stride=1).to(device)
conv3 = SCAConv_DS(in_c, out_c, kernel_size=3, padding=1, stride=1).to(device)
conv4 = SCAConv_DS_OLD(in_c, out_c, kernel_size=3, padding=1, stride=1).to(device)

# warm up
output = conv2(b)
output = conv3(b)
output = conv4(b)
torch.cuda.empty_cache() if torch.cuda.is_available() else None
torch.cuda.reset_peak_memory_stats(device) if torch.cuda.is_available() else None

time.sleep(0.5)

start_time = time.time()
output = conv2(a)
output = conv2(b)
end_time = time.time()
print(f"Time taken for nn.Conv2d: {(end_time-start_time):.5f} seconds")
print(f"Max VRAM usage for nn.Conv2d: {torch.cuda.max_memory_allocated(device) / 1024**2:.2f} MB") if torch.cuda.is_available() else None
print()
torch.cuda.empty_cache() if torch.cuda.is_available() else None

time.sleep(0.5)
torch.cuda.reset_peak_memory_stats(device) if torch.cuda.is_available() else None

start_time = time.time()
output = conv3(a)
output = conv3(b)
end_time = time.time()
print(f"Time taken for SCAC-DS: {(end_time-start_time):.5f} seconds")
print(f"Max VRAM usage for SCAC-DS: {torch.cuda.max_memory_allocated(device) / 1024**2:.2f} MB") if torch.cuda.is_available() else None
print()
torch.cuda.empty_cache() if torch.cuda.is_available() else None

time.sleep(0.5)
torch.cuda.reset_peak_memory_stats(device) if torch.cuda.is_available() else None

start_time = time.time()
output = conv4(a)
output = conv4(b)
end_time = time.time()
print(f"Time taken for SCAC-DS-OLD: {(end_time-start_time):.5f} seconds")
print(f"Max VRAM usage for SCAC-DS-OLD: {torch.cuda.max_memory_allocated(device) / 1024**2:.2f} MB") if torch.cuda.is_available() else None
print()
torch.cuda.empty_cache() if torch.cuda.is_available() else None

# time.sleep(0.5)
# torch.cuda.reset_peak_memory_stats(device) if torch.cuda.is_available() else None

# start_time = time.time()
# output = conv1(a)
# end_time = time.time()
# print(f"Time taken for SCAC-DS-Attn: {(end_time-start_time):.5f} seconds")
# print(f"Max VRAM usage for SCAC-DS-Attn: {torch.cuda.max_memory_allocated(device) / 1024**2:.2f} MB") if torch.cuda.is_available() else None
# print()
# torch.cuda.empty_cache() if torch.cuda.is_available() else None



# a = torch.ones([3, 32*32])
# conv = nn.Conv1d(3,1,stride=52,kernel_size=256,padding=6)
# print(conv(a).shape)