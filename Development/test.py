import torch
import torch.nn as nn
from SCAConv_dev import SCAConv
from Depthwise_Separable_SCAC import SCAConv_DS
import time

batch = 1
in_c = 8
out_c = 16
a = torch.ones(batch, in_c, 512, 512)
conv1 = SCAConv(in_c, out_c, kernel_size=3, padding=1, stride=1)
conv2 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, stride=1)
conv3 = SCAConv_DS(in_c, out_c, kernel_size=3, padding=1, stride=1)

start_time = time.time()
output = conv1(a)
end_time = time.time()
print(f"Time taken for SCA-Conv: {end_time - start_time} seconds")

start_time = time.time()
output = conv3(a)
end_time = time.time()
print(f"Time taken for Depthwise Separable SCA-Conv: {end_time - start_time} seconds")

start_time = time.time()
output = conv2(a)
end_time = time.time()
print(f"Time taken for nn.Conv2d: {end_time - start_time} seconds")


# a = torch.ones(batch, in_c, 1024,1024)
# print(f"Memory usage of tensor 'a': {a.element_size() * a.nelement()/1024**2} MB")
# unfold = nn.Unfold(kernel_size=(3, 3), padding=1, stride=1)
# b = unfold(a)
# print(f"Memory usage of tensor 'b': {b.element_size() * b.nelement()/1024**2} MB")
# print(f"Memory ratio b/a: {b.element_size() * b.nelement() / (a.element_size() * a.nelement())}")
# print(f"Length of unfolded tensor: {b.size(2)}")