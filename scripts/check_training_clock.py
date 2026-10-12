"""Ten-second GPU clock diagnostic; does not alter the training process."""

import time

import torch

matrix = torch.randn(1024, 1024, device="cuda", dtype=torch.bfloat16)
until = time.monotonic() + 10
while time.monotonic() < until:
    result = matrix @ matrix
    torch.cuda.synchronize()
print("GPU clock diagnostic finished")
