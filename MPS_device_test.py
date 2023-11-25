# MPS device test
import torch
if torch.backends.mps.is_available():
    mps_device = torch.device("mps")
    x =torch.ones(1, device=mps_device)
    print("MPS device is available\n", x)
else:
    print("MPS device is not available")