import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch

print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA version:", torch.version.cuda)
print("Number of GPUs:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("GPU name:", torch.cuda.get_device_name(0))

# Test a simple tensor operation on GPU
if torch.cuda.is_available():
    x = torch.rand(5, 3).cuda()
    y = torch.rand(5, 3).cuda()
    z = x + y
    print("GPU tensor addition:", z)
