import sys
print(sys.executable)
import torch
print("CUDA available:", torch.cuda.is_available())
print("Torch version:", torch.__version__)
print("CUDA version:", torch.version.cuda)
print("GPU Name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU found")
