"""
CUDA & PyTorch Verification Script
Checks that PyTorch is installed correctly and can access CUDA GPUs.
"""

import sys


def check_pytorch():
    """Verify PyTorch installation and print version info."""
    print("=" * 60)
    print("  PyTorch & CUDA Verification")
    print("=" * 60)

    # --- Python ---
    print(f"\nPython version : {sys.version}")

    # --- PyTorch ---
    try:
        import torch
    except ImportError:
        print("\n[FAIL] PyTorch is NOT installed.")
        print("       Install it with:  pip install torch torchvision torchaudio")
        return False

    print(f"PyTorch version: {torch.__version__}")

    # --- CUDA availability ---
    cuda_available = torch.cuda.is_available()
    print(f"\nCUDA available : {cuda_available}")

    if not cuda_available:
        print("\n[WARN] CUDA is NOT available. PyTorch will run on CPU only.")
        print("       Possible causes:")
        print("         - No NVIDIA GPU present")
        print("         - NVIDIA drivers not installed / outdated")
        print("         - CPU-only PyTorch build installed")
        return False

    # --- CUDA details ---
    print(f"CUDA version   : {torch.version.cuda}")
    print(f"cuDNN version  : {torch.backends.cudnn.version()}")
    print(f"cuDNN enabled  : {torch.backends.cudnn.enabled}")

    # --- GPU devices ---
    device_count = torch.cuda.device_count()
    print(f"\nGPU count      : {device_count}")

    for i in range(device_count):
        props = torch.cuda.get_device_properties(i)
        total_mem = props.total_memory / (1024 ** 3)
        print(f"\n  GPU {i}: {props.name}")
        print(f"    Compute capability : {props.major}.{props.minor}")
        print(f"    Total memory       : {total_mem:.2f} GB")
        print(f"    Multi-processors   : {props.multi_processor_count}")

    # --- Current device ---
    current = torch.cuda.current_device()
    print(f"\nCurrent device : cuda:{current} ({torch.cuda.get_device_name(current)})")

    # --- Functional test ---
    print("\n" + "-" * 60)
    print("Running functional tests …")

    # Tensor creation on GPU
    try:
        x = torch.randn(3, 3, device="cuda")
        print(f"  [PASS] Created tensor on CUDA  →  device={x.device}")
    except Exception as e:
        print(f"  [FAIL] Could not create tensor on CUDA: {e}")
        return False

    # Basic operations
    try:
        y = torch.randn(3, 3, device="cuda")
        z = torch.matmul(x, y)
        print(f"  [PASS] Matrix multiply on CUDA →  result device={z.device}")
    except Exception as e:
        print(f"  [FAIL] Matrix multiply failed: {e}")
        return False

    # CPU ↔ GPU transfer
    try:
        cpu_tensor = z.cpu()
        gpu_tensor = cpu_tensor.cuda()
        print(f"  [PASS] CPU ↔ GPU transfer      →  {cpu_tensor.device} ↔ {gpu_tensor.device}")
    except Exception as e:
        print(f"  [FAIL] CPU ↔ GPU transfer failed: {e}")
        return False

    # Memory summary
    print("\n" + "-" * 60)
    allocated = torch.cuda.memory_allocated() / (1024 ** 2)
    reserved = torch.cuda.memory_reserved() / (1024 ** 2)
    print(f"GPU memory allocated: {allocated:.2f} MB")
    print(f"GPU memory reserved : {reserved:.2f} MB")

    print("\n" + "=" * 60)
    print("  All checks PASSED ✓")
    print("=" * 60)
    return True


if __name__ == "__main__":
    success = check_pytorch()
    sys.exit(0 if success else 1)
