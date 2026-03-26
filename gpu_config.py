"""
gpu_config.py — Centralized GPU optimization for RTX 4060 (Ada Lovelace, SM 8.9).

Call configure_gpu() once at program start to enable:
  - cuDNN benchmark mode  (fastest convolution algorithm for fixed-size inputs)
  - TF32 matmul + cuDNN   (free ~2× speedup on Ampere / Ada GPUs)
  - Default device selection

These are global PyTorch settings that affect all subsequent operations.
"""

import torch


def configure_gpu() -> str:
    """
    Apply GPU performance settings and return the device string.

    Returns
    -------
    str
        "cuda" if a GPU is available, else "cpu".
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if device == "cuda":
        # cuDNN benchmark: auto-tunes convolution algorithms for fixed input sizes.
        # All YOLO inputs are 640×640 so this gives a significant speedup.
        torch.backends.cudnn.benchmark = True

        # TF32 (TensorFloat-32): uses Tensor Cores for FP32 matmuls at ~2× speed
        # with negligible precision loss. Available on Ampere (SM 8.0+) and Ada.
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        props = torch.cuda.get_device_properties(0)
        vram_gb = round(props.total_memory / 1024**3, 1)

        print(f"""
╔══════════════════════════════════════════════════════════╗
║  GPU CONFIGURATION                                       ║
╠══════════════════════════════════════════════════════════╣
║  Device         : {props.name:<39}║
║  VRAM           : {str(vram_gb) + ' GB':<39}║
║  Compute        : SM {str(props.major) + '.' + str(props.minor):<36}║
║  cuDNN benchmark: {'ON':<39}║
║  TF32 matmul    : {'ON':<39}║
║  TF32 cuDNN     : {'ON':<39}║
║  PyTorch        : {torch.__version__:<39}║
║  CUDA           : {torch.version.cuda:<39}║
╚══════════════════════════════════════════════════════════╝
""")
    else:
        print("[gpu_config] No CUDA GPU found — running on CPU.")

    return device
