"""Fail-closed reproducibility configuration for the fixed CUDA environment."""
import os
import random

def configure(torch,seed):
    if os.environ.get('CUBLAS_WORKSPACE_CONFIG')!=':4096:8' or os.environ.get('PYTHONHASHSEED')!=str(seed):
        raise ValueError('Deterministic process environment not configured before startup')
    if torch.cuda.is_initialized():raise ValueError('Configure runtime before CUDA initialization')
    import numpy as np
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True,warn_only=False)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    return {'torch_version':torch.__version__,'cuda_version':torch.version.cuda,
            'deterministic_algorithms':torch.are_deterministic_algorithms_enabled(),
            'warn_only':torch.is_deterministic_algorithms_warn_only_enabled(),
            'cudnn_benchmark':torch.backends.cudnn.benchmark,'cudnn_deterministic':torch.backends.cudnn.deterministic,
            'matmul_tf32':torch.backends.cuda.matmul.allow_tf32,'cudnn_tf32':torch.backends.cudnn.allow_tf32,
            'cublas_workspace':os.environ['CUBLAS_WORKSPACE_CONFIG'],'pythonhashseed':os.environ['PYTHONHASHSEED'],'seed':seed}
