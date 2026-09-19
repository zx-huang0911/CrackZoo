# Source from Bash before installation or execution; all generated storage stays here.
CRACKZOO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export CRACKZOO_LOCAL="$CRACKZOO_ROOT/.local"
export TMPDIR="$CRACKZOO_LOCAL/tmp"
export XDG_CACHE_HOME="$CRACKZOO_LOCAL/cache"
export UV_CACHE_DIR="$XDG_CACHE_HOME/uv"
export PIP_CACHE_DIR="$XDG_CACHE_HOME/pip"
export TORCH_HOME="$XDG_CACHE_HOME/torch"
export MPLCONFIGDIR="$XDG_CACHE_HOME/matplotlib"
export CUDA_CACHE_PATH="$XDG_CACHE_HOME/cuda"
export TRITON_CACHE_DIR="$XDG_CACHE_HOME/triton"
export TORCHINDUCTOR_CACHE_DIR="$XDG_CACHE_HOME/torchinductor"
export PYTHONPYCACHEPREFIX="$XDG_CACHE_HOME/pycache"
export WANDB_DIR="$CRACKZOO_LOCAL/wandb"
export WANDB_CACHE_DIR="$XDG_CACHE_HOME/wandb"
export WANDB_CONFIG_DIR="$CRACKZOO_LOCAL/config/wandb"
export WANDB_MODE=disabled
export PIP_DISABLE_PIP_VERSION_CHECK=1
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
mkdir -p "$TMPDIR" "$PIP_CACHE_DIR" "$TORCH_HOME" "$MPLCONFIGDIR" \
    "$CUDA_CACHE_PATH" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" \
    "$PYTHONPYCACHEPREFIX" "$WANDB_DIR" "$WANDB_CACHE_DIR" "$WANDB_CONFIG_DIR"
