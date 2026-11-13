import torch


def _format_bytes(num_bytes):
    for unit in ["B", "KB", "MB", "GB"]:
        if num_bytes < 1024:
            return f"{num_bytes:.2f} {unit}"
        num_bytes /= 1024
    return f"{num_bytes:.2f} TB"


def print_total_params(model):
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {total_params:,}")


def calc_model_size_bytes(model):
    """Actual memory used by the model's parameters and buffers, using their real dtypes."""
    params = sum(p.numel() * p.element_size() for p in model.parameters())
    buffers = sum(b.numel() * b.element_size() for b in model.buffers())
    return params + buffers


def _calc_model_memory_size(model, input_dtype=torch.float32):
    """Estimated training memory in GB (parameters + gradients + buffers) if everything were stored in input_dtype."""
    total_params = 0
    total_grads = 0
    for param in model.parameters():
        # Calculate total number of elements per parameter
        param_size = param.numel()
        total_params += param_size
        # Check if gradients are stored for this parameter
        if param.requires_grad:
            total_grads += param_size

    # Calculate buffer size (non-parameters that require memory)
    total_buffers = sum(buf.numel() for buf in model.buffers())

    # Size in bytes = (Number of elements) * (Size of each element in bytes)
    # We assume parameters and gradients are stored in the same type as input dtype
    element_size = torch.tensor(0, dtype=input_dtype).element_size()
    total_memory_bytes = (total_params + total_grads + total_buffers) * element_size

    # Convert bytes to gigabytes
    return total_memory_bytes / (1024**3)


def print_total_params_f32(model):
    print(f"float32 (PyTorch default): {_calc_model_memory_size(model, input_dtype=torch.float32):.2f} GB")


def print_total_params_bf16(model):
    print(f"bfloat16: {_calc_model_memory_size(model, input_dtype=torch.bfloat16):.2f} GB")


def print_model_stats(model):
    print_total_params(model)
    print(f"Model size (weights + buffers): {_format_bytes(calc_model_size_bytes(model))}")
