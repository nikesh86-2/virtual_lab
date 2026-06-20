import torch
import threading

_gpu_lock = threading.Lock()
_next_gpu = 0

def get_next_gpu():
    global _next_gpu

    with _gpu_lock:
        device_count = torch.cuda.device_count()
        gpu_id = _next_gpu % max(1, device_count)
        _next_gpu += 1

    return gpu_id


def set_gpu_for_agent(prefer=None):
    """
    Assigns a GPU to current process/thread.
    """

    if not torch.cuda.is_available():
        return "cpu"

    gpu_id = prefer if prefer is not None else get_next_gpu()

    torch.cuda.set_device(gpu_id)

    return f"cuda:{gpu_id}"

import gc

def clear_gpu():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()
