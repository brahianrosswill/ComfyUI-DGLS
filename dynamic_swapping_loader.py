PROFILE = False

import comfy.sd
import folder_paths
import datetime
from functools import wraps
from torch import nn
import time
import torch
import gc
import threading
import warnings
import psutil
import types
import torch._dynamo
from functools import partial
import torch.utils.checkpoint
from collections import defaultdict
import bitsandbytes as bnb
import queue
from concurrent.futures import ThreadPoolExecutor

if PROFILE == True:
    try:
        import torch.profiler
        import torch.cuda.nvtx as nvtx
    except ImportError:
        PROFILE = False
        nvtx = None


"""
Dynamic GPU Layer Swapping for ComfyUI by obisin
======================================
Enhanced inference with automatic memory optimization for diffusion models.
Supports Cosmos, Flux, Wan2.1, Wan2.2, HunyuanVideo, and generic transformer blocks.

PERFORMANCE DESIGN NOTE:
This file uses module-level functions with local variable binding for optimal performance.
Hot path functions bind frequently-accessed attributes to locals at function start (LOAD_FAST)
to avoid Python's attribute lookup overhead (LOAD_ATTR). Call chains are minimized to 
reduce function call overhead.

Architecture:
- Globals: Intentionally used for performance (documented below)
- Hot functions: Bind globals to locals once per call
- Minimal call depth: Direct function calls without abstraction layers
- Factory pattern: Clean call sites with fast execution

CRITICAL PATH: Functions marked "HOT PATH" - profile before changing!

Author: obisin
"""

class Args:
    def __init__(self):
        self.dynamic_swapping = True
        self.auto_allocate_layers = False
        self.pin_memory = False
        self.initial_gpu_layers = 1  # Number of initial layers to keep permanently on GPU
        self.final_gpu_layers = 1    # Number of final layers to keep permanently on GPU
        self.gpu_layers = None       # Comma-separated list of layer indices to keep permanently on GPU
        self.prefetch = 0            # Number of layers to prefetch ahead (0=off, 1=single layer, 2+=multiple layers)
        self.cuda_streams = False    # Enable CUDA streams for copy-compute overlap (needs more VRAM)
        self.cast_target = None      # Cast FROM dtype TO dtype at start-up (e.g., f32 bf16)
        self.verbose = False         # Enable verbose output with detailed timing and transfer information
        self.cpu_threading = False  # Enable CPU threading for async CPU transfers

args = Args()
TOOLTIPS = {
    "dynamic_swapping": "Smart dynamic layer swapping between GPU and CPU for optimal performance",
    "initial_gpu_layers": "Number of initial layers to keep permanently on GPU. If not specified, uses reasonable defaults based on estimated VRAM",
    "final_gpu_layers": "Number of final layers to keep permanently on GPU",
    "gpu_layers": "Comma-separated list of layer indices to keep permanently on GPU (e.g., '0,1,2,14,18,19,20,21,22'). Overrides initial_gpu_layers and final_gpu_layers",
    "prefetch": "Number of layers to prefetch ahead of single layer transfer",
    "cuda_streams": "Enable CUDA streams for copy-compute overlap (needs more VRAM)",
    "cpu_threading": "Enable CPU threading for async CPU transfers (May cause instability on some systems)",
    "cast_target": "Cast FROM dtype TO dtype at start-up (e.g., f32 bf16) choices=[f32, bf16, f16, f8_e4m3, f8_e5m2, nf4, fp4] WARNING: don’t cast to f8/fp4 here unless your kernels support it",
    "verbose": "Enable verbose output with detailed timing and transfer information",
    "pin_memory": "Pin CPU memory for faster async GPU transfers (uses 2x CPU RAM for swappable layers)",

}

# =============================================================================
# GLOBAL VARIABLES
# =============================================================================
# Global variables for transfer thread
transfer_queue = None
transfer_thread = None
transfer_active = False
original_functions = {}

global add_smart_swapping_to_layer
swap_stats = {'to_gpu': 0, 'to_cpu': 0}
transfer_events = {}

transfer_stats = {
    'to_gpu_times': [], 'to_cpu_times': [], 'to_gpu_speeds': [], 'to_cpu_speeds': [],
    'current_step_gpu_times': [], 'current_step_cpu_times': [],
    'current_step_gpu_speeds': [], 'current_step_cpu_speeds': []}

overlap_stats = {
    'transfer_start_times': {},  # {layer_idx: start_time}
    'transfer_end_times': {},    # {layer_idx: end_time}
    'compute_start_times': {},   # {layer_idx: start_time}
    'compute_end_times': {},     # {layer_idx: end_time}
    'overlaps': []               # List of overlap measurements
}

device_cache = None
packed_layers = {}
layer_sizes_mb = {}

pending_gpu_transfers = {}  # {layer_idx: event}
pending_cpu_transfers = {}  # {layer_idx: event}

gpu_layer_copies = {}  # {layer_idx: gpu_layer_copy}

layers = None

cpu_swappable_layers = set()
gpu_resident_layers = set()
gpu_occupied_swappable = set()
pending_pin = set()

GPU_DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
CPU_DEVICE = 'cpu'

print(f"Device configuration: GPU={GPU_DEVICE}, CPU={CPU_DEVICE}")
casting_handler = None

profiler_active = False
swappable_list = None

# =============================================================================
# DEBUG FUNCTIONS by obisin
# =============================================================================

def _layer_pinning_stats(layer: nn.Module):

    def _tensor_nbytes(t: torch.Tensor) -> int:
        try:
            return t.numel() * t.element_size()
        except Exception:
            return 0

    stats = {
        "pinned_params": 0, "total_params": 0,
        "pinned_buffers": 0, "total_buffers": 0,
        "pinned_bytes": 0,
    }
    for _, p in layer.named_parameters(recurse=True):
        if p is None:
            continue
        stats["total_params"] += 1
        if p.device.type == "cpu" and hasattr(p, "is_pinned") and p.is_pinned():
            stats["pinned_params"] += 1
            stats["pinned_bytes"] += _tensor_nbytes(p)
    for _, b in layer.named_buffers(recurse=True):
        if b is None:
            continue
        stats["total_buffers"] += 1
        if b.device.type == "cpu" and hasattr(b, "is_pinned") and b.is_pinned():
            stats["pinned_buffers"] += 1
            stats["pinned_bytes"] += _tensor_nbytes(b)
    return stats




def debug_print_pinning_report(layers_list, label: str = "", top_k: int = 8):
    return
    def _summarize_pinning(layers_list, top_k: int = 8):
        total_bytes = 0
        per_layer = []
        for i, L in enumerate(layers_list or []):
            s = _layer_pinning_stats(L)
            if s["pinned_bytes"] > 0:
                per_layer.append((i, s["pinned_bytes"], s))
            total_bytes += s["pinned_bytes"]

        per_layer.sort(key=lambda x: x[1], reverse=True)
        return total_bytes, per_layer[:top_k]

    total_bytes, top_layers = _summarize_pinning(layers_list, top_k)
    gb = total_bytes / (1024**3)
    print(f"[PIN-REPORT] {label}  total_pinned≈{gb:.2f} GB")

    for i, bytes_i, s in top_layers:
        gb_i = bytes_i / (1024**3)
        detail = (
            f"layer={i}  pinned≈{gb_i:.3f} GB  "
            f"params {s['pinned_params']}/{s['total_params']}  "
            f"buffers {s['pinned_buffers']}/{s['total_buffers']}"
        )
        print(f"  └─ {detail}")

def debug_assert_no_gpu_params(layers_list, label: str = "", ignore: set | None = None):
    return
    ignore = ignore or set()
    leaks = []
    for i, L in enumerate(layers_list or []):
        if i in ignore:
            continue
        for name, p in L.named_parameters(recurse=True):
            if p is not None and p.device.type == "cuda":
                leaks.append((i, "param", name, str(p.shape)))
        for name, b in L.named_buffers(recurse=True):
            if b is not None and b.device.type == "cuda":
                leaks.append((i, "buffer", name, str(b.shape)))
    if leaks:
        print(f"[GPU-LEAK] {label}  Found {len(leaks)} tensors still on CUDA:")
        for (i, kind, name, shape) in leaks[:50]:
            print(f"  └─ layer={i}  {kind}={name}  shape={shape}")
    else:
        print(f"[GPU-LEAK] {label}  OK (no params/buffers on CUDA)")


def debug_comfy_casting_memory(layer_idx, operation="unknown"):
    """Track memory around ComfyUI's casting operations"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(0)
        reserved = torch.cuda.memory_reserved(0)
        free_vram, total_vram = torch.cuda.mem_get_info(0)

        print(f"CASTING DEBUG Layer {layer_idx} {operation}:")
        print(f"  Allocated: {allocated / 1024 ** 3:.3f} GB")
        print(f"  Reserved: {reserved / 1024 ** 3:.3f} GB")
        print(f"  Free VRAM: {free_vram / 1024 ** 3:.3f} GB")
        print(f"  Fragmentation: {(reserved - allocated) / 1024 ** 3:.3f} GB")


def debug_comfy_memory_state():
    """Debug ComfyUI's internal memory management"""
    import comfy.model_management

    if torch.cuda.is_available():
        # ComfyUI's view of memory
        device = comfy.model_management.get_torch_device()
        free_memory = comfy.model_management.get_free_memory(device)

        # PyTorch's view of memory
        allocated = torch.cuda.memory_allocated(0)
        reserved = torch.cuda.memory_reserved(0)
        free_vram, total_vram = torch.cuda.mem_get_info(0)

        print(f"\n=== MEMORY STATE DEBUG ===")
        print(f"ComfyUI free memory: {free_memory / 1024 ** 3:.3f} GB")
        print(f"PyTorch allocated: {allocated / 1024 ** 3:.3f} GB")
        print(f"PyTorch reserved: {reserved / 1024 ** 3:.3f} GB")
        print(f"CUDA free/total: {free_vram / 1024 ** 3:.3f} / {total_vram / 1024 ** 3:.3f} GB")

        # Check for loaded models in ComfyUI
        loaded_models = comfy.model_management.loaded_models()
        print(f"ComfyUI loaded models: {len(loaded_models)}")
        for i, model in enumerate(loaded_models):
            print(f"  Model {i}: {model.loaded_size() / 1024 ** 3:.3f} GB")

        print("==========================\n")

def debug_layer_locations(layers, device_cache):
    """Debug exact layer locations and memory usage"""
    return
    print(f"\n=== LAYER LOCATION DEBUG ===")

    if layers is None or device_cache is None:
        print("ERROR: layers/device cache is None - debug called before layer setup")
        return

    total_gpu_mem = 0
    total_cpu_mem = 0

    for i, layer in enumerate(layers):
        try:
            # Get actual device
            actual_device = next(layer.parameters()).device
        except StopIteration:
            actual_device = torch.device('cpu')  # No parameters

        # Get cached device
        cached_device = device_cache.get_device(i)

        # Calculate layer memory
        layer_mem = sum(p.numel() * p.element_size() for p in layer.parameters())
        layer_mem_mb = layer_mem / (1024 * 1024)

        # Track totals
        if actual_device.type == 'cuda':
            total_gpu_mem += layer_mem_mb
        else:
            total_cpu_mem += layer_mem_mb

        # Status indicators
        status = "RESIDENT" if i in gpu_resident_layers else "SWAPPABLE"
        mismatch = "MISMATCH" if actual_device != cached_device else "OK"

        print(f"Layer {i:2d}: {actual_device} | Cache: {cached_device} | "
              f"{layer_mem_mb:6.1f}MB | {status} | {mismatch}")

    print(f"\nTOTAL IN GPU MEMORY: {total_gpu_mem:.1f}MB ({total_gpu_mem / 1024:.2f}GB)")
    print(f"TOTAL IN CPU MEMORY: {total_cpu_mem:.1f}MB ({total_cpu_mem / 1024:.2f}GB)")
    print(f"GPU ALLOCATED: {torch.cuda.memory_allocated(0) / 1024 ** 3:.3f}GB")
    print("============================\n")


# =============================================================================
# DYNAMIC LAYER SWAPPING INFERENCE for Comfy-UI - by obisin
# =============================================================================



class LayerDeviceCache:
    def __init__(self, model, layers):  # Add layers parameter
        self.cache = {}
        self.dirty = set()
        # Initialize cache using the layers list
        for i, layer in enumerate(layers):
            self.cache[i] = self.get_layer_device(layer)

    def get_layer_device(self, layer):
        """Your existing function"""
        try:
            return next(layer.parameters()).device
        except StopIteration:
            return torch.device('cpu')

    def get_device(self, layer_idx):  # Simplified - no need for model parameter
        """Fast cached lookup"""
        if layer_idx in self.dirty:
            if layers is not None and layer_idx < len(layers):
                self.cache[layer_idx] = self.get_layer_device(layers[layer_idx])
                self.dirty.remove(layer_idx)
            else:
                return torch.device('cpu')  # Safe fallback
        return self.cache.get(layer_idx, torch.device('cpu'))

    def mark_moved(self, layer_idx, new_device):
        """Update cache when we move a layer"""
        self.cache[layer_idx] = new_device


def fix_inference_tensor_parameters(layer):
    """
    Ensure every parameter/buffer in `layer` has a proper version counter.
    Works recursively and handles dotted names by updating the owning submodule.
    """
    def needs_version_fix(t: torch.Tensor) -> bool:
        try:
            _ = t._version
            return False
        except Exception:
            return True

    fixed = False
    with torch.inference_mode(False), torch.no_grad():
        # 1) Parameters (recursive)
        for name, p in list(layer.named_parameters(recurse=True)):
            if p is None:
                continue
            if needs_version_fix(p):
                owner, leaf = _owner_and_key(layer, name)  # e.g. "img_attn.qkv.weight"
                new_p = torch.nn.Parameter(p.detach().contiguous(), requires_grad=False)
                owner._parameters[leaf] = new_p
                fixed = True

        # 2) Buffers (recursive)
        for name, b in list(layer.named_buffers(recurse=True)):
            if b is None:
                continue
            if needs_version_fix(b):
                owner, leaf = _owner_and_key(layer, name)
                new_b = b.detach().contiguous()
                owner._buffers[leaf] = new_b  # keeps persistence semantics
                fixed = True

    return fixed

def get_cached_layer_size_mb(idx):
    size = layer_sizes_mb.get(idx, 0)
    if size == 0:
        print(f"WARNING: Layer {idx} size not found in cache")
    return size

def _owner_and_key(module: nn.Module, dotted: str):
    """Resolve submodule and leaf key for dotted parameter/buffer names."""
    if '.' in dotted:
        prefix, leaf = dotted.rsplit('.', 1)
        owner = module.get_submodule(prefix)
        return owner, leaf
    else:
        return module, dotted

def pin_cpu_memory(layer, idx):
    """Pin CPU tensors for faster CPU→GPU copies (still CPU)."""
    MIN_MB = int(getattr(args, 'pin_min_mb', 128))
    with torch.no_grad():
        # parameters
        for name, p in list(layer.named_parameters(recurse=True)):
            if p is None or p.device.type != 'cpu' or p.is_pinned():
                continue
            if get_cached_layer_size_mb(idx) <= MIN_MB:
                return
            pinned = p.detach().contiguous().pin_memory()
            owner, leaf = _owner_and_key(layer, name)
            owner._parameters[leaf] = nn.Parameter(pinned, requires_grad=False)

        # buffers
        for name, b in list(layer.named_buffers(recurse=True)):
            if b is None or b.device.type != 'cpu' or b.is_pinned():
                continue
            if get_cached_layer_size_mb(idx) <= MIN_MB:
                return
            pinned = b.detach().contiguous().pin_memory()
            owner, leaf = _owner_and_key(layer, name)
            owner._buffers[leaf] = pinned


def unpin_cpu_memory(layer):
    """Convert pinned CPU params/buffers back to regular CPU (releases Shared GPU memory)."""
    with torch.no_grad():
        # parameters
        for name, p in list(layer.named_parameters(recurse=True)):
            if p is not None and p.device.type == 'cpu' and p.is_pinned():
                owner, leaf = _owner_and_key(layer, name)
                new_cpu = p.detach().contiguous()  # non-pinned CPU
                owner._parameters[leaf] = nn.Parameter(new_cpu, requires_grad=False)
        # buffers
        for name, b in list(layer.named_buffers(recurse=True)):
            if b is not None and b.device.type == 'cpu' and b.is_pinned():
                owner, leaf = _owner_and_key(layer, name)
                owner._buffers[leaf] = b.detach().contiguous()


def _prepare_swappable_layer(layer: nn.Module, idx: int, pin: bool):
    """
    One-time prep for a swappable layer:
      • Move whole layer to GPU so BUFFERS live on GPU
      • Make a single CPU master for each PARAM (optional pinned)
      • Rebind param .data to the CPU master (module remains param-owning)
      • No new nn.Parameter objects created, only .data rebinds
    """
    if getattr(layer, "_dgls_swappable_ready", False):
        return

    # Buffers -> GPU (small), we will only swap params
    layer.to(GPU_DEVICE)

    layer._cpu_master_data = {}    # { dotted_name: CPU tensor }
    layer._gpu_slots = {}          # { dotted_name: CUDA tensor } (allocated lazily per stage/evict)

    with torch.no_grad():
        for pname, p in layer.named_parameters(recurse=True):
            if p is None:
                continue
            # single CPU master copy (one-time)
            cpu_master = p.data.to("cpu", copy=True)
            if pin:
                try:
                    cpu_master = cpu_master.pin_memory()
                except Exception:
                    pass
            # rebind module param .data back to CPU master (keeps module owning the same Parameter wrapper)
            owner, leaf = _owner_and_key(layer, pname)
            # owner._parameters[leaf].data = cpu_master
            _reassign_param(owner, leaf, cpu_master)
            layer._cpu_master_data[pname] = cpu_master

    layer._dgls_swappable_ready = True


# def _reassign_param(owner: nn.Module, leaf: str, tensor: torch.Tensor):
#     # guarantees a fresh Parameter that has a version counter
#     with torch.inference_mode(False), torch.no_grad():
#         owner._parameters[leaf] = nn.Parameter(tensor.detach().contiguous(), requires_grad=False)

def _reassign_param(owner: nn.Module, leaf: str, tensor: torch.Tensor):
    # fast-path: already bound to the same storage
    cur = owner._parameters.get(leaf, None)
    if (cur is not None
        and isinstance(cur, torch.nn.Parameter)
        and cur.shape == tensor.shape
        and cur.dtype == tensor.dtype
        and cur.device == tensor.device
        and cur.data_ptr() == tensor.data_ptr()):
        return  # nothing to do

    # otherwise build a fresh Parameter with a version counter
    with torch.inference_mode(False), torch.no_grad():
        owner._parameters[leaf] = nn.Parameter(
            tensor if tensor.is_contiguous() else tensor.contiguous(),
            requires_grad=False
        )


def stage_to_gpu(layer_idx):
    layer = layers[layer_idx]
    if not hasattr(layer, "_cpu_master_data"):
        _prepare_swappable_layer(layer, layer_idx, pin=args.pin_memory)

    if not hasattr(layer, "_gpu_slots"):
        layer._gpu_slots = {}

    masters = layer._cpu_master_data
    with torch.no_grad():
        for pname, cpu_master in masters.items():
            gpu_t = layer._gpu_slots.get(pname)
            if (gpu_t is None or (not gpu_t.is_cuda)
                or gpu_t.shape != cpu_master.shape or gpu_t.dtype != cpu_master.dtype):
                gpu_t = cpu_master.to(GPU_DEVICE, copy=True, non_blocking=True).contiguous()
                layer._gpu_slots[pname] = gpu_t
            else:
                gpu_t.copy_(cpu_master, non_blocking=True)

            owner, leaf = _owner_and_key(layer, pname)
            _reassign_param(owner, leaf, gpu_t)


def rebind(layer_idx):
    layer = layers[layer_idx]
    masters = getattr(layer, "_cpu_master_data", None)
    if not masters:
        return

    with torch.no_grad():
        for pname, cpu_master in masters.items():
            owner, leaf = _owner_and_key(layer, pname)
            _reassign_param(owner, leaf, cpu_master)   # ← back to CPU Parameter

        slots = getattr(layer, "_gpu_slots", None)
        if slots:
            for k, t in list(slots.items()):
                if isinstance(t, torch.Tensor) and t.is_cuda:
                    del t
                slots.pop(k, None)


def calculate_auto_gpu_layers(layers, args):
    """Auto-select GPU layers allocation logic with safety mechanisms"""

    layer_overhead_mult = 1.05

    device = comfy.model_management.get_torch_device()

    # Step 1: Cleanup existing models
    comfy.model_management.cleanup_models_gc()

    # Step 2: Calculate total memory required with safety multiplier
    total_model_size = sum(comfy.model_management.module_size(layer) for layer in layers)
    memory_required_with_safety = total_model_size * layer_overhead_mult
    minimum_memory_required = comfy.model_management.minimum_inference_memory()
    extra_mem = max(minimum_memory_required,
                    memory_required_with_safety + comfy.model_management.extra_reserved_memory())

    # Step 3: Free memory if needed (ComfyUI's approach)
    initial_free_memory = comfy.model_management.get_free_memory(device)
    if initial_free_memory < memory_required_with_safety + minimum_memory_required:
        comfy.model_management.free_memory(memory_required_with_safety + extra_mem, device)
        print("Freed other models to make space")

    # Step 4: Secondary safety check
    current_free_mem = comfy.model_management.get_free_memory(device)
    if current_free_mem < minimum_memory_required:
        comfy.model_management.free_memory(minimum_memory_required, device)
        current_free_mem = comfy.model_management.get_free_memory(device)

    # Step 5: Platform-specific memory ratio
    if comfy.model_management.is_nvidia():
        min_ratio = 0.0
    else:
        min_ratio = 0.4

    # Step 6: budget calculation
    layer_memory_budget = max(
        128 * 1024 * 1024,
        (current_free_mem - minimum_memory_required),
        min(
            current_free_mem * min_ratio,
            current_free_mem - minimum_memory_required
        )
    )

    # Get currently loaded memory and subtract it
    loaded_memory = sum(model.loaded_size() for model in comfy.model_management.loaded_models())
    layer_memory_budget = max(0.1, layer_memory_budget - loaded_memory)

    # Step 7: Calculate layer sizes and sort (largest first)
    layer_allocation_data = []
    for i, layer in enumerate(layers):
        layer_size = comfy.model_management.module_size(layer)
        layer_allocation_data.append((layer_size, i))

    layer_allocation_data.sort(reverse=True, key=lambda x: x[0])

    # Step 8: Apply overhead from user settings
    if layer_allocation_data:
        max_layer_size = layer_allocation_data[0][0]
        overhead = max_layer_size * layer_overhead_mult * (args.prefetch + 1)

        if args.cpu_threading:
            overhead += max_layer_size * 0.25 * (args.prefetch + 1)

        if args.cuda_streams:
            overhead += max_layer_size * 0.25 * (args.prefetch + 1)

        layer_memory_budget = max(128 * 1024 * 1024, layer_memory_budget - overhead)

    # Step 9: Greedy allocation
    allocated_memory = 0
    selected_gpu_layers = set()

    for layer_size, layer_idx in layer_allocation_data:
        if allocated_memory + layer_size <= layer_memory_budget:
            allocated_memory += layer_size
            selected_gpu_layers.add(layer_idx)
        else:
            break

    # Step 10: Final validation
    if allocated_memory > current_free_mem * 0.85:  # Additional safety check
        print("WARNING: High memory allocation detected, reducing selection")
        # Remove smallest selected layers until under 85% of free memory
        selected_list = [(layer_allocation_data[i][0], layer_allocation_data[i][1]) for i in
                         range(len(layer_allocation_data)) if layer_allocation_data[i][1] in selected_gpu_layers]
        selected_list.sort()  # Smallest first for removal

        while allocated_memory > current_free_mem * 0.8 and selected_list:
            removed_size, removed_idx = selected_list.pop(0)
            selected_gpu_layers.remove(removed_idx)
            allocated_memory -= removed_size

    if args.verbose:
        print(f"Free VRAM: {current_free_mem / (1024 ** 2):.0f}MB")
        print(f"Final budget: {layer_memory_budget / (1024 ** 2):.0f}MB")
        print(f"Auto-selected GPU layers: {sorted(selected_gpu_layers)}")
        print(f"Final allocated: {allocated_memory / (1024 ** 2):.0f}MB")

    return selected_gpu_layers


def calculate_needed_layers(layer_idx, prefetch):
    needed = set()
    needed.add(layer_idx)

    try:
        current_pos = swappable_index.get(layer_idx)
        if current_pos is None:
            raise ValueError

        # cap prefetch to avoid useless wraps
        prefetch = min(prefetch, len(swappable_list) - 1) if swappable_list else 0

        for i in range(1, prefetch + 1):
            next_pos = (current_pos + i) % len(swappable_list)
            needed.add(swappable_list[next_pos])

    except ValueError:
        for i in range(1, prefetch + 1):
            prefetch_idx = layer_idx + i
            if prefetch_idx >= len(layers):
                wrap_offset = prefetch_idx - len(layers)
                if wrap_offset < len(swappable_list):
                    prefetch_idx = swappable_list[wrap_offset]
                else:
                    continue
            if prefetch_idx in cpu_swappable_layers:
                needed.add(prefetch_idx)


    if swappable_list:
        # pos_map = {v: i for i, v in enumerate(swappable_list)}  # O(n) once
        # base = pos_map.get(layer_idx)
        # if base is not None:
        #     n = len(swappable_list)
        #     def order(idx):
        #         pos = pos_map.get(idx)
        #         return ((pos - base) % n) if pos is not None else (n + idx)
        #     return sorted(needed, key=order)
        base = swappable_index.get(layer_idx)
        if base is not None and swappable_list:
            n = len(swappable_list)

            def order(idx):
                pos = swappable_index.get(idx)
                return ((pos - base) % n) if pos is not None else (n + idx)

            return sorted(needed, key=order)

    # fallback: stable numeric order
    return sorted(needed)


def cleanup_excess_layers(keep_layers):
    """Remove layers from GPU that are not in keep_layers set - unified threading"""

    global gpu_occupied_swappable
    layers_to_remove = gpu_occupied_swappable - set(keep_layers)

    if not layers_to_remove:
        return 0

    def unified_cleanup_transfer():
        global gpu_occupied_swappable
        cleaned_count = 0
        for idx in layers_to_remove:
            if args.verbose:
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()

            rebind(idx)
            gpu_layer_copies.pop(idx, None)

            device_cache.mark_moved(idx, torch.device('cpu'))
            swap_stats['to_cpu'] += 1
            cleaned_count += 1

            if args.verbose:
                end_event.record()
                end_event.synchronize()
                transfer_time = start_event.elapsed_time(end_event)
                add_smart_swapping_to_layer.layer_transfer_times[idx].append(transfer_time)

        gpu_occupied_swappable -= layers_to_remove
        return cleaned_count

    if args.cpu_threading and hasattr(add_smart_swapping_to_layer, 'cpu_thread_pool'):
        future = add_smart_swapping_to_layer.cpu_thread_pool.submit(unified_cleanup_transfer)
        # Store future but don't wait - cleanup happens in background
        add_smart_swapping_to_layer.transfer_futures['cleanup'] = future
        if len(add_smart_swapping_to_layer.transfer_futures) > 40:
            oldest_keys = list(add_smart_swapping_to_layer.transfer_futures.keys())[:10]
            for key in oldest_keys:
                del add_smart_swapping_to_layer.transfer_futures[key]
        return len(layers_to_remove)
    else:
        # Direct execution when threading disabled
        return unified_cleanup_transfer()


def fetch_missing_layers(needed_layers, current_layer_idx=None):
    """Ensure all needed layers are on GPU - unified threading with selective waiting"""
    global gpu_occupied_swappable

    def _fetch_operation():
        layers_to_fetch = set(needed_layers) - gpu_occupied_swappable

        fetched_count = 0
        transfer_start = time.perf_counter()

        if args.verbose:
            for idx in layers_to_fetch:
                overlap_stats['transfer_start_times'][idx] = transfer_start

        def transfer_single_layer(idx):
            """Nested function to eliminate repetitive transfer pattern"""
            nonlocal fetched_count
            if args.verbose:
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()

            if idx in gpu_layer_copies:
                gpu_layer_copies.pop(idx, None)

            # One-way
            stage_to_gpu(idx)
            gpu_layer_copies[idx] = True

            if args.verbose:
                transfer_event = torch.cuda.Event()
                transfer_event.record()

                if hasattr(add_smart_swapping_to_layer, 'transfer_events'):
                    add_smart_swapping_to_layer.transfer_events[idx] = transfer_event
                    if len(add_smart_swapping_to_layer.transfer_events) > 40:
                        oldest_keys = list(add_smart_swapping_to_layer.transfer_events.keys())[:10]
                        for key in oldest_keys:
                            del add_smart_swapping_to_layer.transfer_events[key]

            device_cache.mark_moved(idx, torch.device('cuda'))
            swap_stats['to_gpu'] += 1
            fetched_count += 1

            if args.verbose:
                end_event.record()
                end_event.synchronize()
                transfer_time = start_event.elapsed_time(end_event)
                add_smart_swapping_to_layer.layer_transfer_times[idx].append(transfer_time)
                if len(add_smart_swapping_to_layer.layer_transfer_times[idx]) > 40:
                    add_smart_swapping_to_layer.layer_transfer_times[idx] = add_smart_swapping_to_layer.layer_transfer_times[idx][-40:]

        def unified_gpu_transfer():
            nonlocal fetched_count
            if layers_to_fetch:
                if args.cuda_streams and hasattr(add_smart_swapping_to_layer, 'fetch_stream'):
                    with torch.cuda.stream(add_smart_swapping_to_layer.fetch_stream):
                        for idx in layers_to_fetch:
                            transfer_single_layer(idx)
                    gpu_occupied_swappable.update(layers_to_fetch)
                else:
                    for idx in layers_to_fetch:
                        transfer_single_layer(idx)
                    gpu_occupied_swappable.update(layers_to_fetch)
            return fetched_count

        # Use threading if enabled
        if args.cpu_threading and hasattr(add_smart_swapping_to_layer, 'cpu_thread_pool'):
            future = add_smart_swapping_to_layer.cpu_thread_pool.submit(unified_gpu_transfer)

            # Wait only if we need the current layer immediately
            if current_layer_idx is not None and current_layer_idx in layers_to_fetch:
                fetched_count = future.result()  # Block until current layer ready
            else:
                fetched_count = len(layers_to_fetch)  # Estimate
        else:
            fetched_count = unified_gpu_transfer()

        # Wait only for current layer if specified
        if current_layer_idx is not None and current_layer_idx in layers_to_fetch:
            while True:
                try:
                    test_param = next(layers[current_layer_idx].parameters())
                    if test_param.device.type == 'cuda':
                        break
                except StopIteration:
                    break
        if args.verbose:
            # Record transfer end times
            transfer_end = time.perf_counter()
            for idx in layers_to_fetch:
                overlap_stats['transfer_end_times'][idx] = transfer_end

        return fetched_count

    if not hasattr(fetch_missing_layers, 'step_counter'):
        fetch_missing_layers.step_counter = 0
    fetch_missing_layers.step_counter += 1

    return _fetch_operation()

def add_smart_swapping_to_layer(layer, layer_idx, layers_list, gpu_resident_layers, cpu_swappable_layers):
    """Add swapping capability with ComfyUI coordination"""

    # Prevent double-wrapping
    if getattr(layer, '_dgls_swapped', False):
        return

    # Always use original forward, not current forward
    if not hasattr(layer, '_original_forward'):
        layer._original_forward = layer.forward

    if not getattr(layer, "_dgls_fixed", False):
        if fix_inference_tensor_parameters(layer):
            pass  # fixed something
        layer._dgls_fixed = True

    global layers, device_cache
    layers = layers_list
    # original_forward = layer.forward
    original_forward = layer._original_forward if hasattr(layer, '_original_forward') else layer.forward

    # Cache layer type checks
    layer_type = type(layer).__name__
    layer._dgls_is_flux = 'DoubleStream' in layer_type or 'SingleStream' in layer_type
    layer._dgls_kw_only = any(pattern in layer_type for pattern in ['QwenImage', 'HiDream'])

    @wraps(original_forward)
    def swapped_forward(*fwd_args, **kwargs):
        # cached_device = device_cache.get_device(layer_idx)
        def _whole_forward_operation():
            if args.verbose:
                layer_compute_start = time.perf_counter()
                overlap_stats['compute_start_times'][layer_idx] = layer_compute_start

            # Detect new step when we hit layer 0
            if layer_idx == add_smart_swapping_to_layer.first_swappable:
                add_smart_swapping_to_layer.current_step += 1
                add_smart_swapping_to_layer.calls_this_step = 0
                if args.verbose:
                    print(f" New sampling step {add_smart_swapping_to_layer.current_step}")

                    # if add_smart_swapping_to_layer.current_step % 3 == 0:
                    #     analyze_layer_performance()

            if layer_idx in cpu_swappable_layers:
                if PROFILE:
                    nvtx.range_push(f"Smart_Management_L{layer_idx}")
                try:
                    current_device = device_cache.get_device(layer_idx)
                    layer_already_on_gpu = (current_device.type == 'cuda')

                    if not layer_already_on_gpu:
                        needed_layers = calculate_needed_layers(layer_idx, args.prefetch)
                        cleaned = cleanup_excess_layers(needed_layers)
                        fetched = fetch_missing_layers(needed_layers, current_layer_idx=layer_idx)

                    if layer_already_on_gpu:
                        add_smart_swapping_to_layer.prefetch_hits += 1
                    else:
                        add_smart_swapping_to_layer.prefetch_misses += 1

                    device = device_cache.get_device(layer_idx)
                    gpu_success = device.type == 'cuda'

                finally:
                    if PROFILE:
                        nvtx.range_pop()

                device = device_cache.get_device(layer_idx)
                gpu_success = device.type == 'cuda'
            else:
                # Layer not in cpu_swappable_layers (permanent resident)
                device = device_cache.get_device(layer_idx)
                gpu_success = device.type == 'cuda'

            # Detect Flux calling patterns
            layer_type = type(layer).__name__
            is_flux_call = layer._dgls_is_flux
            is_keyword_only_call = layer._dgls_kw_only

            if is_flux_call:
                # Flux patterns - preserve exact argument structure
                if fwd_args:
                    # SingleStreamBlock: block(img, vec=vec, pe=pe)
                    x = fwd_args[0]  # img
                    new_args = fwd_args[1:]  # should be empty
                    new_kwargs = kwargs  # vec=vec, pe=pe
                else:
                    # DoubleStreamBlock: block(img=img, txt=txt, vec=vec, pe=pe)
                    x = kwargs.get('img')
                    new_args = ()
                    new_kwargs = kwargs
            elif is_keyword_only_call:
                x = None
                new_args = ()
                new_kwargs = kwargs
            else:
                # WAN/SDXL pattern
                x = fwd_args[0] if fwd_args else None
                new_args = fwd_args[1:]
                new_kwargs = kwargs

            add_smart_swapping_to_layer.total_forward_calls += 1
            add_smart_swapping_to_layer.calls_this_step += 1

            # if args.verbose and add_smart_swapping_to_layer.total_forward_calls % 100 == 0:
            #     print_memory_optimization_analysis(layers, args)
                # debug_layer_locations(layers, device_cache)

            if PROFILE:
                nvtx.range_push(f"Layer_{layer_idx}_{'Forward'}")
            try:
                if not add_smart_swapping_to_layer.current_forward_logged:
                    current_step = getattr(add_smart_swapping_to_layer, 'current_step', 0)
                    if args.verbose:
                        print(
                            f" Forward pass step {current_step}: layers {min(cpu_swappable_layers)}-{max(cpu_swappable_layers)}")
                    add_smart_swapping_to_layer.current_forward_logged = True

                # Handle GPU failure case
                if not gpu_success:
                    print(f" Layer {layer_idx} failed to be on GPU, forcing aggressive cleanup...")
                    # debug_comfy_memory_state()

                    def emergency_cleanup():
                        for cleanup_idx in cpu_swappable_layers:
                            if cleanup_idx != layer_idx and cleanup_idx in gpu_layer_copies:
                                try:
                                    rebind(cleanup_idx)  # ensure pointers go back to CPU and VRAM is dropped
                                    gpu_layer_copies.pop(cleanup_idx, None)
                                    device_cache.mark_moved(cleanup_idx, torch.device('cpu'))
                                except Exception:
                                    pass

                    if args.cpu_threading and hasattr(add_smart_swapping_to_layer, 'cpu_thread_pool'):
                        cleanup_future = add_smart_swapping_to_layer.cpu_thread_pool.submit(emergency_cleanup)
                        cleanup_future.result()  # Wait for cleanup to complete
                    else:
                        emergency_cleanup()

                    gc.collect()
                    torch.cuda.empty_cache()

                    def emergency_gpu_move():
                        stage_to_gpu(layer_idx)
                        gpu_layer_copies[layer_idx] = True
                        device_cache.mark_moved(layer_idx, torch.device('cuda'))
                        return True

                    try:
                        if args.cpu_threading and hasattr(add_smart_swapping_to_layer, 'cpu_thread_pool'):
                            move_future = add_smart_swapping_to_layer.cpu_thread_pool.submit(emergency_gpu_move)
                            move_future.result()
                            device = torch.device(GPU_DEVICE)
                            gpu_success = True
                            print(f" Layer {layer_idx} moved to GPU after aggressive cleanup")
                        else:
                            emergency_gpu_move()
                            device = torch.device(GPU_DEVICE)
                            gpu_success = True
                            print(f" Layer {layer_idx} moved to GPU after aggressive cleanup")
                    except RuntimeError:
                        print(f" CRITICAL: Layer {layer_idx} cannot fit on GPU, skipping computation!")
                        return x  # Pass input through unchanged


                def move_to_device(tensor, target_device):
                    if isinstance(tensor, torch.Tensor) and tensor.device != target_device:
                        return comfy.model_management.cast_to_device(tensor, target_device, tensor.dtype, copy=False)
                    return tensor

                move_to_device.current_layer = layer_idx

                if is_flux_call:
                    if fwd_args:
                        # SingleStreamBlock - move fwd_args and kwargs
                        new_args = tuple(move_to_device(arg, device) for arg in fwd_args)
                        new_kwargs = {k: move_to_device(v, device) for k, v in kwargs.items()}
                    else:
                        # DoubleStreamBlock - move only kwargs
                        new_args = ()
                        new_kwargs = {k: move_to_device(v, device) for k, v in kwargs.items()}
                elif is_keyword_only_call:
                    # move only kwargs, no positional args
                    x = None
                    new_args = ()
                    new_kwargs = {k: move_to_device(v, device) for k, v in kwargs.items()}
                else:
                    x = move_to_device(x, device)
                    new_kwargs = {k: move_to_device(v, device) for k, v in kwargs.items()}
                    new_args = tuple(move_to_device(arg, device) for arg in new_args)

                # Wait for current layer's transfer to complete before computation
                if args.cuda_streams and layer_idx in cpu_swappable_layers:
                    # Check if we have a transfer event for this specific layer
                    if (hasattr(add_smart_swapping_to_layer, 'transfer_events') and
                            layer_idx in add_smart_swapping_to_layer.transfer_events):
                        # Make compute stream wait for this layer's transfer event
                        event = add_smart_swapping_to_layer.transfer_events[layer_idx]
                        add_smart_swapping_to_layer.compute_stream.wait_event(event)

                # debug_comfy_casting_memory(layer_idx, "PRE_COMPUTE")

                # 4. COMPUTATION with profiling
                def _compute_operation():
                    with torch.no_grad():
                        if args.verbose:
                            compute_start = time.perf_counter()

                        #  ensure default stream waits on H2D completion
                        if args.cuda_streams and layer_idx in cpu_swappable_layers:
                            if hasattr(add_smart_swapping_to_layer, 'transfer_events'):
                                ev = add_smart_swapping_to_layer.transfer_events.get(layer_idx)
                                if ev is not None:
                                    torch.cuda.current_stream().wait_event(ev)

                        if is_flux_call:
                            if fwd_args:
                                result = original_forward(*new_args, **new_kwargs)
                            else:
                                result = original_forward(**new_kwargs)
                        elif is_keyword_only_call:
                            result = original_forward(**new_kwargs)
                        else:
                            result = original_forward(x, *new_args, **new_kwargs)

                        # Store compute timing
                        if args.verbose:
                            compute_end = time.perf_counter()
                            compute_time = (compute_end - compute_start) * 1000  # Convert to ms
                            add_smart_swapping_to_layer.layer_compute_times[layer_idx].append(compute_time)

                        if args.verbose:
                            layer_compute_end = time.perf_counter()
                            overlap_stats['compute_end_times'][layer_idx] = layer_compute_end

                            # Calculate overlap for this layer
                            if (layer_idx in overlap_stats['transfer_start_times'] and
                                    layer_idx in overlap_stats['transfer_end_times']):

                                t_start = overlap_stats['transfer_start_times'][layer_idx]
                                t_end = overlap_stats['transfer_end_times'][layer_idx]
                                c_start = overlap_stats['compute_start_times'][layer_idx]
                                c_end = overlap_stats['compute_end_times'][layer_idx]

                                # Calculate overlap duration
                                overlap_start = max(t_start, c_start)
                                overlap_end = min(t_end, c_end)
                                overlap_duration = max(0, overlap_end - overlap_start)

                                transfer_duration = t_end - t_start
                                compute_duration = c_end - c_start

                                if transfer_duration > 0 and compute_duration > 0:
                                    overlap_ratio = overlap_duration / min(transfer_duration, compute_duration)
                                    overlap_stats['overlaps'].append({
                                        'layer': layer_idx,
                                        'overlap_ms': overlap_duration * 1000,
                                        'transfer_ms': transfer_duration * 1000,
                                        'compute_ms': compute_duration * 1000,
                                        'overlap_ratio': overlap_ratio
                                    })

                        return result

                # Profile compute every 5th step, not every layer
                current_step = getattr(add_smart_swapping_to_layer, 'current_step', 0)
                should_profile_compute = (args.verbose)

                if PROFILE:
                    nvtx.range_push(f"Compute_L{layer_idx}")
                try:
                    result = _compute_operation()
                    # debug_comfy_casting_memory(layer_idx, "POST_COMPUTE")
                    return result
                finally:
                    if PROFILE:
                        nvtx.range_pop()
            finally:
                if PROFILE:
                    nvtx.range_pop()

        result = _whole_forward_operation()
        # debug_comfy_casting_memory(layer_idx, "POST_FORWARD")
        return result

    # Replace forward method
    layer.forward = swapped_forward
    layer._dgls_swapped = True

# =============================================================================
# COMFYUI NODE CLASS - by obisin
# =============================================================================

class DynamicSwappingLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "layers": ("LAYERS",),
                # "initial_gpu_layers": ("INT",{"default": 1, "min": 0, "max": 100, "tooltip": TOOLTIPS["initial_gpu_layers"]}),
                # "final_gpu_layers": ("INT",{"default": 1, "min": 0, "max": 100, "tooltip": TOOLTIPS["final_gpu_layers"]}),
                # "auto_allocate_layers": ("BOOLEAN", {"default": False, "tooltip": "Automatically determine layer placement based on available VRAM. Overrides any other layer placement."}),
                "prefetch": ("INT", {"default": 1, "min": 0, "max": 100, "tooltip": TOOLTIPS["prefetch"]}),
                "cpu_threading": ("BOOLEAN", {"default": False, "tooltip": TOOLTIPS["cpu_threading"]}),
                "cuda_streams": ("BOOLEAN", {"default": False, "tooltip": TOOLTIPS["cuda_streams"]}),
                # "pin_memory": ("BOOLEAN", {"default": False, "tooltip": TOOLTIPS["pin_memory"]}),
                "verbose": ("BOOLEAN", {"default": True, "tooltip": TOOLTIPS["verbose"]}),
            },
            "optional": {
                "gpu_layer_indices": ("STRING", {"default": "", "tooltip": TOOLTIPS["gpu_layers"]}),
                "cast_target": ("STRING", {"default": "", "tooltip": TOOLTIPS["cast_target"]}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)

    FUNCTION = "load_model_with_swapping"
    CATEGORY = "loaders"
    TITLE = "Dynamic Swapping Loader"

    def load_model_with_swapping(self, model, layers, prefetch, verbose,  gpu_layer_indices="",
                                 cast_target="",pin_memory=False, auto_allocate_layers = True,  cuda_streams=False, cpu_threading=False):

        global gpu_resident_layers, cpu_swappable_layers, device_cache, pending_pin
        initial_gpu_layers = 0
        final_gpu_layers = 0


        def _teardown_swapping_state():
            """Complete cleanup of all modified layers and global state"""
            global layers, device_cache, gpu_resident_layers, cpu_swappable_layers

            if args.verbose:
                debug_print_pinning_report(layers, "BEFORE-TEARDOWN")
                debug_assert_no_gpu_params(layers, "BEFORE-TEARDOWN")

            # Restore original forward methods and move everything to CPU
            if layers is not None:
                for layer in layers:
                    if hasattr(layer, '_original_forward'):
                        layer.forward = layer._original_forward
                        delattr(layer, '_original_forward')
                    if hasattr(layer, '_dgls_swapped'):
                        delattr(layer, '_dgls_swapped')
                    if hasattr(layer, '_dgls_fixed'):
                        delattr(layer, '_dgls_fixed')

            # 2) NEW: make sure everything is truly CPU, drop GPU storages, and UNPIN CPU masters
            if args.pin_memory:
                for i, layer in enumerate(layers):
                    # ensure no lingering GPU storages
                    try:
                        rebind(i)  # safe if never staged
                    except Exception:
                        pass

                    if args.pin_memory:
                        try:
                            unpin_cpu_memory(layer)
                        except Exception:
                            pass

                    # drop master refs so Python can GC them
                    for attr in ('_cpu_params', '_cpu_buffers'):
                        if hasattr(layer, attr):
                            try:
                                delattr(layer, attr)
                            except Exception:
                                pass

            # Kill thread pool unconditionally
            if hasattr(add_smart_swapping_to_layer, 'cpu_thread_pool'):
                try:
                    add_smart_swapping_to_layer.cpu_thread_pool.shutdown(wait=True)
                    delattr(add_smart_swapping_to_layer, 'cpu_thread_pool')
                except:
                    pass

            # Kill CUDA streams/events unconditionally
            for attr in ['fetch_stream', 'compute_stream', 'transfer_events', 'streams_initialized',
                         'transfer_futures']:
                if hasattr(add_smart_swapping_to_layer, attr):
                    try:
                        delattr(add_smart_swapping_to_layer, attr)
                    except:
                        pass

            # _final_cuda_sweep_all_layers()

            if args.verbose:
                if layers is not None:
                    debug_print_pinning_report(layers, "AFTER-TEARDOWN")
                    debug_assert_no_gpu_params(layers, "AFTER-TEARDOWN")
                else:
                    print("[PIN-REPORT] AFTER-TEARDOWN  (layers=None)")

            # Reset globals
            gpu_resident_layers.clear()
            cpu_swappable_layers.clear()
            layers = None
            device_cache = None

        # CRITICAL: Complete teardown first
        _teardown_swapping_state()

        if args.verbose:
            print(f"VRAM after teardown: {torch.cuda.memory_allocated(0) / 1024 ** 3:.3f}GB")

        # Check if previous model cleanup happened
        if hasattr(add_smart_swapping_to_layer, 'cpu_thread_pool'):
            print("WARNING: Previous model not cleaned up - thread pool still exists")

        # debug_comfy_memory_state()

        # Reset all stats for new generation
        if hasattr(add_smart_swapping_to_layer, 'total_start_time'):
            del add_smart_swapping_to_layer.total_start_time
        if hasattr(add_smart_swapping_to_layer, 'current_step'):
            add_smart_swapping_to_layer.current_step = 0
        if hasattr(add_smart_swapping_to_layer, 'total_forward_calls'):
            add_smart_swapping_to_layer.total_forward_calls = 0
        if hasattr(add_smart_swapping_to_layer, 'calls_this_step'):
            add_smart_swapping_to_layer.calls_this_step = 0
        if hasattr(add_smart_swapping_to_layer, 'current_forward_logged'):
            add_smart_swapping_to_layer.current_forward_logged = False
        if hasattr(add_smart_swapping_to_layer, 'prefetch_hits'):
            add_smart_swapping_to_layer.prefetch_hits = 0
        if hasattr(add_smart_swapping_to_layer, 'prefetch_misses'):
            add_smart_swapping_to_layer.prefetch_misses = 0
        if hasattr(add_smart_swapping_to_layer, 'layer_transfer_times'):
            add_smart_swapping_to_layer.layer_transfer_times.clear()
        if hasattr(add_smart_swapping_to_layer, 'layer_compute_times'):
            add_smart_swapping_to_layer.layer_compute_times.clear()

        global swap_stats, transfer_stats
        swap_stats = {'to_gpu': 0, 'to_cpu': 0}
        transfer_stats = {
            'to_gpu_times': [], 'to_cpu_times': [], 'to_gpu_speeds': [], 'to_cpu_speeds': [],
            'current_step_gpu_times': [], 'current_step_cpu_times': [],
            'current_step_gpu_speeds': [], 'current_step_cpu_speeds': []
        }

        if not hasattr(add_smart_swapping_to_layer, 'total_start_time'):
            add_smart_swapping_to_layer.total_start_time = time.time()

        # Reset layer-specific stats
        if hasattr(add_smart_swapping_to_layer, 'prefetch_hits'):
            add_smart_swapping_to_layer.prefetch_hits = 0
            add_smart_swapping_to_layer.prefetch_misses = 0

        # Update ALL args with node parameters
        args.dynamic_swapping = True
        args.initial_gpu_layers = initial_gpu_layers
        args.final_gpu_layers = final_gpu_layers
        args.prefetch = prefetch
        args.cuda_streams = cuda_streams
        args.cpu_threading = cpu_threading
        args.verbose = verbose
        args.auto_allocate_layers = auto_allocate_layers
        args.pin_memory = pin_memory
        should_pin = args.pin_memory

        args.gpu_layers = gpu_layer_indices.strip() if gpu_layer_indices.strip() else None
        args.cast_target = cast_target.strip() if cast_target.strip() else None

        add_smart_swapping_to_layer.layer_transfer_times = defaultdict(list)
        add_smart_swapping_to_layer.layer_compute_times = defaultdict(list)

        if not hasattr(add_smart_swapping_to_layer, 'stats_initialized'):
            add_smart_swapping_to_layer.stats_initialized = True
            add_smart_swapping_to_layer.prefetch_hits = 0
            add_smart_swapping_to_layer.prefetch_misses = 0

        if not hasattr(add_smart_swapping_to_layer, 'prefetch_hits'):
            add_smart_swapping_to_layer.prefetch_hits = 0
            add_smart_swapping_to_layer.prefetch_misses = 0

        if not hasattr(add_smart_swapping_to_layer, 'layer_compute_times'):
            add_smart_swapping_to_layer.layer_compute_times = []

        # Initialize coordination state (once) - this should match _patch_model_patcher
        if not hasattr(add_smart_swapping_to_layer, 'current_sampling_step'):
            add_smart_swapping_to_layer.current_sampling_step = 0
            add_smart_swapping_to_layer.last_processed_step = -1
            add_smart_swapping_to_layer.apply_model_call_count = 0
            add_smart_swapping_to_layer.current_forward_logged = False

        # Initialize coordination state (once)
        if not hasattr(add_smart_swapping_to_layer, 'current_sampling_step'):
            add_smart_swapping_to_layer.current_sampling_step = 0
            add_smart_swapping_to_layer.last_processed_step = -1

        if not hasattr(add_smart_swapping_to_layer, 'step_timing_initialized'):
            add_smart_swapping_to_layer.step_timing_initialized = True
            add_smart_swapping_to_layer.step_start_time = time.time()
            add_smart_swapping_to_layer.current_step = 0

        if not hasattr(add_smart_swapping_to_layer, 'total_forward_calls'):
            add_smart_swapping_to_layer.total_forward_calls = 0
            add_smart_swapping_to_layer.current_step = 0
            add_smart_swapping_to_layer.calls_this_step = 0

        # CUDA STREAMS: Initialize streams (once) - only if enabled
        if args.cuda_streams and not hasattr(add_smart_swapping_to_layer, 'streams_initialized'):
            try:
                add_smart_swapping_to_layer.streams_initialized = True
                add_smart_swapping_to_layer.fetch_stream = torch.cuda.Stream()  # All transfers
                add_smart_swapping_to_layer.compute_stream = torch.cuda.Stream()  # Only computation
                add_smart_swapping_to_layer.transfer_events = {}  # Track transfer completion
                print("✓ CUDA Streams enabled for copy-compute overlap")
            except Exception as e:
                print(f"✗ CUDA Streams failed to initialize: {e}")
                args.cuda_streams = False
                print("✓ Falling back to default stream")

        # CPU THREADING: Initialize thread pool (once) - only if enabled
        if args.cpu_threading and not hasattr(add_smart_swapping_to_layer, 'cpu_thread_pool'):
            try:
                if torch.cuda.is_available():
                    _ = torch.empty(1, device=GPU_DEVICE)
                add_smart_swapping_to_layer.cpu_thread_pool = ThreadPoolExecutor(max_workers=1,
                                                                                 thread_name_prefix="cpu_transfer")
                add_smart_swapping_to_layer.transfer_futures = {}
                print("✓ CPU Threading enabled")
            except Exception as e:
                print(f"✗ CPU Threading failed to initialize: {e}")
                args.cpu_threading = False
                print("✓ Falling back to default CPU transfers")

        if args.verbose:
            print(f"\n=== LAYER DTYPE INSPECTION ===")
            print(f"Total layers: {len(layers)}")

            for i, layer in enumerate(layers):
                layer_type = type(layer).__name__

                # Get parameter dtypes for this layer
                param_dtypes = set()
                param_count = 0
                for name, param in layer.named_parameters():
                    param_dtypes.add(param.dtype)
                    param_count += 1

                # Get buffer dtypes
                buffer_dtypes = set()
                buffer_count = 0
                for name, buffer in layer.named_buffers():
                    buffer_dtypes.add(buffer.dtype)
                    buffer_count += 1

                # Format dtype info
                dtype_info = []
                if param_dtypes:
                    dtype_info.append(f"params: {list(param_dtypes)}")
                if buffer_dtypes:
                    dtype_info.append(f"buffers: {list(buffer_dtypes)}")

                dtype_str = ", ".join(dtype_info) if dtype_info else "no tensors"

                print(f"  {i:2d}: {layer_type:<20} | {dtype_str}")
            print("===============================\n")

        if args.dynamic_swapping:
            if args.verbose:
            #     print("Applying ENHANCED smart GPU allocation + dynamic swapping...")
                print(f"Setting up enhanced swapping for {len(layers)} layers...")

            if args.gpu_layers:
                try:
                    specified_gpu_layers = set(map(int, args.gpu_layers.split(',')))
                    if args.verbose:
                        print(f" Using specified GPU layers: {sorted(specified_gpu_layers)}")

                    # Validate layer indices
                    invalid_layers = [idx for idx in specified_gpu_layers if idx >= len(layers) or idx < 0]
                    if invalid_layers:
                        raise ValueError(f"Invalid layer indices: {invalid_layers}. Must be 0-{len(layers) - 1}")

                except ValueError as e:
                    print(f" Error parsing --gpu_layers: {e}")
                    print("Example usage: --gpu_layers 0,1,2,14,18,19,20,21,22")
                    raise e
            elif args.auto_allocate_layers:
                specified_gpu_layers = calculate_auto_gpu_layers(layers, args)

            else: #CODE SHOULD BBE NOW UNREACHABLE- ONLY AUTO OR YOU PICK YOUR OWN INDICES
                specified_gpu_layers = None
                if args.verbose:
                    print(f"Using initial/final layer allocation: {initial_gpu_layers}/{final_gpu_layers}")

            # Clear sets to prevent double-addition from previous runs
            gpu_resident_layers.clear()
            cpu_swappable_layers.clear()

            # PHASE 1: Determine which layers go where based on specified layers or initial/final counts
            # for i, layer in enumerate(layers):
            #     if hasattr(layer, 'to'):
            #         if specified_gpu_layers is not None:
            #             # Use specified layer indices for GPU placement
            #             if i in specified_gpu_layers:
            #                 try:
            #                     layer.to(GPU_DEVICE)
            #                     gpu_resident_layers.add(i)  # This is correct - after successful move
            #                     if args.verbose:
            #                         print(f"Layer {i} (specified) -> GPU permanent")
            #
            #                 except RuntimeError as e:
            #                     print(f"CRITICAL: Cannot fit specified layer {i} on GPU!")
            #                     print(f"GPU memory may be insufficient. Consider removing layer {i} from --gpu_layers")
            #                     raise e
            #             else:
            #                 # Not in specified list, make swappable
            #                 layer.to(CPU_DEVICE)
            #                 if args.pin_memory:
            #
            #                     pending_pin = locals().get("pending_pin", set())
            #                     pending_pin.add(i)
            #                 if not hasattr(layer, '_cpu_params'):
            #                     layer._cpu_params = {n: p for n, p in layer.named_parameters(recurse=True)}
            #                     layer._cpu_buffers = {n: b for n, b in layer.named_buffers(recurse=True)}
            #                 cpu_swappable_layers.add(i)
            #                 if args.verbose:
            #                     print(f"Layer {i} (not specified) -> CPU swappable")
            #         else:
            #             # Use original initial/final logic as fallback
            #             if i < initial_gpu_layers:
            #                 # Initial layers on GPU permanently
            #                 try:
            #                     layer.to(GPU_DEVICE)
            #                     gpu_resident_layers.add(i)
            #                     if args.verbose:
            #                         print(f"Layer {i} (initial) -> GPU permanent")
            #                 except RuntimeError as e:
            #                     print(f"GPU exhausted at layer {i}, moving to CPU with swapping")
            #                     layer.to(CPU_DEVICE)
            #                     if args.pin_memory:
            #                         pending_pin = locals().get("pending_pin", set())
            #                         pending_pin.add(i)
            #                     if not hasattr(layer, '_cpu_params'):
            #                         layer._cpu_params = {n: p for n, p in layer.named_parameters(recurse=True)}
            #                         layer._cpu_buffers = {n: b for n, b in layer.named_buffers(recurse=True)}
            #                     cpu_swappable_layers.add(i)
            #             elif i >= (len(layers) - final_gpu_layers):
            #                 # Final layers on GPU permanently
            #                 try:
            #                     layer.to(GPU_DEVICE)
            #                     gpu_resident_layers.add(i)
            #                     if args.verbose:
            #                         print(f"Layer {i} (final) -> GPU permanent")
            #                 except RuntimeError as e:
            #                     print(f"CRITICAL: Cannot fit final layer {i} on GPU!")
            #                     raise e
            #             else:
            #                 # Middle layers on CPU with swapping capability
            #                 layer.to(CPU_DEVICE)
            #                 if args.pin_memory:
            #                     pending_pin = locals().get("pending_pin", set())
            #                     pending_pin.add(i)
            #                 if not hasattr(layer, '_cpu_params'):
            #                     layer._cpu_params = {n: p for n, p in layer.named_parameters(recurse=True)}
            #                     layer._cpu_buffers = {n: b for n, b in layer.named_buffers(recurse=True)}
            #                 cpu_swappable_layers.add(i)
            #                 if args.verbose:
            #                     print(f"Layer {i} (middle) -> CPU swappable")

            # PHASE 1: Determine which layers go where based on specified layers or initial/final counts
            for i, layer in enumerate(layers):
                if not hasattr(layer, 'to'):
                    continue

                if specified_gpu_layers is not None:
                    # Use specified layer indices for GPU placement
                    if i in specified_gpu_layers:
                        try:
                            layer.to(GPU_DEVICE)
                            gpu_resident_layers.add(i)
                            if args.verbose:
                                print(f"Layer {i} (specified) -> GPU permanent")
                        except RuntimeError as e:
                            print(f"CRITICAL: Cannot fit specified layer {i} on GPU!")
                            print(f"GPU memory may be insufficient. Consider removing layer {i} from --gpu_layers")
                            raise e
                    else:
                        # --- SWAPPABLE: buffers on GPU, params master on CPU (one-time) ---
                        # move whole module to GPU to keep BUFFERS there
                        layer.to(GPU_DEVICE)

                        # one-time CPU masters for every PARAM (no new Parameter objects; rebind .data)
                        if not hasattr(layer, '_cpu_master_data'):
                            layer._cpu_master_data = {}
                            with torch.no_grad():
                                for pname, p in layer.named_parameters(recurse=True):
                                    if p is None:
                                        continue
                                    cpu_master = p.data.to('cpu', copy=True)
                                    if args.pin_memory:
                                        try:
                                            cpu_master = cpu_master.pin_memory()
                                        except Exception:
                                            pass
                                    owner, leaf = _owner_and_key(layer, pname)
                                    # owner._parameters[leaf].data = cpu_master  # module still owns same Parameter
                                    _reassign_param(owner, leaf, cpu_master)
                                    layer._cpu_master_data[pname] = cpu_master

                        # keep compatibility with your teardown/pin code
                        if not hasattr(layer, '_cpu_params'):
                            layer._cpu_params = {n: p for n, p in layer.named_parameters(recurse=True)}
                            layer._cpu_buffers = {n: b for n, b in layer.named_buffers(recurse=True)}

                        # record swappable
                        cpu_swappable_layers.add(i)
                        if args.pin_memory:
                            pending_pin.add(i)

                        if args.verbose:
                            print(f"Layer {i} (not specified) -> CPU swappable "
                                  f"(buffers on GPU, params master on CPU)")
                else:
                    # Use original initial/final logic as fallback
                    if i < initial_gpu_layers:
                        # Initial layers on GPU permanently
                        try:
                            layer.to(GPU_DEVICE)
                            gpu_resident_layers.add(i)
                            if args.verbose:
                                print(f"Layer {i} (initial) -> GPU permanent")
                        except RuntimeError:
                            # couldn't keep resident; make it swappable instead
                            # try swappable layout with buffers on GPU
                            try:
                                layer.to(GPU_DEVICE)
                                # build CPU masters once
                                if not hasattr(layer, '_cpu_master_data'):
                                    layer._cpu_master_data = {}
                                    with torch.no_grad():
                                        for pname, p in layer.named_parameters(recurse=True):
                                            if p is None:
                                                continue
                                            cpu_master = p.data.to('cpu', copy=True)
                                            if args.pin_memory:
                                                try:
                                                    cpu_master = cpu_master.pin_memory()
                                                except Exception:
                                                    pass
                                            owner, leaf = _owner_and_key(layer, pname)
                                            # owner._parameters[leaf].data = cpu_master
                                            _reassign_param(owner, leaf, cpu_master)
                                            layer._cpu_master_data[pname] = cpu_master
                            except RuntimeError:
                                # absolute fallback: leave everything on CPU for now
                                layer.to(CPU_DEVICE)
                                if not hasattr(layer, '_cpu_master_data'):
                                    layer._cpu_master_data = {
                                        pname: p.data for pname, p in layer.named_parameters(recurse=True) if
                                        p is not None
                                    }

                            if not hasattr(layer, '_cpu_params'):
                                layer._cpu_params = {n: p for n, p in layer.named_parameters(recurse=True)}
                                layer._cpu_buffers = {n: b for n, b in layer.named_buffers(recurse=True)}
                            cpu_swappable_layers.add(i)
                            if args.pin_memory:
                                pending_pin.add(i)
                            if args.verbose:
                                print(f"Layer {i} (initial->swappable) -> CPU swappable "
                                      f"(buffers on GPU if possible, params master on CPU)")
                    elif i >= (len(layers) - final_gpu_layers):
                        # Final layers on GPU permanently
                        try:
                            layer.to(GPU_DEVICE)
                            gpu_resident_layers.add(i)
                            if args.verbose:
                                print(f"Layer {i} (final) -> GPU permanent")
                        except RuntimeError as e:
                            print(f"CRITICAL: Cannot fit final layer {i} on GPU!")
                            raise e
                    else:
                        # --- SWAPPABLE middle: buffers on GPU, params master on CPU (one-time) ---
                        layer.to(GPU_DEVICE)

                        if not hasattr(layer, '_cpu_master_data'):
                            layer._cpu_master_data = {}
                            with torch.no_grad():
                                for pname, p in layer.named_parameters(recurse=True):
                                    if p is None:
                                        continue
                                    cpu_master = p.data.to('cpu', copy=True)
                                    if args.pin_memory:
                                        try:
                                            cpu_master = cpu_master.pin_memory()
                                        except Exception:
                                            pass
                                    owner, leaf = _owner_and_key(layer, pname)
                                    # owner._parameters[leaf].data = cpu_master
                                    _reassign_param(owner, leaf, cpu_master)
                                    layer._cpu_master_data[pname] = cpu_master

                        if not hasattr(layer, '_cpu_params'):
                            layer._cpu_params = {n: p for n, p in layer.named_parameters(recurse=True)}
                            layer._cpu_buffers = {n: b for n, b in layer.named_buffers(recurse=True)}

                        cpu_swappable_layers.add(i)
                        if args.pin_memory:

                            pending_pin.add(i)

                        if args.verbose:
                            print(f"Layer {i} (middle) -> CPU swappable "
                                  f"(buffers on GPU, params master on CPU)")

            global swappable_list
            swappable_list = sorted(cpu_swappable_layers)
            global swappable_index
            swappable_index = {v: i for i, v in enumerate(swappable_list)}
            print(f"✓ {len(gpu_resident_layers)} layers permanently on GPU: {sorted(gpu_resident_layers)}")
            print(f"✓ {len(cpu_swappable_layers)} layers on CPU with smart swapping: {sorted(cpu_swappable_layers)}")

            add_smart_swapping_to_layer.first_swappable = (min(cpu_swappable_layers) if cpu_swappable_layers else None)
            global gpu_occupied_swappable
            gpu_occupied_swappable = set()
            for i in cpu_swappable_layers:
                # Check actual device of each swappable layer at start
                try:
                    param = next(layers[i].parameters())
                    if param.device.type == 'cuda':
                        gpu_occupied_swappable.add(i)
                except StopIteration:
                    pass

            print("Calculating layer sizes...")
            global layer_sizes_mb
            for i, layer in enumerate(layers):
                layer_sizes_mb[i] = sum(p.numel() * p.element_size() for p in layer.parameters()) / (1024 * 1024)
                if args.verbose:
                    print(f"   Layer {i}: {layer_sizes_mb[i]:.1f}MB")
            print()

            if cast_target:
                # Parse the cast_target string first
                try:
                    pairs  = [tuple(ch.strip().split()) for ch in cast_target.split(',') if ch.strip()]
                    casting_handler = CastingHandler()
                    casting_handler.precast_all_layers(cpu_swappable_layers, layers, pairs)
                    if args.pin_memory:
                        pending_pin.clear()
                        for idx in cpu_swappable_layers:
                            pending_pin.add(idx)
                except ValueError:
                    print(f"  Invalid cast_target format: '{cast_target}'. Expected: 'from_dtype to_dtype'")

            def _planned_pin_bytes(cpu_swappable_layers, layers) -> int:
                min_mb = int(getattr(args, 'pin_min_mb', 128))
                total_mb = 0
                for i in cpu_swappable_layers:
                    if get_cached_layer_size_mb(i) >= min_mb:
                        total_mb += get_cached_layer_size_mb(i)
                return int(total_mb * 1024 * 1024)

            def _enough_free_ram_for_pin(plan_bytes: int) -> bool:
                # keep a headroom so the OS stays happy
                try:
                    import psutil
                    vm = psutil.virtual_memory()
                    # reserve at least 6GB or 20% of total, whichever larger
                    reserve = max(5 * 1024 ** 3, int(0.1 * vm.total))
                    return plan_bytes + reserve <= vm.available
                except Exception:

                    baseline = 64 * 1024 ** 3
                    reserve = int(0.1 * baseline)
                    return plan_bytes <= reserve

            if args.pin_memory:
                plan = _planned_pin_bytes(cpu_swappable_layers, layers)
                if args.verbose:
                    print(f"[PINNING] planned={plan / 1024 ** 2:.1f} MB, gate>= {getattr(args, 'pin_min_mb', 128)} MB per layer")
                if not _enough_free_ram_for_pin(plan):
                    print("[PINNING] Disabled: not enough free RAM for pinning.")
                    should_pin = False

            if should_pin:
                for idx in sorted(pending_pin):
                    L = layers[idx]
                    pin_cpu_memory(L, idx)  # whole-layer pin
                    # refresh masters to the now-pinned tensors
                    L._cpu_params = {n: p for n, p in L.named_parameters(recurse=True)}
                    L._cpu_buffers = {n: b for n, b in L.named_buffers(recurse=True)}

            if args.verbose:
                debug_print_pinning_report(layers, "AFTER-SETUP")
                debug_assert_no_gpu_params(layers, "AFTER-SETUP", ignore=gpu_resident_layers)

                # print("\n=== Module Size Debug ===")
                # total_module_mem = 0
                # for name, module in model.model.named_modules():
                #     if hasattr(module, 'weight') or len(list(module.parameters())) > 0:
                #         module_mem = sum(p.numel() * p.element_size() for p in module.parameters())
                #         if module_mem > 0:
                #             print(f"Module {name}: {module_mem / 1024 ** 2:.1f} MB")
                #             total_module_mem += module_mem
                # print(f"Total module memory: {total_module_mem / 1024 ** 3:.2f} GB")

            global device_cache
            device_cache = LayerDeviceCache(model, layers)

            if args.verbose:
                analysis_result = print_memory_optimization_analysis( layers, args)
                debug_layer_locations(layers, device_cache)

            if cast_target:
                print(f"\n=== LAYER CASTED DTYPES ===")
                print(f"Total layers: {len(layers)}")

                for i, layer in enumerate(layers):
                    layer_type = type(layer).__name__

                    # Get parameter dtypes for this layer
                    param_dtypes = set()
                    param_count = 0
                    for name, param in layer.named_parameters():
                        param_dtypes.add(param.dtype)
                        param_count += 1

                    # Get buffer dtypes
                    buffer_dtypes = set()
                    buffer_count = 0
                    for name, buffer in layer.named_buffers():
                        buffer_dtypes.add(buffer.dtype)
                        buffer_count += 1

                    # Format dtype info
                    dtype_info = []
                    if param_dtypes:
                        dtype_info.append(f"params: {list(param_dtypes)}")
                    if buffer_dtypes:
                        dtype_info.append(f"buffers: {list(buffer_dtypes)}")

                    dtype_str = ", ".join(dtype_info) if dtype_info else "no tensors"

                    print(f"  {i:2d}: {layer_type:<20} | {dtype_str}")

                print("Calculating layer sizes for Casted...")
                print(f"\n=== LAYER CASTED SIZES ===")
                # global layer_sizes_mb
                for i, layer in enumerate(layers):
                    layer_sizes_mb[i] = sum(p.numel() * p.element_size() for p in layer.parameters()) / (1024 * 1024)
                    if args.verbose:
                        print(f"   Layer {i}: {layer_sizes_mb[i]:.1f}MB")

                print("===============================\n")
                print()

        # #DEBUGGING: Add this after casting
        # for i in [15, 20, 25]:  # Sample swappable layers
        #     for name, param in layers[i].named_parameters():
        #         print(
        #             f"CHECK: Layer {i} param {name}: {param.dtype}, size: {param.numel() * param.element_size() / 1024 ** 2:.1f}MB")

        print("Fixing inference tensor parameters...")
        total_fixed = 0
        for i, layer in enumerate(layers):
            if not getattr(layer, "_dgls_fixed", False):
                if fix_inference_tensor_parameters(layer):
                    total_fixed += 1
                layer._dgls_fixed = True

        if args.verbose and total_fixed > 0:
            print(f"Fixed inference parameters in {total_fixed} layers")

        if args.verbose:
            print("Validating tensor parameters...")
            for i, layer in enumerate(layers):
                broken_tensors = []
                for name, param in layer.named_parameters():
                    try:
                        _ = param._version
                    except Exception:
                        broken_tensors.append(f"{name}")

                if broken_tensors:
                    print(f"WARNING: Layer {i} still has broken tensors: {broken_tensors}")

        for layer_idx in cpu_swappable_layers:
            add_smart_swapping_to_layer(
                layers[layer_idx],
                layer_idx,
                layers,
                gpu_resident_layers,
                cpu_swappable_layers)

        from comfy.patcher_extension import CallbacksMP
        def _release_pinned(*_a, **_k):
            if not getattr(args, "pin_memory", False):
                return
            # Unpin only what this run marked swappable
            for idx in cpu_swappable_layers:
                try:
                    L = layers[idx]
                except Exception:
                    continue
                try:
                    unpin_cpu_memory(L)
                    if hasattr(L, "_cpu_params"):  delattr(L, "_cpu_params")
                    if hasattr(L, "_cpu_buffers"): delattr(L, "_cpu_buffers")
                except Exception:
                    pass
            try:
                import torch.cuda.memory as _tcm
                if hasattr(_tcm, "_host_allocator_emptyCache"):
                    _tcm._host_allocator_emptyCache()
            except Exception:
                pass

        if args.pin_memory:
            try:
                model.add_callback(CallbacksMP.ON_DETACH, _release_pinned)
                print("ON_DETACH worked")
            except Exception:
                try:
                    model.add_callback(CallbacksMP.ON_CLEANUP, _release_pinned)
                    print("ON_CLEANUP worked")
                except Exception:
                    print(f"[PINNING] cleanup registration failed: {e}")
            gc.collect()

        if args.verbose:
            print("✓ Dynamic swapping successfully integrated with ComfyUI")


        return (model,)

    def __del__(self):
        """Clean up modified layers when node is destroyed"""
        global layers, gpu_resident_layers, cpu_swappable_layers, device_cache

        # Restore original forward methods
        if 'layers' in globals() and layers is not None:
            for i, layer in enumerate(layers):
                if hasattr(layer, '_original_forward'):
                    layer.forward = layer._original_forward
                    delattr(layer, '_original_forward')

        # Clear global state
        if gpu_resident_layers:
            gpu_resident_layers.clear()
        if cpu_swappable_layers:
            cpu_swappable_layers.clear()

        layers = None
        device_cache = None

# =============================================================================
# MIXED PRECISION FRAMEWORK CODE by obisin
# =============================================================================


class CastingHandler:
    def __init__(self):
        self.original_clip_grad_norm = None
        self.dtype_conversion_map = {
            torch.float8_e4m3fn: torch.bfloat16,
            torch.float8_e5m2: torch.bfloat16,
        }
        self.safe_dtypes = {torch.float32, torch.float16, torch.bfloat16}
        self.is_patched = False


    def cast_to_dtype(self, param, target_device, cast_target=None):
        """Cast specific source dtype to target dtype"""
        source_dtype = param.dtype

        if cast_target:
            cast_from, cast_to = cast_target

            # Map string to torch dtypes
            dtype_map = {
                'f32': torch.float32,
                'bf16': torch.bfloat16,
                'f16': torch.float16,
                'f8_e4m3': torch.float8_e4m3fn,
                'f8_e5m2': torch.float8_e5m2,
                'nf4': 'nf4',  # Special handling needed NOT TESTED
                'fp4': 'fp4'  # Special handling needed
            }

            # Only cast if source matches the FROM dtype
            if source_dtype == dtype_map[cast_from]:
                # Handle 4-bit quantization
                if cast_to in ['nf4', 'fp4']:
                    try:
                        import bitsandbytes as bnb
                        quantized, quant_state = bnb.functional.quantize_4bit(
                            param.data, blocksize=64, quant_type=cast_to
                        )
                        # Return dequantized version on target device
                        return bnb.functional.dequantize_4bit(
                            quantized.to(target_device), quant_state.to(target_device)
                        )
                    except ImportError:
                        raise RuntimeError(
                            " bitsandbytes required for 4-bit quantization. Install with: pip install bitsandbytes")
                    except Exception as e:
                        raise RuntimeError(f" 4-bit quantization failed: {e}")
                else:
                    # Regular dtype casting
                    return param.to(target_device, dtype=dtype_map[cast_to], non_blocking=True)

        # No casting needed
        return param.to(target_device, non_blocking=True)

    def _token_to_torch_dtype(self,tok: str):
        def _normalize_dtype_token(tok: str):
            t = tok.strip().lower().replace("torch.", "")
            aliases = {
                "f32": "float32", "fp32": "float32", "float": "float32", "float32": "float32",
                "f16": "float16", "fp16": "float16", "half": "float16", "float16": "float16",
                "bf16": "bfloat16", "bfloat16": "bfloat16",
                "f8e4m3": "float8_e4m3fn", "e4m3": "float8_e4m3fn", "float8_e4m3": "float8_e4m3fn",
                "float8_e4m3fn": "float8_e4m3fn",
                "f8e5m2": "float8_e5m2", "e5m2": "float8_e5m2", "float8_e5m2": "float8_e5m2",
            }
            return aliases.get(t, t)

        t = _normalize_dtype_token(tok)
        # Guard against builds that don’t have float8
        mapping = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        if hasattr(torch, "float8_e4m3fn"):
            mapping["float8_e4m3fn"] = torch.float8_e4m3fn
        if hasattr(torch, "float8_e5m2"):
            mapping["float8_e5m2"] = torch.float8_e5m2
        if t not in mapping:
            raise ValueError(f"Unsupported dtype token: {tok}")
        return mapping[t]

    def _parse_cast_target(self, cast_target) -> list[tuple[torch.dtype, torch.dtype]]:
        """
        Accepts: 'f32 f16' or 'f32 f16, bf16 f16'
        Returns list of (from_dtype, to_dtype) pairs.
        """
        if not cast_target:
            return []
        if isinstance(cast_target, (list, tuple)) and cast_target and isinstance(cast_target[0], (list, tuple)):
            parts = cast_target
        else:
            parts = []
            for chunk in str(cast_target).split(","):
                chunk = chunk.strip()
                if not chunk:
                    continue
                toks = chunk.split()
                if len(toks) != 2:
                    raise ValueError(f"Bad cast_target chunk: {chunk}")
                parts.append((toks[0], toks[1]))

        pairs = [(self._token_to_torch_dtype(src), self._token_to_torch_dtype(dst)) for src, dst in parts]
        return pairs

    def precast_all_layers(self, cpu_swappable_layers, layers, cast_target=None):
        """
        Cast only tensors matching the requested source dtype(s) to the target dtype(s).
        - Parameters: considered for casting.
        - Buffers: only floating-point buffers; keep BN running stats in fp32.
        - Owner-aware writes; free old tensors immediately to limit peaks.
        """
        pairs = self._parse_cast_target(cast_target)
        if not pairs:
            print("  No cast_target specified, skipping precasting")
            return

        print(" Pre-casting layer dtypes (one-time setup)...")
        casted_layers = 0
        casts_done_params = 0
        casts_done_buffers = 0

        for i, layer in enumerate(layers):
            if i not in cpu_swappable_layers:
                continue

            layer_casted = False

            # --- Parameters ---
            for name, param in layer.named_parameters(recurse=True):
                if param is None or not torch.is_floating_point(param):
                    continue
                # Match the first mapping whose 'from' dtype equals current dtype
                to_dtype = None
                for src_dt, dst_dt in pairs:
                    if param.dtype == src_dt:
                        to_dtype = dst_dt
                        break
                if to_dtype is None:
                    continue

                # Safety: don’t cast to float8 here unless your kernels support it for these tensors
                if to_dtype in (getattr(torch, "float8_e4m3fn", None), getattr(torch, "float8_e5m2", None)):
                    # Skip silently or log if verbose
                    continue

                new_param = param.to(dtype=to_dtype, device=param.device)
                if new_param.dtype != param.dtype:
                    with torch.inference_mode(False), torch.no_grad():
                        owner, leaf = _owner_and_key(layer, name)
                        old = owner._parameters.get(leaf, None)
                        owner._parameters[leaf] = nn.Parameter(new_param.detach(), requires_grad=False)
                        if old is not None:
                            del old
                    layer_casted = True
                    casts_done_params += 1

            # --- Buffers (float buffers only; keep BN stats in fp32) ---
            for bname, buf in layer.named_buffers(recurse=True):
                if buf is None or not torch.is_floating_point(buf):
                    continue
                # keep batchnorm running stats in fp32
                if bname.endswith("running_mean") or bname.endswith("running_var"):
                    continue

                to_dtype = None
                for src_dt, dst_dt in pairs:
                    if buf.dtype == src_dt:
                        to_dtype = dst_dt
                        break
                if to_dtype is None:
                    continue
                if to_dtype in (getattr(torch, "float8_e4m3fn", None), getattr(torch, "float8_e5m2", None)):
                    continue

                new_buf = buf.to(dtype=to_dtype, device=buf.device)
                if new_buf.dtype != buf.dtype:
                    with torch.inference_mode(False), torch.no_grad():
                        owner, leaf = _owner_and_key(layer, bname)
                        oldb = owner._buffers.get(leaf, None)
                        if leaf in owner._buffers:
                            owner._buffers[leaf] = new_buf.detach()
                        else:
                            owner.register_buffer(leaf, new_buf.detach(), persistent=True)
                        if oldb is not None:
                            del oldb
                    layer_casted = True
                    casts_done_buffers += 1

            if layer_casted:
                casted_layers += 1

        print(f"✓ Pre-casted {casted_layers} layers "
              f"(params changed: {casts_done_params}, buffers changed: {casts_done_buffers})")


# =============================================================================
# ANALYSIS FUNCTIONS by obisin
# =============================================================================

def analyze_layer_performance():
    """Analyze which layers should be permanent GPU residents based on performance"""
    if not (hasattr(add_smart_swapping_to_layer, 'layer_transfer_times') or
            hasattr(add_smart_swapping_to_layer, 'layer_compute_times')):
        print("No performance data available yet")
        return

    print(f"\n=== LAYER PERFORMANCE ANALYSIS (Step {add_smart_swapping_to_layer.current_step}) ===")

    # Calculate current totals
    current_total_transfer = sum(sum(times) for times in add_smart_swapping_to_layer.layer_transfer_times.values())
    current_total_compute = sum(sum(times) for times in add_smart_swapping_to_layer.layer_compute_times.values())

    # Get count for averaging
    total_values = sum(len(times) for times in add_smart_swapping_to_layer.layer_transfer_times.values())
    total_values += sum(len(times) for times in add_smart_swapping_to_layer.layer_compute_times.values())

    # OVERVIEW: Show averages
    avg_total = (current_total_transfer + current_total_compute) / total_values if total_values > 0 else 0
    avg_transfer = current_total_transfer / total_values if total_values > 0 else 0
    avg_compute = current_total_compute / total_values if total_values > 0 else 0

    gpu_mem_gb = torch.cuda.memory_allocated() / 1024 ** 3 if torch.cuda.is_available() else 0

    current_time = time.time()
    total_elapsed = current_time - add_smart_swapping_to_layer.total_start_time
    steps_completed = add_smart_swapping_to_layer.current_step
    seconds_per_iteration = total_elapsed / steps_completed if steps_completed > 0 else 0

    print(f"OVERVIEW: CPU {avg_total:.1f}ms | Transfer {avg_transfer:.1f}ms | Compute {avg_compute:.1f}ms | GPU: {gpu_mem_gb:.2f}GB")
    print(f"TIMING: {seconds_per_iteration:.2f}s/it ({steps_completed} steps in {total_elapsed:.1f}s)")
    print(f"STATS: GPU Swaps {swap_stats['to_gpu']} | CPU Swaps {swap_stats['to_cpu']} | Prefetch Hits {getattr(add_smart_swapping_to_layer, 'prefetch_hits', 0)} | Prefetch Calls {getattr(add_smart_swapping_to_layer, 'prefetch_misses', 0)}")

    combined_data = {}

    if hasattr(add_smart_swapping_to_layer, 'layer_transfer_times'):
        for layer_idx, transfer_times in add_smart_swapping_to_layer.layer_transfer_times.items():
            if transfer_times:
                avg_transfer = sum(transfer_times) / len(transfer_times)
                combined_data[layer_idx] = {'event_transfer_time': avg_transfer}

    if hasattr(add_smart_swapping_to_layer, 'layer_compute_times'):
        for layer_idx, compute_times in add_smart_swapping_to_layer.layer_compute_times.items():
            if compute_times:
                avg_compute = sum(compute_times) / len(compute_times)
                if layer_idx in combined_data:
                    combined_data[layer_idx]['event_compute_time'] = avg_compute
                else:
                    combined_data[layer_idx] = {'event_compute_time': avg_compute}

    if combined_data:
        # Calculate threshold from accumulated averages
        all_cpu_times = [data.get('event_transfer_time', 0) + data.get('event_compute_time', 0) for data in
                         combined_data.values()]
        avg_cpu = sum(all_cpu_times) / len(all_cpu_times)
        std_cpu = (sum((x - avg_cpu) ** 2 for x in all_cpu_times) / len(all_cpu_times)) ** 0.5
        cpu_threshold = avg_cpu + std_cpu

        print(f"Threshold (avg + 1 std): {cpu_threshold:.1f}ms")
        print()

        # Show problematic layers
        problematic_layers = []
        for layer_idx in sorted(combined_data.keys()):
            data = combined_data[layer_idx]

            event_transfer = data.get('event_transfer_time', 0)
            event_compute = data.get('event_compute_time', 0)
            layer_total = event_transfer + event_compute

            if layer_total > cpu_threshold:
                bottleneck = "TRANSFER" if event_transfer > event_compute else "COMPUTE"
                status = "RESIDENT" if layer_idx in gpu_resident_layers else "SWAPPABLE"

                print(f"Layer {layer_idx:2d}: CPU {layer_total:5.1f}ms | Transfer {event_transfer:4.1f}ms | Compute {event_compute:5.1f}ms | {bottleneck:8s} | {status}")

                problematic_layers.append(str(layer_idx))

        if problematic_layers:
            current_residents = sorted(gpu_resident_layers)
            recommended = current_residents + [int(x) for x in problematic_layers]
            print(f"\nRECOMMENDED LAYERS FOR GPU: \"{','.join(map(str, sorted(set(recommended))))}\"")
            print("No. of Layers: ", len(recommended))

    print("=" * 80)

def print_memory_optimization_analysis(layers, args):
    """
    Streamlined memory analysis for DGLS Dynamic Swapping Loader.
    Provides clear, actionable recommendations without unnecessary detail.
    """
    import platform
    import psutil
    import torch
    import time
    import comfy.model_management

    available_for_layers = 0
    gpu_memory_used = 0
    unused_memory = 0
    swapping_overhead = 0
    total_vram = 0
    inference_memory = 0

    # Check if this is first run or per-step update
    is_startup = not hasattr(print_memory_optimization_analysis, 'has_run')
    if is_startup:
        print_memory_optimization_analysis.has_run = True
        print("\n" + "=" * 80)
        print(" " * 20 + "DGLS MEMORY OPTIMIZATION ANALYSIS")
        print("=" * 80)
    else:
        print("\n" + "-" * 80)
        print(" " * 25 + "STEP PERFORMANCE UPDATE")
        print("-" * 80)

    # ========================================================================
    # SYSTEM ANALYSIS (startup only)
    # ========================================================================
    if is_startup:
        print("\n--- SYSTEM ANALYSIS ---")

        if torch.cuda.is_available():
            gpu_props = torch.cuda.get_device_properties(0)
            total_vram = gpu_props.total_memory
            free_vram, _ = torch.cuda.mem_get_info(0)
            used_vram = total_vram - free_vram

            print(f"\nGPU DEVICE:")
            print(f"  Name: {gpu_props.name}")
            print(f"  Total VRAM: {total_vram / 1024 ** 3:.2f} GB")
            print(f"  Free VRAM: {free_vram / 1024 ** 3:.2f} GB")
            print(f"  Used VRAM: {used_vram / 1024 ** 3:.2f} GB ({used_vram / total_vram * 100:.1f}%)")
        else:
            print("\nWARNING: No CUDA device available!")
            total_vram = 0
            free_vram = 0

        cpu_mem = psutil.virtual_memory()
        print(f"\nSYSTEM MEMORY:")
        print(f"  Total RAM: {cpu_mem.total / 1024 ** 3:.2f} GB")
        print(f"  Available RAM: {cpu_mem.available / 1024 ** 3:.2f} GB")

    # ========================================================================
    # LAYER ANALYSIS
    # ========================================================================
    if is_startup:
        print("\n--- MODEL LAYER ANALYSIS ---")

        layer_sizes_bytes = []
        for i, layer in enumerate(layers):
            size_bytes = sum(p.numel() * p.element_size() for p in layer.parameters())
            layer_sizes_bytes.append(size_bytes)

        total_model_size = sum(layer_sizes_bytes)
        avg_layer_size = total_model_size / len(layers) if layers else 0
        max_layer_size = max(layer_sizes_bytes) if layer_sizes_bytes else 0
        min_layer_size = min(layer_sizes_bytes) if layer_sizes_bytes else 0

        print(f"\nMODEL STATISTICS:")
        print(f"  Total layers: {len(layers)}")
        print(f"  Model size: {total_model_size / 1024 ** 3:.2f} GB")
        print(f"  Average layer: {avg_layer_size / 1024 ** 2:.1f} MB")
        print(f"  Max layer size: {max_layer_size / 1024 ** 2:.1f} MB")
        print(f"  Min layer size: {min_layer_size / 1024 ** 2:.1f} MB")
    else:
        # Recalculate for step updates
        layer_sizes_bytes = [sum(p.numel() * p.element_size() for p in layer.parameters())
                             for layer in layers]
        avg_layer_size = sum(layer_sizes_bytes) / len(layers) if layers else 0

    # ========================================================================
    # MEMORY BREAKDOWN
    # ========================================================================
    # if is_startup:
    #     print("\n--- MEMORY BREAKDOWN ---")
    #
    #     if torch.cuda.is_available():
    #         # Get actual memory state
    #         stats = torch.cuda.memory_stats(0)
    #         mem_free_cuda, _ = torch.cuda.mem_get_info(0)
    #         mem_free_torch = stats['reserved_bytes.all.current'] - stats['active_bytes.all.current']
    #         total_free = mem_free_cuda + mem_free_torch
    #
    #         device = comfy.model_management.get_torch_device()
    #         total_free = comfy.model_management.get_free_memory(device)
    #
    #         # reserved_memory = comfy.model_management.extra_reserved_memory()  # OS/driver overhead
    #         inference_memory = comfy.model_management.minimum_inference_memory()  # Base inference needs
    #
    #         # Calculate layer sizes
    #         layer_sizes_bytes = []
    #         for i, layer in enumerate(layers):
    #             size_bytes = sum(p.numel() * p.element_size() for p in layer.parameters())
    #             layer_sizes_bytes.append(size_bytes)
    #
    #         # Calculate swapping overhead
    #         gpu_layers = gpu_resident_layers.copy()
    #         swappable_layers = cpu_swappable_layers.copy()
    #
    #         swapping_overhead = 0
    #         prefetch_overhead = 0
    #         threading_overhead = 0
    #         max_swappable_size = 0
    #         if len(swappable_layers) > 0:
    #             max_swappable_size = max(layer_sizes_bytes[i] for i in swappable_layers)
    #             prefetch_overhead = (args.prefetch+1) * max_swappable_size * 1.25
    #             threading_overhead = prefetch_overhead * 2
    #
    #         if args.cpu_threading:
    #             swapping_overhead = prefetch_overhead + threading_overhead
    #         else:
    #             swapping_overhead = prefetch_overhead
    #
    #         # Calculate available memory step by step
    #         after_inference = total_free - inference_memory
    #         after_sampling = after_inference
    #         available_for_layers = max(0, after_sampling - swapping_overhead)
    #
    #         print(f"MEMORY CALCULATION:")
    #         print(f"  Total free memory: {total_free / 1024 ** 3:.2f} GB")
    #         # print(f"  Reserved (OS/driver): {reserved_memory / 1024 ** 3:.2f} GB")
    #         print(f"  Minus inference overhead: -{inference_memory / 1024 ** 3:.2f} GB = {after_inference / 1024 ** 3:.2f} GB")
    #         print(f"  Minus swapping overhead: -{swapping_overhead / 1024 ** 3:.2f} GB = {available_for_layers / 1024 ** 3:.2f} GB")
    #         # layers_min = calculate_auto_gpu_layers(layers, args)
    #         if max_swappable_size > 0:
    #             print(f"  RECOMMENDED AMOUNT OF STARTING LAYERS: {(available_for_layers/ 1024 ** 3) / ((max_swappable_size/ 1024 ** 3) * 2.1):.1f} Layers with prefetch=0")
    #         # print(f"  RECOMMENDED AMOUNT OF STARTING LAYERS: {len(layers_min)} Layers for current settings")
    #
    #
    #         # Current allocation
    #         gpu_memory_used = sum(layer_sizes_bytes[i] for i in gpu_layers) if gpu_layers else 0
    #         unused_memory = available_for_layers - gpu_memory_used
    #
    #         print(f"\nCURRENT MEMORY ALLOCATION:")
    #         print(f"  GPU layers: {len(gpu_layers)}/{len(layers)} (indices: {sorted(list(gpu_layers))})")
    #         print(f"  Memory Used for Layers: {gpu_memory_used / 1024 ** 3:.2f} GB")
    #         print(f"  Memory left for more layers: {available_for_layers / 1024 ** 3:.2f} GB")
    #
    #         print()
    #     else:
    #         available_for_layers = 0
    #         gpu_memory_used = 0
    #         unused_memory = 0
    #         swapping_overhead = 0

    # else:
    #     # ====================================================================
    #     # STEP UPDATES (only during steps)
    #     # ====================================================================
    #     print("\n--- CURRENT GPU MEMORY ---")
    #
    #     if torch.cuda.is_available():
    #         free_vram, _ = torch.cuda.mem_get_info(0)
    #         allocated_memory = torch.cuda.memory_allocated(0)
    #         reserved_memory = torch.cuda.memory_reserved(0)
    #
    #         print(f"  Free VRAM: {free_vram / 1024 ** 3:.2f} GB")
    #         print(f"  Allocated: {allocated_memory / 1024 ** 3:.2f} GB")
    #         print(f"  Reserved: {reserved_memory / 1024 ** 3:.2f} GB")
    #
    #     print(f"\n--- SYSTEM MEMORY: ---")
    #     cpu_mem = psutil.virtual_memory()
    #     print(f"  Total RAM: {cpu_mem.total / 1024 ** 3:.2f} GB")
    #     print(f"  Available RAM: {cpu_mem.available / 1024 ** 3:.2f} GB")

    return {
        # 'available_memory_gb': available_for_layers / 1024 ** 3 if available_for_layers else 0,
        # 'unused_memory_gb': unused_memory / 1024 ** 3 if available_for_layers else 0,
        # 'gpu_layers': list(gpu_layers) if available_for_layers else [],
        # 'swapping_overhead_gb': swapping_overhead / 1024 ** 3 if available_for_layers else 0
    }
