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
    "batch_move": "Use batch layer moving (experimental, may cause device issues)",
    "cast_target": "Cast FROM dtype TO dtype at start-up (e.g., f32 bf16) choices=[f32, bf16, f16, f8_e4m3, f8_e5m2, nf4, fp4]",
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

layers = None

cpu_swappable_layers = set()
gpu_resident_layers = set()

GPU_DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
CPU_DEVICE = 'cpu'

print(f"Device configuration: GPU={GPU_DEVICE}, CPU={CPU_DEVICE}")
casting_handler = None

profiler_active = False
swappable_list = None

# =============================================================================
# DEBUG FUNCTIONS by obisin
# =============================================================================

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

def _has_version_counter(t: torch.Tensor) -> bool:
    try:
        _ = t._version
        return True
    except (RuntimeError, AttributeError):
        return False

def fix_inference_tensor_parameters(layer):
    """
    Ensure all tensors owned by this module (params, buffers, common plain-tensor attrs)
    have a normal version counter (i.e., were not created under inference_mode()).
    Run with inference_mode(False) + no_grad before wrapping forward.
    """
    fixed = False
    with torch.inference_mode(False), torch.no_grad():
        # 1) Parameters: re-register as fresh Parameters if needed
        for name, p in list(layer.named_parameters(recurse=False)):
            if p is not None and not _has_version_counter(p):
                new_p = p.detach().clone().to(p.device, p.dtype)
                setattr(layer, name, torch.nn.Parameter(new_p, requires_grad=False))
                fixed = True

        # 2) Buffers: re-register if needed
        for name, b in list(layer.named_buffers(recurse=False)):
            if b is not None and not _has_version_counter(b):
                new_b = b.detach().clone().to(b.device, b.dtype)
                layer.register_buffer(name, new_b, persistent=True)
                fixed = True

        # 3) Plain tensor attributes used by WAN/Flux blocks
        for attr_name in ("modulation", "freqs", "pe", "vec", "norm", "scale"):
            if hasattr(layer, attr_name):
                t = getattr(layer, attr_name)
                if isinstance(t, torch.Tensor) and not _has_version_counter(t):
                    new_t = t.detach().clone().to(t.device, t.dtype)
                    setattr(layer, attr_name, new_t)
                    fixed = True

    return fixed


def get_cached_layer_size_mb(idx):
    size = layer_sizes_mb.get(idx, 0)
    if size == 0:
        print(f"WARNING: Layer {idx} size not found in cache")
    return size

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

def _pin_layer_cpu_tensors(layer):
    """Pin CPU tensors for real async transfers"""
    with torch.no_grad():
        for name, p in list(layer.named_parameters(recurse=False)):
            if p.device.type == 'cpu' and not p.is_pinned():
                pinned = p.detach().pin_memory()
                setattr(layer, name, nn.Parameter(pinned, requires_grad=False))
        for name, b in list(layer.named_buffers(recurse=False)):
            if b.device.type == 'cpu' and not b.is_pinned():
                layer.register_buffer(name, b.detach().pin_memory(), persistent=True)


def calculate_auto_gpu_layers(layers, args):
    """Auto-select GPU layers allocation logic with safety mechanisms"""

    device = comfy.model_management.get_torch_device()

    # Step 1: Cleanup existing models
    comfy.model_management.cleanup_models_gc()

    # Step 2: Calculate total memory required with safety multiplier
    total_model_size = sum(comfy.model_management.module_size(layer) for layer in layers)
    memory_required_with_safety = total_model_size * 1.1
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
        overhead = max_layer_size * 1.1 * (args.prefetch + 1)

        if args.cpu_threading:
            overhead += max_layer_size * 0.33 * (args.prefetch + 1)

        if args.cuda_streams:
            overhead += max_layer_size * 0.33 * (args.prefetch + 1)

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
        pos_map = {v: i for i, v in enumerate(swappable_list)}  # O(n) once
        base = pos_map.get(layer_idx)
        if base is not None:
            n = len(swappable_list)
            def order(idx):
                pos = pos_map.get(idx)
                return ((pos - base) % n) if pos is not None else (n + idx)
            return sorted(needed, key=order)

    # fallback: stable numeric order
    return sorted(needed)


def cleanup_excess_layers(keep_layers):
    """Remove layers from GPU that are not in keep_layers set - unified threading"""
    layers_to_remove = []
    for idx in cpu_swappable_layers:
        if (idx < len(layers) and
                idx not in keep_layers and
                device_cache.get_device(idx).type == 'cuda'):
            layers_to_remove.append(idx)

    if not layers_to_remove:
        return 0

    def unified_cleanup_transfer():
        cleaned_count = 0
        for idx in layers_to_remove:
            if args.verbose:
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()

            layers[idx].to(CPU_DEVICE)
            # move_to(layers[idx], CPU_DEVICE)
            if args.cuda_streams and args.pin_memory:
                _pin_layer_cpu_tensors(layers[idx])
            device_cache.mark_moved(idx, torch.device('cpu'))
            swap_stats['to_cpu'] += 1
            cleaned_count += 1

            if args.verbose:
                end_event.record()
                end_event.synchronize()
                transfer_time = start_event.elapsed_time(end_event)
                add_smart_swapping_to_layer.layer_transfer_times[idx].append(transfer_time)

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

    def _fetch_operation():
        layers_to_fetch = [idx for idx in needed_layers
                           if (idx < len(layers) and
                               idx in cpu_swappable_layers and
                               device_cache.get_device(idx).type == 'cpu')]

        fetched_count = 0
        transfer_start = time.perf_counter()

        for idx in layers_to_fetch:
            overlap_stats['transfer_start_times'][idx] = transfer_start

        def transfer_single_layer(idx):
            """Nested function to eliminate repetitive transfer pattern"""
            nonlocal fetched_count
            if args.verbose:
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()

            layers[idx].to(GPU_DEVICE, non_blocking=True)
            # move_to(layers[idx], GPU_DEVICE)

            transfer_event = torch.cuda.Event()
            transfer_event.record()

            if hasattr(add_smart_swapping_to_layer, 'transfer_events'):
                add_smart_swapping_to_layer.transfer_events[idx] = transfer_event
                # Cap events dict at 50 entries
                if len(add_smart_swapping_to_layer.transfer_events) > 40:
                    # Remove oldest 10 entries
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
                else:
                    for idx in layers_to_fetch:
                        transfer_single_layer(idx)
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
        def _whole_forward_operation():
            layer_compute_start = time.perf_counter()
            overlap_stats['compute_start_times'][layer_idx] = layer_compute_start

            # Detect new step when we hit layer 0 again (reset to early layers)
            # In swapped_forward, after the new step detection:
            if args.verbose and layer_idx == add_smart_swapping_to_layer.first_swappable:
                add_smart_swapping_to_layer.current_step += 1
                add_smart_swapping_to_layer.calls_this_step = 0
                if args.verbose:
                    print(f" New sampling step {add_smart_swapping_to_layer.current_step}")

                    if add_smart_swapping_to_layer.current_step % 3 == 0:
                        analyze_layer_performance()

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
                        final_device = device_cache.get_device(layer_idx)

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

            if args.verbose and add_smart_swapping_to_layer.total_forward_calls % 100 == 0:
                print_memory_optimization_analysis(layers, args)
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

                    # ALL cleanup now goes through the thread
                    def emergency_cleanup():
                        for cleanup_idx in cpu_swappable_layers:
                            if cleanup_idx != layer_idx and cleanup_idx < len(layers):
                                try:
                                    layers[cleanup_idx].to(CPU_DEVICE)
                                    # move_to(layers[cleanup_idx], CPU_DEVICE)
                                    device_cache.mark_moved(cleanup_idx, torch.device('cpu'))
                                except:
                                    pass

                    if args.cpu_threading and hasattr(add_smart_swapping_to_layer, 'cpu_thread_pool'):
                        cleanup_future = add_smart_swapping_to_layer.cpu_thread_pool.submit(emergency_cleanup)
                        cleanup_future.result()  # Wait for cleanup to complete
                    else:
                        emergency_cleanup()

                    gc.collect()
                    torch.cuda.empty_cache()

                    # Try again after cleanup - also through thread
                    def emergency_gpu_move():
                        layer.to(GPU_DEVICE)
                        # move_to(layer, GPU_DEVICE)
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
                    if not isinstance(tensor, torch.Tensor):
                        return tensor
                    # Fix inference mode tensors BEFORE moving
                    if hasattr(tensor, 'device'):
                        # Check if tensor needs inference mode fix
                        try:
                            _ = tensor._version
                            fixed_tensor = tensor
                        except (RuntimeError, AttributeError):
                            # Clone to get normal tensor
                            with torch.inference_mode(False), torch.no_grad():
                                fixed_tensor = tensor.detach()#.clone()

                        # Move to target device
                        if fixed_tensor.device != target_device:
                            return fixed_tensor.to(target_device, non_blocking=True)
                        else:
                            return fixed_tensor

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
                    with torch.inference_mode(False), torch.no_grad():
                        if args.verbose:
                            compute_start = time.perf_counter()

                        # === NEW: ensure default stream waits on H2D completion ===
                        if args.cuda_streams and layer_idx in cpu_swappable_layers:
                            if hasattr(add_smart_swapping_to_layer, 'transfer_events'):
                                ev = add_smart_swapping_to_layer.transfer_events.get(layer_idx)
                                if ev is not None:
                                    torch.cuda.current_stream().wait_event(ev)

                        # if args.cuda_streams and hasattr(add_smart_swapping_to_layer, 'compute_stream'):
                        #     with torch.cuda.stream(add_smart_swapping_to_layer.compute_stream):
                        #         # fix_inference_tensor_parameters(layer)
                        #         if is_flux_call:
                        #             if fwd_args:
                        #                 result = original_forward(*new_args, **new_kwargs)
                        #             else:
                        #                 result = original_forward(**new_kwargs)
                        #
                        #         elif is_keyword_only_call:
                        #             result = original_forward(**new_kwargs)
                        #         else:
                        #             result = original_forward(x, *new_args, **new_kwargs)
                        # else:
                        # fix_inference_tensor_parameters(layer)
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

def move_to(layer, target_device):
    """Move only weight/bias tensors, not the entire module"""
    for name, param in layer.named_parameters():
        if 'weight' in name or 'bias' in name:
            param.data = param.data.to(target_device, copy=False)
    for name, buffer in layer.named_buffers():
        buffer.data = buffer.data.to(target_device, copy=False)

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
                "initial_gpu_layers": ("INT",{"default": 1, "min": 0, "max": 100, "tooltip": TOOLTIPS["initial_gpu_layers"]}),
                "final_gpu_layers": ("INT",{"default": 1, "min": 0, "max": 100, "tooltip": TOOLTIPS["final_gpu_layers"]}),
                "auto_allocate_layers": ("BOOLEAN", {"default": False, "tooltip": "Automatically determine layer placement based on available VRAM. Overrides any other layer placement."}),
                "prefetch": ("INT", {"default": 1, "min": 0, "max": 100, "tooltip": TOOLTIPS["prefetch"]}),
                "cpu_threading": ("BOOLEAN", {"default": False, "tooltip": TOOLTIPS["cpu_threading"]}),
                "cuda_streams": ("BOOLEAN", {"default": False, "tooltip": TOOLTIPS["cuda_streams"]}),
                "pin_memory": ("BOOLEAN", {"default": False, "tooltip": TOOLTIPS["pin_memory"]}),
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


    def load_model_with_swapping(self, model, layers, initial_gpu_layers, final_gpu_layers, auto_allocate_layers,
                                 prefetch, verbose, pin_memory, gpu_layer_indices="",
                                 cast_target="",  cuda_streams=False, cpu_threading=False):
        global gpu_resident_layers, cpu_swappable_layers, device_cache

        def _teardown_swapping_state():
            """Complete cleanup of all modified layers and global state"""
            global layers, device_cache, gpu_resident_layers, cpu_swappable_layers

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



        # #FORCE first and last layer onto card regardless of settings for stability
        # if gpu_layer_indices and gpu_layer_indices.strip():
        #     specified_gpu_layers = set(map(int, gpu_layer_indices.split(',')))
        #
        #     specified_gpu_layers.add(0)  # First layer
        #     specified_gpu_layers.add(len(layers) - 1)  # Last layer
        #     if args.verbose:
        #         print(f" Automatically added layers 0 and {len(layers) - 1} for stability")
        # else:
        #     # Use initial/final logic, but ensure at least 1 initial and 1 final
        #     initial_gpu_layers = max(1, args.initial_gpu_layers)  # At least 1
        #     final_gpu_layers = max(1, args.final_gpu_layers)  # At least 1

        if args.dynamic_swapping:
            if args.verbose:
            #     print("Applying ENHANCED smart GPU allocation + dynamic swapping...")
                print(f"Setting up enhanced swapping for {len(layers)} layers...")

            if args.auto_allocate_layers:
                specified_gpu_layers = calculate_auto_gpu_layers(layers, args)
            elif args.gpu_layers and not args.auto_allocate_layers:
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
            else:
                specified_gpu_layers = None
                if args.verbose:
                    print(f"Using initial/final layer allocation: {initial_gpu_layers}/{final_gpu_layers}")

            # global current_model
            # current_model = model

            # Clear sets to prevent double-addition from previous runs
            gpu_resident_layers.clear()
            cpu_swappable_layers.clear()

            # PHASE 1: Determine which layers go where based on specified layers or initial/final counts
            for i, layer in enumerate(layers):
                if hasattr(layer, 'to'):
                    if specified_gpu_layers is not None:
                        # Use specified layer indices for GPU placement
                        if i in specified_gpu_layers:
                            try:
                                layer.to(GPU_DEVICE)
                                # move_to(layer, GPU_DEVICE)
                                gpu_resident_layers.add(i)  # This is correct - after successful move
                                if args.verbose:
                                    print(f"Layer {i} (specified) -> GPU permanent")

                                    # ADD: Check memory after each GPU layer
                                    # if torch.cuda.is_available():
                                    #     current_memory = torch.cuda.memory_allocated(0)
                                    #     layer_memory = current_memory - baseline_memory
                                    #     if args.verbose:
                                    #         print(f"Layer {i} -> GPU: +{layer_memory / 1024 ** 2:.1f}MB (total: {current_memory / 1024 ** 3:.3f}GB)")

                            except RuntimeError as e:
                                print(f"CRITICAL: Cannot fit specified layer {i} on GPU!")
                                print(f"GPU memory may be insufficient. Consider removing layer {i} from --gpu_layers")
                                raise e
                        else:
                            # Not in specified list, make swappable
                            layer.to(CPU_DEVICE)
                            if args.cuda_streams and args.pin_memory:
                                _pin_layer_cpu_tensors(layer)
                            # move_to(layer, CPU_DEVICE)
                            cpu_swappable_layers.add(i)
                            if args.verbose:
                                print(f"Layer {i} (not specified) -> CPU swappable")
                    else:
                        # Use original initial/final logic as fallback
                        if i < initial_gpu_layers:
                            # Initial layers on GPU permanently
                            try:
                                layer.to(GPU_DEVICE)
                                # move_to(layer, GPU_DEVICE)
                                gpu_resident_layers.add(i)
                                if args.verbose:
                                    print(f"Layer {i} (initial) -> GPU permanent")
                            except RuntimeError as e:
                                print(f"GPU exhausted at layer {i}, moving to CPU with swapping")
                                layer.to(CPU_DEVICE)
                                # move_to(layer, CPU_DEVICE)
                                cpu_swappable_layers.add(i)
                        elif i >= (len(layers) - final_gpu_layers):
                            # Final layers on GPU permanently
                            try:
                                layer.to(GPU_DEVICE)
                                # move_to(layer, GPU_DEVICE)
                                gpu_resident_layers.add(i)
                                if args.verbose:
                                    print(f"Layer {i} (final) -> GPU permanent")
                            except RuntimeError as e:
                                print(f"CRITICAL: Cannot fit final layer {i} on GPU!")
                                raise e
                        else:
                            # Middle layers on CPU with swapping capability
                            layer.to(CPU_DEVICE)
                            if args.cuda_streams and args.pin_memory:
                                _pin_layer_cpu_tensors(layer)
                            # move_to(layer, CPU_DEVICE)

                            cpu_swappable_layers.add(i)
                            if args.verbose:
                                print(f"Layer {i} (middle) -> CPU swappable")

            global swappable_list
            swappable_list = sorted(cpu_swappable_layers)
            global swappable_index
            swappable_index = {v: i for i, v in enumerate(swappable_list)}
            print(f"✓ {len(gpu_resident_layers)} layers permanently on GPU: {sorted(gpu_resident_layers)}")
            print(f"✓ {len(cpu_swappable_layers)} layers on CPU with smart swapping: {sorted(cpu_swappable_layers)}")

            add_smart_swapping_to_layer.first_swappable = (min(cpu_swappable_layers) if cpu_swappable_layers else None)


            if cast_target:
                # Parse the cast_target string first
                try:
                    cast_from, cast_to = cast_target.strip().split()
                    parsed_cast_target = (cast_from, cast_to)

                    casting_handler = CastingHandler()
                    casting_handler.precast_all_layers(cpu_swappable_layers, layers,
                                                       parsed_cast_target)
                except ValueError:
                    print(f"  Invalid cast_target format: '{cast_target}'. Expected: 'from_dtype to_dtype'")

            # if args.verbose:
            #     if torch.cuda.is_available():
            #         final_memory = torch.cuda.memory_allocated(0)
            #         dgls_memory = final_memory - baseline_memory
            #         print(
            #             f" DGLS TOTAL MEMORY: {dgls_memory / 1024 ** 3:.3f} GB for {len(gpu_resident_layers)} GPU layers")
            #         print(f" Average per GPU layer: {dgls_memory / len(gpu_resident_layers) / 1024 ** 2:.1f}MB")

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
                # debug_layer_locations(layers, device_cache)

            print("Calculating layer sizes...")
            global layer_sizes_mb
            for i, layer in enumerate(layers):
                layer_sizes_mb[i] = sum(p.numel() * p.element_size() for p in layer.parameters()) / (1024 * 1024)
                if args.verbose:
                    print(f"   Layer {i}: {layer_sizes_mb[i]:.1f}MB")
            print()


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

        # # Add this after casting
        # for i in [15, 20, 25]:  # Sample swappable layers
        #     for name, param in layers[i].named_parameters():
        #         print(
        #             f"CHECK: Layer {i} param {name}: {param.dtype}, size: {param.numel() * param.element_size() / 1024 ** 2:.1f}MB")

        for layer_idx in cpu_swappable_layers:
            add_smart_swapping_to_layer(
                layers[layer_idx],
                layer_idx,
                layers,
                gpu_resident_layers,
                cpu_swappable_layers)

        # for layer_idx in gpu_resident_layers:
        #     layer = layers[layer_idx]
        #     original_forward = layer.forward
        #
        #     def create_resident_forward(layer_idx, original_forward):
        #         current_args = args
        #         @wraps(original_forward)
        #         def resident_forward( *args_tuple, **kwargs):
        #             result = original_forward( *args_tuple, **kwargs)
        #             return result
        #         return resident_forward
        #
        #     layer.forward = create_resident_forward(layer_idx, original_forward)

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

    def precast_all_layers(self, cpu_swappable_layers, layers, cast_target=None):
        """Cast all layer dtypes once at startup using existing cast_to_dtype"""
        if not cast_target:
            print("  No cast_target specified, skipping precasting")
            return

        print(" Pre-casting layer dtypes (one-time setup)...")
        casted_layers = 0

        for i, layer in enumerate(layers):
            if i in cpu_swappable_layers:  # Only cast swappable layers
                layer_casted = False

                for name, param in layer.named_parameters():
                    # Use your existing function but stay on current device
                    new_param = self.cast_to_dtype(param, param.device, cast_target)  # Keep on same device
                    if new_param.dtype != param.dtype:
                        # param.data = new_param.data
                        with torch.inference_mode(False), torch.no_grad():
                            setattr(layer, name, nn.Parameter(new_param.detach(), requires_grad=False))
                        layer_casted = True

                if layer_casted:
                    casted_layers += 1

        print(f"✓ Pre-casted {casted_layers} layers using existing cast_to_dtype function")


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
    if is_startup:
        print("\n--- MEMORY BREAKDOWN ---")

        if torch.cuda.is_available():
            # Get actual memory state
            stats = torch.cuda.memory_stats(0)
            mem_free_cuda, _ = torch.cuda.mem_get_info(0)
            mem_free_torch = stats['reserved_bytes.all.current'] - stats['active_bytes.all.current']
            total_free = mem_free_cuda + mem_free_torch

            device = comfy.model_management.get_torch_device()
            total_free = comfy.model_management.get_free_memory(device)

            # reserved_memory = comfy.model_management.extra_reserved_memory()  # OS/driver overhead
            inference_memory = comfy.model_management.minimum_inference_memory()  # Base inference needs

            # Calculate layer sizes
            layer_sizes_bytes = []
            for i, layer in enumerate(layers):
                size_bytes = sum(p.numel() * p.element_size() for p in layer.parameters())
                layer_sizes_bytes.append(size_bytes)

            # Calculate swapping overhead
            gpu_layers = gpu_resident_layers.copy()
            swappable_layers = cpu_swappable_layers.copy()

            swapping_overhead = 0
            prefetch_overhead = 0
            threading_overhead = 0
            max_swappable_size = 0
            if len(swappable_layers) > 0:
                max_swappable_size = max(layer_sizes_bytes[i] for i in swappable_layers)
                prefetch_overhead = (args.prefetch+1) * max_swappable_size * 1.25
                threading_overhead = prefetch_overhead * 2

            if args.cpu_threading:
                swapping_overhead = prefetch_overhead + threading_overhead
            else:
                swapping_overhead = prefetch_overhead

            # Calculate available memory step by step
            after_inference = total_free - inference_memory
            after_sampling = after_inference
            available_for_layers = max(0, after_sampling - swapping_overhead)

            print(f"MEMORY CALCULATION:")
            print(f"  Total free memory: {total_free / 1024 ** 3:.2f} GB")
            # print(f"  Reserved (OS/driver): {reserved_memory / 1024 ** 3:.2f} GB")
            print(f"  Minus inference overhead: -{inference_memory / 1024 ** 3:.2f} GB = {after_inference / 1024 ** 3:.2f} GB")
            print(f"  Minus swapping overhead: -{swapping_overhead / 1024 ** 3:.2f} GB = {available_for_layers / 1024 ** 3:.2f} GB")
            # layers_min = calculate_auto_gpu_layers(layers, args)
            if max_swappable_size > 0:
                print(f"  RECOMMENDED AMOUNT OF STARTING LAYERS: {(available_for_layers/ 1024 ** 3) / ((max_swappable_size/ 1024 ** 3) * 2.1):.1f} Layers with prefetch=0")
            # print(f"  RECOMMENDED AMOUNT OF STARTING LAYERS: {len(layers_min)} Layers for current settings")


            # Current allocation
            gpu_memory_used = sum(layer_sizes_bytes[i] for i in gpu_layers) if gpu_layers else 0
            unused_memory = available_for_layers - gpu_memory_used

            print(f"\nCURRENT MEMORY ALLOCATION:")
            print(f"  GPU layers: {len(gpu_layers)}/{len(layers)} (indices: {sorted(list(gpu_layers))})")
            print(f"  Memory Used for Layers: {gpu_memory_used / 1024 ** 3:.2f} GB")
            print(f"  Memory left for more layers: {available_for_layers / 1024 ** 3:.2f} GB")

            print()
        else:
            available_for_layers = 0
            gpu_memory_used = 0
            unused_memory = 0
            swapping_overhead = 0

    else:
        # ====================================================================
        # STEP UPDATES (only during steps)
        # ====================================================================
        print("\n--- CURRENT GPU MEMORY ---")

        if torch.cuda.is_available():
            free_vram, _ = torch.cuda.mem_get_info(0)
            allocated_memory = torch.cuda.memory_allocated(0)
            reserved_memory = torch.cuda.memory_reserved(0)

            print(f"  Free VRAM: {free_vram / 1024 ** 3:.2f} GB")
            print(f"  Allocated: {allocated_memory / 1024 ** 3:.2f} GB")
            print(f"  Reserved: {reserved_memory / 1024 ** 3:.2f} GB")

        print(f"\n--- SYSTEM MEMORY: ---")
        cpu_mem = psutil.virtual_memory()
        print(f"  Total RAM: {cpu_mem.total / 1024 ** 3:.2f} GB")
        print(f"  Available RAM: {cpu_mem.available / 1024 ** 3:.2f} GB")

    return {
        'available_memory_gb': available_for_layers / 1024 ** 3 if available_for_layers else 0,
        'unused_memory_gb': unused_memory / 1024 ** 3 if available_for_layers else 0,
        'gpu_layers': list(gpu_layers) if available_for_layers else [],
        'swapping_overhead_gb': swapping_overhead / 1024 ** 3 if available_for_layers else 0
    }
