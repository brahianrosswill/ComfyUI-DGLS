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
        # Set defaults - these will be overridden by ComfyUI node
        self.dynamic_swapping = True
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
    "prefetch": "Number of layers to prefetch ahead (0=off, 1=single layer, 2+=multiple layers), might not work with mixed layer type models",
    "cuda_streams": "Enable CUDA streams for copy-compute overlap (needs more VRAM)",
    "cpu_threading": "Enable CPU threading for async CPU transfers (fire-and-forget cleanup)",
    "batch_move": "Use batch layer moving (experimental, may cause device issues)",
    "cast_target": "Cast FROM dtype TO dtype at start-up (e.g., f32 bf16) choices=[f32, bf16, f16, f8_e4m3, f8_e5m2, nf4, fp4]",
    "verbose": "Enable verbose output with detailed timing and transfer information"
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

device_cache = None
packed_layers = {}
layer_sizes_mb = {}

pending_gpu_transfers = {}  # {layer_idx: event}
pending_cpu_transfers = {}  # {layer_idx: event}
# VERBOSE = True

GPU_DEVICE = 'cuda'
CPU_DEVICE = 'cpu'

layers = None

cpu_swappable_layers = set()
gpu_resident_layers = set()

GPU_DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
CPU_DEVICE = 'cpu'

print(f"Device configuration: GPU={GPU_DEVICE}, CPU={CPU_DEVICE}")
casting_handler = None

# =============================================================================
# CORE FUNCTIONS by obisin
# =============================================================================


def print_memory_optimization_analysis(model, layers, args):
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
            print(f"  Used VRAM: {used_vram / 1024 ** 3:.2f} GB ({used_vram / total_vram * 100:.1f}%) (After Layer loading on GPU)")
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

            reserved_memory = comfy.model_management.extra_reserved_memory()  # OS/driver overhead
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
            if len(swappable_layers) > 0:
                max_swappable_size = max(layer_sizes_bytes[i] for i in swappable_layers)
                prefetch_overhead = args.prefetch * max_swappable_size * 1.25


            swapping_overhead = prefetch_overhead #+ threading_overhead

            # Calculate available memory step by step
            after_inference = total_free - inference_memory
            after_sampling = after_inference #- sampling_memory
            available_for_layers = max(0, after_sampling - swapping_overhead)

            print(f"MEMORY CALCULATION:")
            print(f"  Total free memory: {total_free / 1024 ** 3:.2f} GB")
            print(f"  Reserved (OS/driver): {reserved_memory / 1024 ** 3:.2f} GB")
            print(f"  Minus inference overhead: -{inference_memory / 1024 ** 3:.2f} GB = {after_inference / 1024 ** 3:.2f} GB")
            # print(f"  Minus sampling memory: -{sampling_memory / 1024 ** 3:.2f} GB = {after_sampling / 1024 ** 3:.2f} GB")
            print(f"  Minus swapping overhead: -{swapping_overhead / 1024 ** 3:.2f} GB = {available_for_layers / 1024 ** 3:.2f} GB")


            # Current allocation
            gpu_memory_used = sum(layer_sizes_bytes[i] for i in gpu_layers) if gpu_layers else 0
            unused_memory = available_for_layers - gpu_memory_used

            print(f"\nCURRENT MEMORY ALLOCATION:")
            print(f"  GPU layers: {len(gpu_layers)}/{len(layers)} (indices: {sorted(list(gpu_layers))})")
            print(f"  Memory Used for Layers: {gpu_memory_used / 1024 ** 3:.2f} GB")
            print(f"  Memory left for more layers: {available_for_layers / 1024 ** 3:.2f} GB")
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


    # ========================================================================
    # PERFORMANCE ANALYSIS
    # ========================================================================
    avg_transfer = 0
    if not is_startup:
        print("\n--- PERFORMANCE ANALYSIS ---")

        # Show compute times if available
        if hasattr(add_smart_swapping_to_layer,
                   'layer_compute_times') and add_smart_swapping_to_layer.layer_compute_times:
            compute_times = add_smart_swapping_to_layer.layer_compute_times[-20:]
            if compute_times:
                avg_compute = sum(compute_times) / len(compute_times)
                print(f"  Layer compute: {avg_compute * 1000:.1f}ms")
        else:
            print(f"  No compute measurements yet")

        # Show transfer times if available
        if transfer_stats['current_step_gpu_times']:
            avg_transfer = (sum(transfer_stats['current_step_gpu_times']) / len(transfer_stats['current_step_gpu_times'])) / (args.prefetch+1)
            print(f"  Layer transfer: {avg_transfer * 1000:.1f}ms")

        if transfer_stats['current_step_gpu_times']:
            bavg_transfer = (sum(transfer_stats['current_step_gpu_times']) / len(transfer_stats['current_step_gpu_times']))
            print(f"  Batch transfer: {bavg_transfer * 1000:.1f}ms")

        if transfer_stats['current_step_gpu_speeds']:
            avg_speed = sum(transfer_stats['current_step_gpu_speeds']) / len(
                transfer_stats['current_step_gpu_speeds'])
            print(f"  Transfer speed: {avg_speed:.0f} MB/s")

        if not is_startup:
            if len(transfer_stats['current_step_gpu_times']) > 75:
                transfer_stats['current_step_gpu_times'] = transfer_stats['current_step_gpu_times'][-75:]
                transfer_stats['current_step_cpu_times'] = transfer_stats['current_step_cpu_times'][-75:]
                transfer_stats['current_step_gpu_speeds'] = transfer_stats['current_step_gpu_speeds'][-75:]
                transfer_stats['current_step_cpu_speeds'] = transfer_stats['current_step_cpu_speeds'][-75:]

    print("\n" + "=" * 80 + "\n")

    # Return useful data
    return {
        'available_memory_gb': available_for_layers / 1024 ** 3 if available_for_layers else 0,
        'unused_memory_gb': unused_memory / 1024 ** 3 if available_for_layers else 0,
        'gpu_layers': list(gpu_layers) if available_for_layers else [],
        'swapping_overhead_gb': swapping_overhead / 1024 ** 3 if available_for_layers else 0
    }

def fix_inference_tensor_parameters(layer):
    """Fix inference tensor parameters in a layer if needed"""
    # Quick check if ANY parameters need fixing
    needs_fixing = False
    with torch.no_grad():
        for name, param in layer.named_parameters():
            if param is not None:
                try:
                    _ = param._version
                except (RuntimeError, AttributeError):
                    needs_fixing = True
                    break

    # Only fix if needed
    if needs_fixing:
        with torch.no_grad():
            for name, param in layer.named_parameters():
                if param is not None:
                    try:
                        _ = param._version
                    except (RuntimeError, AttributeError):
                        param.data = param.data.detach().clone()

    return needs_fixing

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

def calculate_needed_layers(layer_idx, prefetch):
    needed = set()
    needed.add(layer_idx)

    # Get sorted list of swappable layers
    swappable_list = sorted(cpu_swappable_layers)

    # if not hasattr(calculate_needed_layers, 'printed_list'):
    #     print(f"DEBUG: swappable_list = {swappable_list}")
    #     calculate_needed_layers.printed_list = True

    try:
        # Find current layer's position in the swappable list
        current_pos = swappable_list.index(layer_idx)

        # Take next 'prefetch' layers, wrapping around if needed
        for i in range(1, prefetch + 1):
            next_pos = (current_pos + i) % len(swappable_list)
            needed.add(swappable_list[next_pos])

    except ValueError:
        # Current layer not in swappable list - shouldn't happen but fallback to old logic
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

    # if VERBOSE and layer_idx % 10 == 0:
    #     print(f"Layer {layer_idx}: needed {len(needed)} layers {sorted(needed)} (prefetch={prefetch})")
    return needed

def cleanup_excess_layers(keep_layers):
    """Remove layers from GPU that are not in keep_layers set"""
    if PROFILE:
        nvtx.range_push(f"Cleanup_Excess_{len(cpu_swappable_layers - keep_layers)}layers")
    try:
        cleaned_count = 0
        start_time = time.time()

        # Collect all layers to move in batch
        layers_to_remove = []
        for idx in cpu_swappable_layers:
            if (idx < len(layers) and
                    idx not in keep_layers and
                    device_cache.get_device(idx).type == 'cuda'):
                layers_to_remove.append(idx)

        # Batch CPU transfer
        if layers_to_remove:
            if args.cpu_threading and hasattr(add_smart_swapping_to_layer, 'cpu_thread_pool'):
                def batch_cpu_transfer():
                    for idx in layers_to_remove:
                        layers[idx].to(CPU_DEVICE)
                        device_cache.mark_moved(idx, torch.device('cpu'))
                        swap_stats['to_cpu'] += 1

                add_smart_swapping_to_layer.cpu_thread_pool.submit(batch_cpu_transfer)
            else:
                # Blocking batch transfer
                for idx in layers_to_remove:
                    layers[idx].to(CPU_DEVICE)
                    device_cache.mark_moved(idx, torch.device('cpu'))
                    swap_stats['to_cpu'] += 1

            cleaned_count = len(layers_to_remove)

        # Clean up transfer events
        for idx in layers_to_remove:
            if (hasattr(add_smart_swapping_to_layer, 'transfer_events') and
                    idx in add_smart_swapping_to_layer.transfer_events):
                del add_smart_swapping_to_layer.transfer_events[idx]

        # Timing
        end_time = time.time()
        transfer_time = end_time - start_time
        if cleaned_count > 0 and transfer_time > 0:
            per_layer_time = transfer_time / cleaned_count
            for _ in range(cleaned_count):
                transfer_stats['current_step_cpu_times'].append(per_layer_time)

        return cleaned_count
    finally:
        if PROFILE:
            nvtx.range_pop()

def fetch_missing_layers(needed_layers, current_layer_idx=None):
    """Ensure all needed layers are on GPU, but return as soon as current layer is ready"""
    layers_to_fetch = [idx for idx in needed_layers
                       if (idx < len(layers) and
                           idx in cpu_swappable_layers and
                           device_cache.get_device(idx).type == 'cpu')]

    if PROFILE:
        nvtx.range_push(f"Fetch_Missing_{len(needed_layers)}layers")
    try:
        fetched_count = 0
        start_time = time.time()

        if layers_to_fetch:
            if args.cuda_streams and hasattr(add_smart_swapping_to_layer, 'fetch_stream'):
                with torch.cuda.stream(add_smart_swapping_to_layer.fetch_stream):
                    # Batch GPU transfer in stream
                    for idx in layers_to_fetch:
                        layers[idx].to(GPU_DEVICE, non_blocking=True)
                        transfer_event = torch.cuda.Event()
                        transfer_event.record()
                        if hasattr(add_smart_swapping_to_layer, 'transfer_events'):
                            add_smart_swapping_to_layer.transfer_events[idx] = transfer_event
                        device_cache.mark_moved(idx, torch.device('cuda'))
                        swap_stats['to_gpu'] += 1
                        fetched_count += 1
            else:
                # Batch GPU transfer without streams
                for idx in layers_to_fetch:
                    layers[idx].to(GPU_DEVICE, non_blocking=True)
                    device_cache.mark_moved(idx, torch.device('cuda'))
                    swap_stats['to_gpu'] += 1
                    fetched_count += 1

        # Wait only for current layer if specified
        if current_layer_idx is not None and current_layer_idx in layers_to_fetch:
            while True:
                try:
                    test_param = next(layers[current_layer_idx].parameters())
                    if test_param.device.type == 'cuda':
                        break
                except StopIteration:
                    break

        end_time = time.time()
        transfer_time = end_time - start_time
        if fetched_count > 0 and transfer_time > 0:
            transfer_stats['current_step_gpu_times'].append(transfer_time)
        else:
            transfer_stats['current_step_gpu_times'].append(0)

        return fetched_count
    finally:
        if PROFILE:
            nvtx.range_pop()


# =============================================================================
# DYNAMIC LAYER SWAPPING INFERENCE for Comfy-UI - by obisin
# =============================================================================

def add_smart_swapping_to_layer(layer, layer_idx, layers_list, gpu_resident_layers, cpu_swappable_layers):
    """Add swapping capability with ComfyUI coordination"""

    global layers, device_cache
    layers = layers_list
    original_forward = layer.forward

    @wraps(original_forward)
    def swapped_forward(*fwd_args, **kwargs):

        # print(f"EXECUTION_ORDER: Layer {layer_idx} called, type: {type(layer).__name__}")

        # 1. SMART LAYER MANAGEMENT - do this first

        # if layer_idx in cpu_swappable_layers and VERBOSE and layer_idx % 5 == 0:
        #     actual_device = next(layers[layer_idx].parameters()).device.type
        #     cache_device = device_cache.get_device(layer_idx).type
        #     print(f"Layer {layer_idx}: cache={cache_device}, actual={actual_device}, prefetch={args.prefetch}")

        # Detect new step when we hit layer 0 again (reset to early layers)
        if layer_idx == 0:  # Early layer indicates new step
            add_smart_swapping_to_layer.current_step += 1
            add_smart_swapping_to_layer.calls_this_step = 0
            if VERBOSE:
                print(f" New sampling step {add_smart_swapping_to_layer.current_step}")

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

                # if VERBOSE and layer_idx % 20 == 0:
                #     current_step = getattr(add_smart_swapping_to_layer, 'current_step', 0)
                #     if (
                #             add_smart_swapping_to_layer.prefetch_hits + add_smart_swapping_to_layer.prefetch_misses) > 0:
                #         hit_rate = add_smart_swapping_to_layer.prefetch_hits / (
                #                 add_smart_swapping_to_layer.prefetch_hits + add_smart_swapping_to_layer.prefetch_misses) * 100
                #         print(
                #             f" Prefetch hit rate: {hit_rate:.1f}% ({add_smart_swapping_to_layer.prefetch_hits}/{add_smart_swapping_to_layer.prefetch_hits + add_smart_swapping_to_layer.prefetch_misses})")

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
        is_flux_call = 'DoubleStream' in layer_type or 'SingleStream' in layer_type

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
        else:
            # WAN/SDXL pattern
            x = fwd_args[0] if fwd_args else None
            new_args = fwd_args[1:]
            new_kwargs = kwargs

        add_smart_swapping_to_layer.total_forward_calls += 1
        add_smart_swapping_to_layer.calls_this_step += 1

        if VERBOSE and add_smart_swapping_to_layer.total_forward_calls % 75 == 0:
            global current_model
            if 'current_model' in globals():
                print_memory_optimization_analysis(current_model, layers, args)

        if PROFILE:
            nvtx.range_push(f"Layer_{layer_idx}_{'Forward'}")
        try:

            if not add_smart_swapping_to_layer.current_forward_logged:
                current_step = getattr(add_smart_swapping_to_layer, 'current_step', 0)
                if VERBOSE:
                    print(f" Forward pass step {current_step}: layers {min(cpu_swappable_layers)}-{max(cpu_swappable_layers)}")
                add_smart_swapping_to_layer.current_forward_logged = True


            # Handle GPU failure case
            if not gpu_success:
                print(f" Layer {layer_idx} failed to be on GPU, forcing aggressive cleanup...")
                for cleanup_idx in cpu_swappable_layers:
                    if cleanup_idx != layer_idx and cleanup_idx < len(layers):
                        try:
                            layers[cleanup_idx].to('cpu')
                        except:
                            pass

                gc.collect()
                torch.cuda.empty_cache()

                # Try again after cleanup
                try:
                    layer.to(GPU_DEVICE)
                    device = torch.device(GPU_DEVICE)
                    gpu_success = True
                    print(f" Layer {layer_idx} moved to GPU after aggressive cleanup")
                except RuntimeError:
                    print(f" CRITICAL: Layer {layer_idx} cannot fit on GPU, skipping computation!")

                    return x  # Pass input through unchanged

            def move_to_device(tensor, target_device):
                # Only move device, don't touch dtype at all
                if hasattr(tensor, 'device') and tensor.device != target_device:
                    tensor = tensor.to(target_device)

                # Fix inference tensors (without dtype changes)
                if isinstance(tensor, torch.Tensor):
                    try:
                        _ = tensor._version
                        return tensor
                    except (RuntimeError, AttributeError):
                        return tensor.detach().clone()

                return tensor

            move_to_device.current_layer = layer_idx

            # Move tensors to device (respect the calling pattern you already set up)
            if is_flux_call:
                if fwd_args:
                    # SingleStreamBlock - move fwd_args and kwargs
                    new_args = tuple(move_to_device(arg, device) for arg in fwd_args)
                    new_kwargs = {k: move_to_device(v, device) for k, v in kwargs.items()}
                else:
                    # DoubleStreamBlock - move only kwargs
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

            # Right before the computation block, add:
            # if layer_idx in cpu_swappable_layers:
            #     # Verify layer is actually on GPU, not just cached as GPU
            #     try:
            #         actual_device = next(layers[layer_idx].parameters()).device
            #         cached_device = device_cache.get_device(layer_idx)
            #         if actual_device != cached_device:
            #             print(
            #                 f"DEVICE MISMATCH Layer {layer_idx}: cache says {cached_device}, actual is {actual_device}")
            #             # Force sync before continuing
            #             torch.cuda.synchronize()
            #     except StopIteration:
            #         pass

            # 4. COMPUTATION
            if PROFILE:
                nvtx.range_push(f"Compute_L{layer_idx}")
            try:
                layer_compute_start = time.time()
                if args.cuda_streams and hasattr(add_smart_swapping_to_layer, 'compute_stream'):
                    with torch.cuda.stream(add_smart_swapping_to_layer.compute_stream):
                        fix_inference_tensor_parameters(layer)
                        if is_flux_call:
                            if fwd_args:
                                result = original_forward(*new_args, **new_kwargs)
                            else:
                                result = original_forward(**new_kwargs)
                        else:
                            result = original_forward(x, *new_args, **new_kwargs)

                        # This should never be reached due to returns above
                        layer_compute_end = time.time()
                        layer_compute_time = layer_compute_end - layer_compute_start
                        add_smart_swapping_to_layer.layer_compute_times.append(layer_compute_time)

                        return result
                else:
                    fix_inference_tensor_parameters(layer)
                    if is_flux_call:
                        if fwd_args:
                            result = original_forward(*new_args, **new_kwargs)
                        else:
                            result = original_forward(**new_kwargs)
                    else:
                        result = original_forward(x, *new_args, **new_kwargs)

                # Final timing and return for non-stream path
                layer_compute_end = time.time()
                layer_compute_time = layer_compute_end - layer_compute_start
                add_smart_swapping_to_layer.layer_compute_times.append(layer_compute_time)
                return result
            finally:
                if PROFILE:
                    nvtx.range_pop()
        finally:
            if PROFILE:
                nvtx.range_pop()

    # Replace forward method
    layer.forward = swapped_forward

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
                "initial_gpu_layers": ("INT",{"default": 2, "min": 0, "max": 100, "tooltip": TOOLTIPS["initial_gpu_layers"]}),
                "final_gpu_layers": ("INT",{"default": 2, "min": 0, "max": 100, "tooltip": TOOLTIPS["final_gpu_layers"]}),
                "prefetch": ("INT", {"default": 1, "min": 0, "max": 100, "tooltip": TOOLTIPS["prefetch"]}),
                "cpu_threading": ("BOOLEAN", {"default": False, "tooltip": TOOLTIPS["cpu_threading"]}),
                "cuda_streams": ("BOOLEAN", {"default": False, "tooltip": TOOLTIPS["cuda_streams"]}),
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


    def load_model_with_swapping(self, model, layers, initial_gpu_layers, final_gpu_layers,
                                 prefetch, verbose, gpu_layer_indices="",
                                 cast_target="", cuda_streams=False, cpu_threading=False):

        global swap_stats, transfer_stats
        swap_stats = {'to_gpu': 0, 'to_cpu': 0}
        transfer_stats = {
            'to_gpu_times': [], 'to_cpu_times': [], 'to_gpu_speeds': [], 'to_cpu_speeds': [],
            'current_step_gpu_times': [], 'current_step_cpu_times': [],
            'current_step_gpu_speeds': [], 'current_step_cpu_speeds': []
        }

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
        global VERBOSE
        VERBOSE = args.verbose

        args.gpu_layers = gpu_layer_indices.strip() if gpu_layer_indices.strip() else None
        args.cast_target = cast_target.strip() if cast_target.strip() else None

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
                # torch.cuda.synchronize()  # Ensure streams are ready
                print("✓ CUDA Streams enabled for copy-compute overlap")
            except Exception as e:
                print(f"✗ CUDA Streams failed to initialize: {e}")
                args.cuda_streams = False
                print("✓ Falling back to default stream")

        # CPU THREADING: Initialize thread pool (once) - only if enabled
        if args.cpu_threading and not hasattr(add_smart_swapping_to_layer, 'cpu_thread_pool'):
            try:
                add_smart_swapping_to_layer.cpu_thread_pool = ThreadPoolExecutor(max_workers=4,
                                                                                 thread_name_prefix="cpu_transfer")
                print("✓ CPU Threading enabled")
            except Exception as e:
                print(f"✗ CPU Threading failed to initialize: {e}")
                args.cpu_threading = False
                print("✓ Falling back to default CPU transfers")

        if VERBOSE:
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


        if gpu_layer_indices and gpu_layer_indices.strip():
            specified_gpu_layers = set(map(int, gpu_layer_indices.split(',')))

            specified_gpu_layers.add(0)  # First layer
            specified_gpu_layers.add(len(layers) - 1)  # Last layer
            if VERBOSE:
                print(f" Automatically added layers 0 and {len(layers) - 1} for stability")
        else:
            # Use initial/final logic, but ensure at least 1 initial and 1 final
            initial_gpu_layers = max(1, args.initial_gpu_layers)  # At least 1
            final_gpu_layers = max(1, args.final_gpu_layers)  # At least 1

        if args.dynamic_swapping:
            if VERBOSE:
            #     print("Applying ENHANCED smart GPU allocation + dynamic swapping...")
                print(f"Setting up enhanced swapping for {len(layers)} layers...")

            if args.gpu_layers:
                try:
                    specified_gpu_layers = set(map(int, args.gpu_layers.split(',')))
                    if VERBOSE:
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
                if VERBOSE:
                    print(f"Using initial/final layer allocation: {initial_gpu_layers}/{final_gpu_layers}")


            global current_model
            current_model = model

            # Store original values before any modifications
            original_initial_gpu_layers = initial_gpu_layers
            original_final_gpu_layers = final_gpu_layers

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
                                gpu_resident_layers.add(i)  # This is correct - after successful move
                                if verbose:
                                    print(f"Layer {i} (specified) -> GPU permanent")

                                    # ADD: Check memory after each GPU layer
                                    # if torch.cuda.is_available():
                                    #     current_memory = torch.cuda.memory_allocated(0)
                                    #     layer_memory = current_memory - baseline_memory
                                    #     if VERBOSE:
                                    #         print(f"Layer {i} -> GPU: +{layer_memory / 1024 ** 2:.1f}MB (total: {current_memory / 1024 ** 3:.3f}GB)")

                            except RuntimeError as e:
                                print(f"CRITICAL: Cannot fit specified layer {i} on GPU!")
                                print(f"GPU memory may be insufficient. Consider removing layer {i} from --gpu_layers")
                                raise e
                        else:
                            # Not in specified list, make swappable
                            layer.to(CPU_DEVICE)
                            cpu_swappable_layers.add(i)
                            if verbose:
                                print(f"Layer {i} (not specified) -> CPU swappable")
                    else:
                        # Use original initial/final logic as fallback
                        if i < initial_gpu_layers:
                            # Initial layers on GPU permanently
                            try:
                                layer.to(GPU_DEVICE)
                                gpu_resident_layers.add(i)  # MOVED: Only add if successful
                                if verbose:
                                    print(f"Layer {i} (initial) -> GPU permanent")
                            except RuntimeError as e:
                                print(f"GPU exhausted at layer {i}, moving to CPU with swapping")
                                layer.to(CPU_DEVICE)
                                cpu_swappable_layers.add(i)
                        elif i >= (len(layers) - final_gpu_layers):
                            # Final layers on GPU permanently
                            try:
                                layer.to(GPU_DEVICE)
                                gpu_resident_layers.add(i)  # MOVED: Only add if successful
                                if verbose:
                                    print(f"Layer {i} (final) -> GPU permanent")
                            except RuntimeError as e:
                                print(f"CRITICAL: Cannot fit final layer {i} on GPU!")
                                raise e
                        else:
                            # Middle layers on CPU with swapping capability
                            layer.to(CPU_DEVICE)
                            cpu_swappable_layers.add(i)
                            if verbose:
                                print(f"Layer {i} (middle) -> CPU swappable")

            print(f"✓ {len(gpu_resident_layers)} layers permanently on GPU: {sorted(gpu_resident_layers)}")
            print(f"✓ {len(cpu_swappable_layers)} layers on CPU with smart swapping: {sorted(cpu_swappable_layers)}")

            # if VERBOSE:
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

            if VERBOSE:
                analysis_result = print_memory_optimization_analysis(model, layers, args)

            global device_cache
            device_cache = LayerDeviceCache(model, layers)

            print("Calculating layer sizes...")
            global layer_sizes_mb
            for i, layer in enumerate(layers):
                layer_sizes_mb[i] = sum(p.numel() * p.element_size() for p in layer.parameters()) / (1024 * 1024)
                if VERBOSE:
                    print(f"   Layer {i}: {layer_sizes_mb[i]:.1f}MB")

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
                    if VERBOSE:
                        print(f"   Layer {i}: {layer_sizes_mb[i]:.1f}MB")

                print("===============================\n")

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

        for layer_idx in gpu_resident_layers:
            layer = layers[layer_idx]
            original_forward = layer.forward

            def create_resident_forward(layer_idx, original_forward):
                current_args = args
                @wraps(original_forward)
                def resident_forward( *args_tuple, **kwargs):

                    # if VERBOSE:
                    layer_compute_start = time.time()

                    result = original_forward( *args_tuple, **kwargs)

                    # if VERBOSE:
                    layer_compute_end = time.time()
                    layer_compute_time = layer_compute_end - layer_compute_start
                    add_smart_swapping_to_layer.layer_compute_times.append(layer_compute_time)
                    return result

                return resident_forward

                layer.forward = create_resident_forward(layer_idx, original_forward)

        if VERBOSE:
            print("✓ Dynamic swapping successfully integrated with ComfyUI")

        return (model,)

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
                        param.data = new_param.data
                        layer_casted = True

                if layer_casted:
                    casted_layers += 1

        print(f"✓ Pre-casted {casted_layers} layers using existing cast_to_dtype function")
