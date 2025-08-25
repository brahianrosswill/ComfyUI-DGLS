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
        self.threading = False       # Enable background threading for automatic layer management
        self.cuda_streams = False    # Enable CUDA streams for copy-compute overlap (needs more VRAM)
        self.batch_move = False      # Use batch layer moving (experimental, may cause device issues)
        self.cast_target = None      # Cast FROM dtype TO dtype at start-up (e.g., f32 bf16)
        self.selective_packing = False   # Size threshold in MB for using packed transfers (default: 64MB)
        self.packing_threshold_mb = 64
        self.event_sync = False      # Use CUDA events instead of torch.cuda.synchronize() for better performance
        self.verbose = False         # Enable verbose output with detailed timing and transfer information
        self.device_sync_mode = 'off' # Not needed for inference
        self.sync_only_on_resume = False
        self.compile = False
        self.autocast = None
        self.mixed_precision = 'auto'
        self.compute_casting = "disabled"

args = Args()
TOOLTIPS = {
    "dynamic_swapping": "Smart dynamic layer swapping between GPU and CPU for optimal performance",
    "initial_gpu_layers": "Number of initial layers to keep permanently on GPU. If not specified, uses reasonable defaults based on estimated VRAM",
    "final_gpu_layers": "Number of final layers to keep permanently on GPU",
    "gpu_layers": "Comma-separated list of layer indices to keep permanently on GPU (e.g., '0,1,2,14,18,19,20,21,22'). Overrides initial_gpu_layers and final_gpu_layers",
    "prefetch": "Number of layers to prefetch ahead (0=off, 1=single layer, 2+=multiple layers), might not work with mixed layer type models",
    "threading": "Enable background threading for automatic layer management CAUTION: Can be unstable in some systems or with some models",
    "cuda_streams": "Enable CUDA streams for copy-compute overlap (needs more VRAM)",
    "batch_move": "Use batch layer moving (experimental, may cause device issues)",
    "cast_target": "Cast FROM dtype TO dtype at start-up (e.g., f32 bf16) choices=[f32, bf16, f16, f8_e4m3, f8_e5m2, nf4, fp4]", #DISABLED DUE TO COMFY AUTOCAST
    "selective_packing": "Size threshold in MB for using packed transfers (default: 64MB)",
    "event_sync": "Use CUDA events instead of torch.cuda.synchronize() for better performance",
    "compile": "Enable torch.compile optimization for permanent GPU layers (experimental)",
    "verbose": "Enable verbose output with detailed timing and transfer information"
}

# =============================================================================
# GLOBAL VARIABLES
# =============================================================================
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
    # LAYER ANALYSIS (startup only)
    # ========================================================================
    if is_startup:
        print("\n--- MODEL LAYER ANALYSIS ---")

        layer_sizes_bytes = []
        for i, layer in enumerate(layers):
            size_bytes = sum(p.numel() * p.element_size() for p in layer.parameters())
            layer_sizes_bytes.append(size_bytes)

        total_model_size = sum(layer_sizes_bytes)
        avg_layer_size = total_model_size / len(layers) if layers else 0

        print(f"\nMODEL STATISTICS:")
        print(f"  Total layers: {len(layers)}")
        print(f"  Model size: {total_model_size / 1024 ** 3:.2f} GB")
        print(f"  Average layer: {avg_layer_size / 1024 ** 2:.1f} MB")
    else:
        # Recalculate for step updates
        layer_sizes_bytes = [sum(p.numel() * p.element_size() for p in layer.parameters())
                             for layer in layers]
        avg_layer_size = sum(layer_sizes_bytes) / len(layers) if layers else 0

    # ========================================================================
    # MEMORY BREAKDOWN
    # ========================================================================
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
        if len(swappable_layers) > 0:
            max_swappable_size = max(layer_sizes_bytes[i] for i in swappable_layers)
            prefetch_overhead = args.prefetch * max_swappable_size * 1.1

            if args.threading:
                # Threading needs more overhead
                prefetch_overhead = args.prefetch * max_swappable_size * 1.1 if args.prefetch > 0 else 0
                threading_overhead = max_swappable_size * 0.5  # 50% of largest layer for threading safety
            else:
                # Sequential is more memory efficient
                prefetch_overhead = args.prefetch * max_swappable_size * 1.1 if args.prefetch > 0 else 0
                threading_overhead = 0

            swapping_overhead = prefetch_overhead + threading_overhead

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
        print(f"  AVAILABLE MEMORY: {available_for_layers / 1024 ** 3:.2f} GB")

        # Current allocation
        gpu_memory_used = sum(layer_sizes_bytes[i] for i in gpu_layers) if gpu_layers else 0
        unused_memory = available_for_layers - gpu_memory_used

        print(f"\nCURRENT ALLOCATION:")
        print(f"  GPU layers: {len(gpu_layers)}/{len(layers)} (indices: {sorted(list(gpu_layers))})")
        print(f"  Memory Used: {gpu_memory_used / 1024 ** 3:.2f} GB of {available_for_layers / 1024 ** 3:.2f} GB available")
        # print(f"  Remaining Memory: {unused_memory / 1024 ** 3:.2f} GB")

    else:
        available_for_layers = 0
        gpu_memory_used = 0
        unused_memory = 0
        swapping_overhead = 0

    # ========================================================================
    # OPTIMIZATION OPTIONS
    # ========================================================================
    # if available_for_layers > 0:
    #     print("\n--- OPTIMIZATION OPTIONS ---")
    #
    #     swappable_count = len(cpu_swappable_layers)
    #     if swappable_count > 0:
    #         max_swappable_size = max(layer_sizes_bytes[i] for i in cpu_swappable_layers)
    #
    #         print(f"\nSWAPPING SETTING COSTS:")
    #         print(f"  Add 1 GPU layer: costs {avg_layer_size / 1024 ** 2:.0f}MB")
    #         print(f"  Prefetch +1: costs {max_swappable_size * 1.1 / 1024 ** 2:.0f}MB")
    #         print(f"  Enable threading: costs {max_swappable_size * 0.5 / 1024 ** 2:.0f}MB")
    #
    #         print(f"\nCURRENT SETTINGS:")
    #         print(f"  prefetch={args.prefetch}, threading={args.threading}, batch_move={args.batch_move}")
    #         print(f"  Total overhead: {swapping_overhead / 1024 ** 2:.0f}MB")

    # ========================================================================
    # OPTIMIZATION OPTIONS
    # ========================================================================
    if available_for_layers > 0:
        print("\n--- OPTIMIZATION OPTIONS ---")

        swappable_count = len(cpu_swappable_layers)
        if swappable_count > 0:
            max_swappable_size = max(layer_sizes_bytes[i] for i in cpu_swappable_layers)

            print(f"\nSWAPPING SETTING COSTS:")
            print(f"  Add 1 GPU layer: costs {avg_layer_size / 1024 ** 2:.0f}MB")
            print(f"  Prefetch +1: costs {max_swappable_size * 1.1 / 1024 ** 2:.0f}MB")
            print(f"  Enable threading: costs {max_swappable_size * 0.5 / 1024 ** 2:.0f}MB")

            print(f"\nCURRENT SETTINGS:")
            print(f"  prefetch={args.prefetch}, threading={args.threading}, batch_move={args.batch_move}")
            print(f"  Total overhead: {swapping_overhead / 1024 ** 2:.0f}MB")

            print(f"\nOPTIMAL LAYER ALLOCATION:")

            # Show both threading scenarios
            for threading_mode in [False, True]:
                threading_label = "WITH THREADING" if threading_mode else "WITHOUT THREADING"
                current_marker = " ← current" if threading_mode == args.threading else ""

                print(f"\n  {threading_label}{current_marker}:")

                # Calculate threading overhead for this mode
                threading_overhead = max_swappable_size * 0.5 if threading_mode else 0

                # Show prefetch impact for this threading mode
                for test_prefetch in [0, 1, 2, 3, 4, 5]:
                    prefetch_overhead = test_prefetch * max_swappable_size * 1.1 if test_prefetch > 0 else 0
                    total_overhead = prefetch_overhead + threading_overhead

                    # Memory available for permanent GPU layers after overhead
                    memory_for_permanent_layers = available_for_layers - total_overhead

                    # Calculate optimal layers based on worst case (largest layer size)
                    optimal_permanent_layers = max(0, int(memory_for_permanent_layers / max_swappable_size))

                    # Mark current settings
                    prefetch_marker = " ← current" if test_prefetch == args.prefetch and threading_mode == args.threading else ""

                    print(
                        f"    prefetch={test_prefetch}: {optimal_permanent_layers} layers (overhead: {total_overhead / 1024 ** 2:.0f}MB){prefetch_marker}")



    # ========================================================================
    # MEMORY ALLOCATION BREAKDOWN
    # ========================================================================

    print(f"\n--- MEMORY ALLOCATION BREAKDOWN ---")
    print(f"CURRENT USAGE:")
    print(f"  GPU layers: {len(gpu_layers)}/{len(layers)} using {gpu_memory_used / 1024 ** 3:.2f} GB")
    print(f"  Swapping overhead: {swapping_overhead / 1024 ** 3:.2f} GB")
    print(f"  Total committed: {(gpu_memory_used + swapping_overhead) / 1024 ** 3:.2f} GB")
    # print(f"  Available for allocation: {available_for_layers / 1024 ** 3:.2f} GB")
    print(f"  Remaining Memory: {(available_for_layers - gpu_memory_used - swapping_overhead) / 1024 ** 3:.2f} GB")


    true_unused = available_for_layers - gpu_memory_used - swapping_overhead
    if true_unused > 0:
        layers_that_fit = int(true_unused / avg_layer_size)
        print(f"\nUSABLE MEMORY: {true_unused / 1024 ** 3:.2f} GB can fit {layers_that_fit} more layers")
    else:
        print(f"\nWARNING: Over-allocated by {abs(true_unused) / 1024 ** 3:.2f} GB")

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
            avg_transfer = sum(transfer_stats['current_step_gpu_times']) / len(transfer_stats['current_step_gpu_times'])
            print(f"  Layer transfer: {avg_transfer * 1000:.1f}ms")

        if transfer_stats['current_step_gpu_speeds']:
            avg_speed = sum(transfer_stats['current_step_gpu_speeds']) / len(
                transfer_stats['current_step_gpu_speeds'])
            print(f"  Transfer speed: {avg_speed:.0f} MB/s")

        # Only show ratio if both are available
        if hasattr(add_smart_swapping_to_layer,
                   'layer_compute_times') and add_smart_swapping_to_layer.layer_compute_times and transfer_stats['current_step_gpu_times']:
            compute_times = add_smart_swapping_to_layer.layer_compute_times[-20:]
            if compute_times and avg_transfer > 0:
                avg_compute = sum(compute_times) / len(compute_times)
                transfer_compute_ratio = avg_transfer / avg_compute
                print(f"  Transfer/Compute ratio: {transfer_compute_ratio:.1f}x")

                if args.threading:
                    overlap_percent = max(0, (1 - 1 / transfer_compute_ratio) * 100)
                    print(f"  Current threading overlap: ~{overlap_percent:.0f}%")

                if args.threading:
                    if transfer_compute_ratio <= 3.0:
                        print(
                            f"  GOOD FOR THREADING: {avg_transfer * 1000:.0f}ms transfer vs {avg_compute * 1000:.0f}ms compute")
                        print(f"  → Threading can overlap ~{(1 - 1 / transfer_compute_ratio) * 100:.0f}% of transfer time")
                        print(f"  → Consider enabling threading for better performance")
                        print(f"  → Increase compute time: larger images, higher batch size, more steps")

                    elif transfer_compute_ratio <= 7.5:
                        print(f"  THREADING VIABLE: Some overlap possible")
                        print(f"  → Threading can hide ~{(1 - 1 / transfer_compute_ratio) * 100:.0f}% of transfer time")
                        print(f"  → Enable threading + increase prefetch=2-3")

                    else:
                        print(f"  THREADING LIMITED: Transfers {transfer_compute_ratio:.1f}x longer than compute")
                        print(f"  → Threading may be counter-productive due to overhead")
                        print(f"  → Focus on: more GPU layers or higher prefetch (sequential)")
                        print(f"  → Or increase compute: larger batch/image size, if you want to continue with threading")
                else:
                    if transfer_compute_ratio > 7.5:
                        print(f"  HIGH BOTTLENECK: Transfers take {transfer_compute_ratio:.1f}x compute time")
                        print(f"  → Maximize prefetch to reduce transfer frequency")
                        print(f"  → and/or add more GPU layers if memory allows")
        else:
            print(f"  No transfer measurements this step")

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
    # return layer_sizes_mb.get(idx, 0)


def event_based_sync(operation_name, idx=None):
    """Replace torch.cuda.synchronize() with specific event tracking"""
    if args.event_sync:
        event = torch.cuda.Event()
        event.record()

        key = f"{operation_name}_{idx}" if idx is not None else operation_name
        transfer_events[key] = event
        return event
    else:
        torch.cuda.synchronize()
        return None


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


def safe_move_to_cpu(layer, idx):
    """Move layer back to CPU and clean up GPU memory"""
    if PROFILE:
        nvtx.range_push(f"CPU_Transfer_L{idx}")
    try:
        try:
            current_device = next(layer.parameters()).device
            if current_device.type == 'cpu':
                return  # Already on CPU
        except StopIteration:
            # Layer has no parameters
            pass



        # if VERBOSE:
        start_time = time.time()
        layer.to(CPU_DEVICE)
        add_smart_swapping_to_layer.cleanup_stream = torch.cuda.Stream()
        device_cache.mark_moved(idx, torch.device('cpu'))
        swap_stats['to_cpu'] += 1

        end_time = time.time()
        transfer_time = end_time - start_time
        # print(f"safe_move_to_cpu: Layer {idx}: {transfer_time * 1000:.1f}ms transfer time recorded")
        if transfer_time > 0:
            layer_size_mb = get_cached_layer_size_mb(idx)
            speed_mbps = layer_size_mb / transfer_time
            transfer_stats['current_step_cpu_times'].append(transfer_time)
            transfer_stats['current_step_cpu_speeds'].append(speed_mbps)


        return True
    finally:
        if PROFILE:
            nvtx.range_pop()


def safe_move_to_gpu(layer, idx):
    """Move layer to GPU with dtype casting"""

    if PROFILE:
        nvtx.range_push(f"GPU_Transfer_L{idx}")
    try:
        try:

            current_device = device_cache.get_device(idx)
            if current_device.type == 'cuda':
                return True

            # if VERBOSE:
            start_time = time.time()
            layer.to(GPU_DEVICE, non_blocking=True)
            # event_based_sync("gpu_transfer", idx)
            device_cache.mark_moved(idx, torch.device('cuda'))
            swap_stats['to_gpu'] += 1

            end_time = time.time()
            transfer_time = end_time - start_time
            # print(f"safe_move_to_gpu: Layer {idx}: {transfer_time * 1000:.1f}ms transfer time recorded")
            if transfer_time > 0:
                layer_size_mb = get_cached_layer_size_mb(idx)
                speed_mbps = layer_size_mb / transfer_time
                transfer_stats['current_step_gpu_times'].append(transfer_time)
                transfer_stats['current_step_gpu_speeds'].append(speed_mbps)


            return True

        except RuntimeError as e:
            if "out of memory" in str(e):
                return False
            raise e
    finally:
        if PROFILE:
            nvtx.range_pop()


# THREADING: Thread-safe GPU operations
def safe_move_to_gpu_threaded(layer, idx):
    """Thread-safe move to GPU"""
    with add_smart_swapping_to_layer.gpu_lock:
        if device_cache.get_device(idx).type == 'cpu':
            return safe_move_to_gpu(layer, idx)
        return True


def safe_move_to_cpu_threaded(layer, idx):
    """Thread-safe move to CPU"""
    with add_smart_swapping_to_layer.gpu_lock:
        if layers is not None and device_cache is not None and device_cache.get_device(idx).type == 'cuda':
            return safe_move_to_cpu(layer, idx)
        return True

def stop_background_threading():
    """Stop the background threading system"""
    if hasattr(add_smart_swapping_to_layer, 'training_active'):
        add_smart_swapping_to_layer.training_active = False
        if add_smart_swapping_to_layer.background_thread:
            add_smart_swapping_to_layer.background_thread.join(timeout=1.0)
        print(" Background threading stopped")


def calculate_needed_layers(layer_idx, prefetch):
    needed = set()
    needed.add(layer_idx)

    if layers is None:
        return needed

    for i in range(1, prefetch + 1):
        prefetch_idx = layer_idx + i

        # CIRCULAR: If we go past the end, wrap to the beginning of swappable layers
        if prefetch_idx >= len(layers):
            prefetch_idx = min(cpu_swappable_layers) + (prefetch_idx - len(layers))

        if prefetch_idx in cpu_swappable_layers:
            needed.add(prefetch_idx)

    return needed


# def cleanup_excess_layers(keep_layers):
#     """Remove layers from GPU that are not in keep_layers set"""
#     # current_step = getattr(add_smart_swapping_to_layer, 'current_sampling_step', 0)
#
#     if PROFILE:
#         nvtx.range_push(f"Cleanup_Excess_{len(cpu_swappable_layers - keep_layers)}layers")
#     try:
#         if args.batch_move:
#             # Batch approach (your current code)
#             layers_to_remove = []
#             for idx in cpu_swappable_layers:
#                 if (idx < len(layers) and
#                         idx not in keep_layers and
#                         device_cache.get_device(idx).type == 'cuda'):
#                     layers_to_remove.append(idx)
#
#
#             cleaned_count = batch_safe_move_to_cpu(layers_to_remove)
#         else:
#             # Individual approach (fallback)
#             cleaned_count = 0
#             layers_to_remove = []
#             for idx in cpu_swappable_layers:
#                 if (idx < len(layers) and
#                         idx not in keep_layers and
#                         device_cache.get_device(idx).type == 'cuda'):
#                     layers_to_remove.append(idx)
#                     safe_move_to_cpu(layers[idx], idx)
#                     cleaned_count += 1
#
#         return cleaned_count
#     finally:
#         if PROFILE:
#             nvtx.range_pop()

def cleanup_excess_layers(keep_layers):
    """Remove layers from GPU that are not in keep_layers set"""
    if PROFILE:
        nvtx.range_push(f"Cleanup_Excess_{len(cpu_swappable_layers - keep_layers)}layers")
    try:
        cleaned_count = 0


        if args.batch_move:
            layers_to_remove = []
            for idx in cpu_swappable_layers:
                if (idx < len(layers) and
                        idx not in keep_layers and
                        device_cache.get_device(idx).type == 'cuda'):
                    layers_to_remove.append(idx)
            cleaned_count = batch_safe_move_to_cpu(layers_to_remove)
        else:
            # Add basic batch processing here
            start_time = time.time()
            for idx in cpu_swappable_layers:
                if (idx < len(layers) and
                        idx not in keep_layers and
                        device_cache.get_device(idx).type == 'cuda'):
                    layers[idx].to(CPU_DEVICE)
                    device_cache.mark_moved(idx, torch.device('cpu'))
                    swap_stats['to_cpu'] += 1
                    cleaned_count += 1

            # event_based_sync('cleanup_excess_layers')
            end_time = time.time()
            transfer_time = end_time - start_time

            if cleaned_count > 0:
                if transfer_time > 0:
                    per_layer_time = transfer_time / cleaned_count
                    for _ in range(cleaned_count):
                        transfer_stats['current_step_cpu_times'].append(per_layer_time)

        return cleaned_count
    finally:
        if PROFILE:
            nvtx.range_pop()


def fetch_missing_layers(needed_layers):
    """Ensure all needed layers are on GPU"""
    layers_to_fetch = [idx for idx in needed_layers
                       if (idx < len(layers) and
                           idx in cpu_swappable_layers and
                           device_cache.get_device(idx).type == 'cpu')]

    if PROFILE:
        nvtx.range_push(f"Fetch_Missing_{len(needed_layers)}layers")
    try:
        fetched_count = 0

        if args.batch_move:
            fetched_count = batch_safe_move_to_gpu(layers_to_fetch)
        else:
            start_time = time.time()
            for idx in layers_to_fetch:
                if (idx < len(layers) and
                        device_cache.get_device(idx).type == 'cpu'):
                    layers[idx].to(GPU_DEVICE, non_blocking=True)
                    device_cache.mark_moved(idx, torch.device('cuda'))
                    swap_stats['to_gpu'] += 1
                    fetched_count += 1

            end_time = time.time()
            transfer_time = end_time - start_time
            # Single sync and timing for the batch
            if fetched_count > 0:
                event_based_sync("batch_individual_transfers", None)
                if transfer_time > 0:
                    total_size_mb = sum(get_cached_layer_size_mb(idx) for idx in layers_to_fetch)
                    speed_mbps = total_size_mb / transfer_time

                    per_layer_time = transfer_time / fetched_count
                    for _ in range(fetched_count):
                        transfer_stats['current_step_gpu_times'].append(per_layer_time)
                        transfer_stats['current_step_gpu_speeds'].append(speed_mbps / fetched_count)

        return fetched_count
    finally:
        if PROFILE:
            nvtx.range_pop()


def batch_safe_move_to_gpu_packed(layer_indices, threshold_mb=64):
    """Batch move to GPU with selective packing using mega tensor"""
    if not layer_indices:
        return 0

    moved_count = 0
    start_time = time.time()

    # Separate layers by size for different packing strategies
    large_layers = []
    small_layers = []

    for idx in layer_indices:
        if (idx < len(layers) and
                device_cache.get_device(idx).type == 'cpu'):
            layer_size_mb = get_cached_layer_size_mb(idx)
            if layer_size_mb > threshold_mb and idx in packed_layers:
                large_layers.append(idx)
            else:
                small_layers.append(idx)

    # Process large layers with PackedCPUBlock + mega tensor batching
    if large_layers:
        batch_size = args.mega_tensor_size

        for i in range(0, len(large_layers), batch_size):
            batch_indices = large_layers[i:i + batch_size]

            # Create mega buffer from packed layers
            mega_buffers_by_dtype = {}
            layer_specs = []

            for idx in batch_indices:
                if idx in packed_layers:
                    packed_block = packed_layers[idx]
                    layer_spec = {'idx': idx, 'dtype_offsets': {}}

                    for dtype, buffer in packed_block.packed_buffers.items():
                        if dtype not in mega_buffers_by_dtype:
                            mega_buffers_by_dtype[dtype] = []

                        layer_spec['dtype_offsets'][dtype] = len(mega_buffers_by_dtype[dtype])
                        mega_buffers_by_dtype[dtype].append(buffer)

                    layer_specs.append(layer_spec)

            # Transfer each dtype as mega tensor
            for dtype, buffers in mega_buffers_by_dtype.items():
                if buffers:
                    # Pack into mega tensor
                    mega_buffer = torch.cat(buffers, dim=0)

                    # Single transfer for this dtype
                    mega_gpu = mega_buffer.to(GPU_DEVICE, non_blocking=True)

                    # Unpack back to individual packed blocks
                    offset = 0
                    for spec in layer_specs:
                        if dtype in spec['dtype_offsets']:
                            idx = spec['idx']
                            packed_block = packed_layers[idx]
                            buffer_size = packed_block.packed_buffers[dtype].numel()

                            # Extract this layer's portion
                            layer_gpu_buffer = mega_gpu[offset:offset + buffer_size]
                            packed_block.gpu_blocks[dtype] = layer_gpu_buffer

                            # Rebind layer parameters to GPU views
                            tensor_offset = 0
                            for tensor_spec in packed_block.tensor_specs:
                                if tensor_spec['dtype'] == dtype:
                                    start_idx = tensor_offset
                                    end_idx = start_idx + tensor_spec['size']
                                    gpu_view = layer_gpu_buffer[start_idx:end_idx].view(tensor_spec['shape'])

                                    if tensor_spec['is_param']:
                                        tensor_spec['param_ref'].data = gpu_view
                                    else:
                                        tensor_spec['buffer_ref'].data = gpu_view

                                    tensor_offset += tensor_spec['size']

                            offset += buffer_size

            # Update device cache for batch
            for idx in batch_indices:
                device_cache.mark_moved(idx, torch.device('cuda'))
                swap_stats['to_gpu'] += 1
                moved_count += 1

    # Process small layers with regular mega tensor batching
    if small_layers:
        batch_size = args.mega_tensor_size

        for i in range(0, len(small_layers), batch_size):
            batch_indices = small_layers[i:i + batch_size]
            layers_to_move = [idx for idx in batch_indices
                              if device_cache.get_device(idx).type == 'cpu']

            if layers_to_move:
                # Pack this batch of layers into single tensor
                mega_tensor, unpack_specs = pack_layers_to_mega_tensor(layers_to_move)

                # Single .to() call for this batch
                mega_gpu = mega_tensor.to(GPU_DEVICE, non_blocking=True)

                # Unpack back to individual layers
                unpack_mega_tensor_to_layers(mega_gpu, unpack_specs, layers_to_move)

                # Update device cache
                for idx in layers_to_move:
                    device_cache.mark_moved(idx, torch.device('cuda'))
                    swap_stats['to_gpu'] += 1
                    moved_count += 1

    # Single sync and timing for entire operation
    if moved_count > 0:
        event_based_sync("batch_gpu_transfer", None)
        end_time = time.time()
        transfer_time = end_time - start_time

        if transfer_time > 0:
            total_size_mb = sum(get_cached_layer_size_mb(idx) for idx in layer_indices
                                if device_cache.get_device(idx).type == 'cuda')
            speed_mbps = total_size_mb / transfer_time

            # Record per-layer timing for analysis
            per_layer_time = transfer_time / moved_count if moved_count > 0 else transfer_time
            for _ in range(moved_count):
                transfer_stats['current_step_gpu_times'].append(per_layer_time)
                transfer_stats['current_step_gpu_speeds'].append(speed_mbps / moved_count)

    return moved_count


def batch_safe_move_to_cpu_packed(layer_indices, threshold_mb=64):
    """Batch move to CPU with selective packing"""
    if not layer_indices:
        return 0

    moved_count = 0
    start_time = time.time()
    large_layers = []
    small_layers = []

    # Separate layers by size
    for idx in layer_indices:
        if (idx < len(layers) and
                device_cache.get_device(idx).type == 'cuda'):
            layer_size_mb = get_cached_layer_size_mb(idx)
            if layer_size_mb > threshold_mb:
                large_layers.append(idx)
            else:
                small_layers.append(idx)

    # Handle large layers with packed transfer
    if large_layers:
        for idx in large_layers:
            success = safe_move_to_cpu_packed(layers[idx], idx)
            if success:
                moved_count += 1

    # Handle small layers with direct transfer
    if small_layers:
        for idx in small_layers:
            success = safe_move_to_cpu(layers[idx], idx)
            if success:
                moved_count += 1


    # Single sync at the end
    if moved_count > 0:
        # event_based_sync("batch_cpu_transfer", None)
        end_time = time.time()
        transfer_time = end_time - start_time

        if transfer_time > 0:
            total_size_mb = sum(get_cached_layer_size_mb(idx) for idx in layer_indices
                                if device_cache.get_device(idx).type == 'cuda')
            speed_mbps = total_size_mb / transfer_time
            transfer_stats['current_step_gpu_times'].append(transfer_time)
            transfer_stats['current_step_gpu_speeds'].append(speed_mbps)

    return moved_count


def cleanup_excess_layers_packed(keep_layers, threshold_mb=64):
    """Remove layers from GPU with selective packing awareness"""

    if args.batch_move:
        # Collect all layers to remove, then use batch function
        layers_to_remove = []
        for idx in cpu_swappable_layers:
            if (idx < len(layers) and
                    idx not in keep_layers and
                    device_cache.get_device(idx).type == 'cuda'):
                layers_to_remove.append(idx)
        return batch_safe_move_to_cpu_packed(layers_to_remove, threshold_mb)

    else:
        # Individual processing
        moved_count = 0
        for idx in cpu_swappable_layers:
            if (idx < len(layers) and
                    idx not in keep_layers and
                    device_cache.get_device(idx).type == 'cuda'):

                layer_size_mb = get_cached_layer_size_mb(idx)

                if layer_size_mb > threshold_mb:
                    success = safe_move_to_cpu_packed(layers[idx], idx)
                else:
                    success = safe_move_to_cpu(layers[idx], idx)

                if success:
                    moved_count += 1

        return moved_count


def fetch_missing_layers_packed(needed_layers, threshold_mb=64):
    """Use packed transfers only for large layers, direct for small ones"""

    if args.batch_move:
        # Use batch function
        return batch_safe_move_to_gpu_packed(needed_layers, threshold_mb)

    else:
        # Individual processing
        fetched_count = 0
        for idx in needed_layers:
            if (idx < len(layers) and
                    idx in cpu_swappable_layers and
                    device_cache.get_device(idx).type == 'cpu'):

                layer_size_mb = get_cached_layer_size_mb(idx)

                if layer_size_mb > threshold_mb and idx in packed_layers:
                    success = safe_move_to_gpu_packed(layers[idx], idx)
                else:
                    success = safe_move_to_gpu(layers[idx], idx)

                if success:
                    fetched_count += 1

        return fetched_count


def background_layer_manager():
    """Background thread that maintains sliding window of layers"""
    last_seen_step = -1

    while add_smart_swapping_to_layer.training_active:
        try:
            # Detect new sampling step - reset and be extra aggressive
            current_step = getattr(add_smart_swapping_to_layer, 'current_sampling_step', 0)
            current_idx = add_smart_swapping_to_layer.current_layer_idx

            if current_step != last_seen_step:
                last_seen_step = current_step

                # Preload for new step using actual prefetch setting
                first_swappable = min(cpu_swappable_layers)
                next_step_needed = calculate_needed_layers(first_swappable, args.prefetch)

                for layer_idx in next_step_needed:
                    if device_cache.get_device(layer_idx).type == 'cpu':
                        safe_move_to_gpu_threaded(layers[layer_idx], layer_idx)

                #OVERLY AGGRESSIVE DEBUG VERSION
                # first_swappable = min(cpu_swappable_layers)
                # if device_cache.get_device(first_swappable).type == 'cpu':
                #     # print(f"🎯 Pre-warming first swappable layer {first_swappable}")
                #     safe_move_to_gpu_threaded(layers[first_swappable], first_swappable)
                #
                # # Aggressively preload first few layers for new step
                # for early_idx in range(min(5, len(layers))):  # First 5 layers
                #     if (early_idx in cpu_swappable_layers and
                #             device_cache.get_device(early_idx).type == 'cpu'):
                #         safe_move_to_gpu_threaded(layers[early_idx], early_idx)

            if device_cache is None or layers is None:
                time.sleep(0.1)  # Wait for initialization
                continue

            if PROFILE:
                nvtx.range_push("Background_Manager_Cycle")

            current_idx = add_smart_swapping_to_layer.current_layer_idx

            needed_layers = set()
            for future_idx in range(current_idx, min(current_idx + args.prefetch + 2, len(layers))):
                if future_idx in cpu_swappable_layers:
                    needed_layers.add(future_idx)

            # Also include the standard prefetch calculation
            standard_needed = calculate_needed_layers(current_idx, args.prefetch)
            needed_layers.update(standard_needed)

            # needed_layers = calculate_needed_layers(current_idx, args.prefetch)

            # Collect layers to move instead of moving one by one
            layers_to_cpu = []
            layers_to_gpu = []

            if args.batch_move:
                # One in, one out: remove old layers, add new ones
                for idx in cpu_swappable_layers:
                    if idx not in needed_layers and device_cache.get_device(idx).type == 'cuda':
                        layers_to_cpu.append(idx)

                for idx in needed_layers:
                    if device_cache.get_device(idx).type == 'cpu':
                        layers_to_gpu.append(idx)

                # Batch move with thread safety
                if layers_to_cpu or layers_to_gpu:
                    with add_smart_swapping_to_layer.gpu_lock:
                        if layers_to_cpu:
                            batch_safe_move_to_cpu(layers_to_cpu)
                        if layers_to_gpu:
                            batch_safe_move_to_gpu(layers_to_gpu)
            else:
                # One in, one out: remove old layers, add new ones
                for idx in cpu_swappable_layers:
                    if idx not in needed_layers and device_cache.get_device(idx).type == 'cuda':
                        safe_move_to_cpu_threaded(layers[idx], idx)

                for idx in needed_layers:
                    if device_cache.get_device(idx).type == 'cpu':
                        safe_move_to_gpu_threaded(layers[idx], idx)

            time.sleep(0.000001) #do not touch. very needed for sync

        except Exception as e:
            print(f" Background thread error: {e}")
            time.sleep(0.1)
        finally:
            if PROFILE:
                nvtx.range_pop()


def batch_safe_move_to_cpu(layer_indices):
    """Move multiple layers to CPU in batch"""
    if not layer_indices:
        return 0
    if PROFILE:
        nvtx.range_push(f"Batch_CPU_Transfer_{len(layer_indices)}layers")
    try:
        moved_count = 0
        start_time = time.time()
        for idx in layer_indices:
            if (idx < len(layers) and
                    device_cache.get_device(idx).type == 'cuda'):
                # if VERBOSE:
                layers[idx].to(CPU_DEVICE)
                device_cache.mark_moved(idx, torch.device('cpu'))
                swap_stats['to_cpu'] += 1
                moved_count += 1

        end_time = time.time()
        transfer_time = end_time - start_time

        if moved_count > 0:
            # event_based_sync("batch_cpu_transfer", None)
            if transfer_time > 0:
                total_size_mb = sum(get_cached_layer_size_mb(idx) for idx in layer_indices
                                    if device_cache.get_device(idx).type == 'cpu')
                speed_mbps = total_size_mb / transfer_time
                transfer_stats['current_step_cpu_times'].append(transfer_time)
                transfer_stats['current_step_cpu_speeds'].append(speed_mbps)


        return moved_count
    finally:
        if PROFILE:
            nvtx.range_pop()

def batch_safe_move_to_gpu(layer_indices):
    """Batch move to GPU with dtype casting"""
    if not layer_indices:
        return 0

    if PROFILE:
        nvtx.range_push(f"Batch_GPU_Transfer_{len(layer_indices)}layers")
    try:
        moved_count = 0
        # if VERBOSE:
        start_time = time.time()
        # Process in batches based on batch_size setting
        # batch_size = getattr(args, 'layer_batch_size', 3)  # Default batch 3 layers
        batch_size = args.mega_tensor_size

        for i in range(0, len(layer_indices), batch_size):
            batch_indices = layer_indices[i:i + batch_size]
            layers_to_move = [idx for idx in batch_indices
                              if device_cache.get_device(idx).type == 'cpu']

            if layers_to_move:
                # Pack this batch of layers into single tensor
                mega_tensor, unpack_specs = pack_layers_to_mega_tensor(layers_to_move)

                # Single .to() call for this batch
                mega_gpu = mega_tensor.to(GPU_DEVICE, non_blocking=True)

                # Unpack back to individual layers
                unpack_mega_tensor_to_layers(mega_gpu, unpack_specs, layers_to_move)

                # Update device cache
                for idx in layers_to_move:
                    device_cache.mark_moved(idx, torch.device('cuda'))
                    swap_stats['to_gpu'] += 1
                    moved_count += 1

        event_based_sync("batch_gpu_transfer", None)
        end_time = time.time()
        transfer_time = end_time - start_time


        if moved_count > 0 and transfer_time > 0:
            total_size_mb = sum(get_cached_layer_size_mb(idx) for idx in layer_indices
                                if device_cache.get_device(idx).type == 'cuda')
            speed_mbps = total_size_mb / transfer_time

            # Record per-layer timing for analysis
            per_layer_time = transfer_time / moved_count if moved_count > 0 else transfer_time
            for _ in range(moved_count):
                transfer_stats['current_step_gpu_times'].append(per_layer_time)
                transfer_stats['current_step_gpu_speeds'].append(speed_mbps / moved_count)

        return moved_count
    finally:
        if PROFILE:
            nvtx.range_pop()


def pack_layers_to_mega_tensor(layer_indices):
    """Pack multiple layers into single tensor"""
    all_params = []
    unpack_specs = []

    for idx in layer_indices:
        layer_params = []
        for param in layers[idx].parameters():
            param_data = param.data.cpu().flatten()
            all_params.append(param_data)
            layer_params.append({
                'shape': param.shape,
                'size': param.numel()
            })
        unpack_specs.append(layer_params)

    mega_tensor = torch.cat(all_params, dim=0)
    return mega_tensor, unpack_specs

def unpack_mega_tensor_to_layers(mega_gpu, unpack_specs, layer_indices):
    """Unpack mega tensor back to individual layer parameters"""
    offset = 0
    for layer_idx, layer_specs in zip(layer_indices, unpack_specs):
        for param, spec in zip(layers[layer_idx].parameters(), layer_specs):
            size = spec['size']
            shape = spec['shape']
            param.data = mega_gpu[offset:offset + size].view(shape)
            offset += size

# =============================================================================
# PACKED CODE by obisin
# =============================================================================

class PackedCPUBlock:
    def __init__(self, layer):
        self.packed_buffers = {}  # One buffer per dtype
        self.tensor_specs = []
        self.total_elements = 0
        self.gpu_blocks = {}  # Keep GPU buffers alive when resident
        self.gpu_events = {}  # CUDA events for synchronization
        self.pack_layer(layer)

    def pack_layer(self, layer):
        """Pack all layer parameters/buffers into contiguous CPU blocks by dtype"""
        dtype_groups = {}

        # Build param/buffer maps once (avoid rebuilding in loops)
        param_map = layer._parameters
        buffer_map = layer._buffers

        # Collect parameters by dtype
        for name, param in layer.named_parameters(recurse=False):
            if param is None:
                continue

            # Move to CPU first if needed (read-only copy during collection)
            src_data = param.detach().to('cpu', copy=True) if param.device.type != 'cpu' else param.data
            dtype = param.dtype

            if dtype not in dtype_groups:
                dtype_groups[dtype] = []
            dtype_groups[dtype].append({
                'name': name,
                'data': src_data.flatten(),
                'shape': param.shape,
                'is_param': True,
                'param_ref': param
            })

        # Collect buffers by dtype
        for name, buffer in layer.named_buffers(recurse=True):
            if buffer is None:
                continue

            # Move to CPU first if needed (read-only copy during collection)
            src_data = buffer.detach().to('cpu', copy=True) if buffer.device.type != 'cpu' else buffer.data
            dtype = buffer.dtype

            if dtype not in dtype_groups:
                dtype_groups[dtype] = []
            dtype_groups[dtype].append({
                'name': name,
                'data': src_data.flatten(),
                'shape': buffer.shape,
                'is_param': False,
                'buffer_ref': buffer
            })

        # Pack each dtype group into contiguous buffers
        for dtype, tensors in dtype_groups.items():
            if not tensors:
                continue

            total_size = sum(t['data'].numel() for t in tensors)
            packed_buffer = torch.empty(total_size, dtype=dtype)
            self.packed_buffers[dtype] = packed_buffer

            # Pack data and create specs
            offset = 0
            for tensor_info in tensors:
                data = tensor_info['data']
                size = data.numel()

                # Copy into packed buffer
                packed_buffer[offset:offset + size] = data

                # Store spec for later rebinding
                spec = {
                    'name': tensor_info['name'],
                    'dtype': dtype,
                    'offset': offset,
                    'size': size,
                    'shape': tensor_info['shape'],
                    'is_param': tensor_info['is_param'],
                    'param_ref': tensor_info.get('param_ref'),
                    'buffer_ref': tensor_info.get('buffer_ref')
                }
                self.tensor_specs.append(spec)
                offset += size

            self.total_elements += total_size

        # ATOMIC COMMIT: Rebind CPU params/buffers to views into packed_buffers
        # This eliminates duplicate host RAM
        for spec in self.tensor_specs:
            dtype = spec['dtype']
            start_idx = spec['offset']
            end_idx = start_idx + spec['size']
            cpu_view = self.packed_buffers[dtype][start_idx:end_idx].view(spec['shape'])

            if spec['is_param']:
                spec['param_ref'].data = cpu_view
            else:
                spec['buffer_ref'].data = cpu_view

    def unpack_to_gpu(self, layer):
        """Move packed buffers to GPU and rebind layer parameters"""
        # Group specs by dtype for efficient transfer
        dtype_specs = {}
        for spec in self.tensor_specs:
            dtype = spec['dtype']
            if dtype not in dtype_specs:
                dtype_specs[dtype] = []
            dtype_specs[dtype].append(spec)

        # Transfer each dtype group
        for dtype, specs in dtype_specs.items():
            if dtype not in self.packed_buffers:
                continue

            packed_cpu = self.packed_buffers[dtype]

            gpu_buffer = packed_cpu.to(GPU_DEVICE, non_blocking=True)

            # Record event and keep buffer alive
            event = torch.cuda.Event()
            event.record()
            self.gpu_blocks[dtype] = gpu_buffer
            self.gpu_events[dtype] = event

            # ATOMIC COMMIT: Rebind all params/buffers for this dtype
            for spec in specs:
                start_idx = spec['offset']
                end_idx = start_idx + spec['size']
                gpu_view = gpu_buffer[start_idx:end_idx].view(spec['shape'])

                if spec['is_param']:
                    spec['param_ref'].data = gpu_view
                else:
                    spec['buffer_ref'].data = gpu_view

    def unpack_to_cpu(self, layer):
        """Move GPU data back to packed CPU buffers and rebind layer parameters"""
        # Group specs by dtype
        dtype_specs = {}
        for spec in self.tensor_specs:
            dtype = spec['dtype']
            if dtype not in dtype_specs:
                dtype_specs[dtype] = []
            dtype_specs[dtype].append(spec)

        # Copy GPU → CPU for each dtype
        for dtype, specs in dtype_specs.items():
            if dtype not in self.gpu_blocks:
                continue

            gpu_buffer = self.gpu_blocks[dtype]
            packed_cpu = self.packed_buffers[dtype]

            # Synchronous copy back to packed CPU buffer
            packed_cpu.copy_(gpu_buffer.to('cpu'))
            # packed_cpu.copy_(gpu_buffer)

            # ATOMIC COMMIT: Rebind params/buffers back to CPU views
            for spec in specs:
                start_idx = spec['offset']
                end_idx = start_idx + spec['size']
                cpu_view = packed_cpu[start_idx:end_idx].view(spec['shape'])

                if spec['is_param']:
                    spec['param_ref'].data = cpu_view
                else:
                    spec['buffer_ref'].data = cpu_view

        # Release GPU memory
        self.gpu_blocks.clear()
        self.gpu_events.clear()

    def wait_for_gpu_transfer(self):
        """Wait for all async GPU transfers to complete"""
        for event in self.gpu_events.values():
            event.wait()

    def get_memory_usage(self):
        """Return memory usage info"""
        cpu_bytes = sum(buf.numel() * buf.element_size() for buf in self.packed_buffers.values())
        gpu_bytes = sum(buf.numel() * buf.element_size() for buf in self.gpu_blocks.values())
        return {
            'cpu_bytes': cpu_bytes,
            'gpu_bytes': gpu_bytes,
            'total_elements': self.total_elements
        }

def safe_move_to_gpu_packed(layer, idx):
    """Move layer to GPU using pre-packed buffer"""
    if PROFILE:
        nvtx.range_push(f"GPU_Transfer_Packed_L{idx}")
    try:
        current_device = device_cache.get_device(idx)
        if current_device.type == 'cuda':
            return True

        transfer_time = 0

        # if VERBOSE:
        start_time = time.time()

        if idx in packed_layers:
            # Use packed transfer (automatically uses pinned if enabled)
            packed_layers[idx].unpack_to_gpu(layer)
        else:
            # Fallback to normal transfer
            layer.to(GPU_DEVICE, non_blocking=True)

        event_based_sync("gpu_transfer", idx)
        # if VERBOSE:
        end_time = time.time()
        transfer_time = end_time - start_time
        # print(f"safe_move_to_gpu_packed: Layer {idx}: {transfer_time * 1000:.1f}ms transfer time recorded")
        if transfer_time > 0:
            layer_size_mb = get_cached_layer_size_mb(idx)
            speed_mbps = layer_size_mb / transfer_time
            # Track current step stats
            transfer_stats['current_step_gpu_times'].append(transfer_time)
            transfer_stats['current_step_gpu_speeds'].append(speed_mbps)

        device_cache.mark_moved(idx, torch.device('cuda'))
        swap_stats['to_gpu'] += 1
        return True

    except RuntimeError as e:
        if "out of memory" in str(e):
            print(f" OOM moving packed layer {idx}")
            return False
        raise e
    finally:
        if PROFILE:
            nvtx.range_pop()


def safe_move_to_cpu_packed(layer, idx):
    """Move layer to CPU - packed version already stored"""
    if PROFILE:
        nvtx.range_push(f"CPU_Transfer_Packed_L{idx}")
    try:
        try:
            current_device = next(layer.parameters()).device
            if current_device.type == 'cpu':
                return True
        except StopIteration:
            pass

        transfer_time = 0
        # if VERBOSE:
        start_time = time.time()

        layer.to(CPU_DEVICE)

        # if VERBOSE:
        end_time = time.time()
        transfer_time = end_time - start_time
        # print(f"safe_move_to_cpu_packed: Layer {idx}: {transfer_time * 1000:.1f}ms transfer time recorded")
        if transfer_time > 0:
            layer_size_mb = get_cached_layer_size_mb(idx)
            speed_mbps = layer_size_mb / transfer_time
            # Track current step stats
            transfer_stats['current_step_cpu_times'].append(transfer_time)
            transfer_stats['current_step_cpu_speeds'].append(speed_mbps)

        device_cache.mark_moved(idx, torch.device('cpu'))
        swap_stats['to_cpu'] += 1
        return True
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

    # THREADING: Initialize threading components (once)
    if not hasattr(add_smart_swapping_to_layer, 'threading_initialized'):
        add_smart_swapping_to_layer.threading_initialized = True
        add_smart_swapping_to_layer.gpu_lock = threading.Lock()
        add_smart_swapping_to_layer.current_layer_idx = 0
        add_smart_swapping_to_layer.training_active = True
        add_smart_swapping_to_layer.background_thread = None

    # CUDA STREAMS: Initialize streams (once) - only if enabled
    if args.cuda_streams and not hasattr(add_smart_swapping_to_layer, 'streams_initialized'):
        try:
            add_smart_swapping_to_layer.streams_initialized = True
            add_smart_swapping_to_layer.copy_stream = torch.cuda.Stream()
            add_smart_swapping_to_layer.compute_stream = torch.cuda.Stream()
            # Synchronize streams immediately after creation
            # torch.cuda.synchronize()
            # print(" CUDA Streams enabled for copy-compute overlap")
        except Exception as e:
            print(f" CUDA Streams failed to initialize: {e}")
            args.cuda_streams = False
            print(" Falling back to default stream")

    # THREADING: Start background thread (once)
    if args.threading and add_smart_swapping_to_layer.background_thread is None:
        add_smart_swapping_to_layer.background_thread = threading.Thread(
            target=background_layer_manager,
            daemon=True
        )
        add_smart_swapping_to_layer.background_thread.start()

    @wraps(original_forward)
    def swapped_forward(*fwd_args, **kwargs):

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

        if not hasattr(add_smart_swapping_to_layer, 'step_timing_initialized'):
            add_smart_swapping_to_layer.step_timing_initialized = True
            add_smart_swapping_to_layer.step_start_time = time.time()
            add_smart_swapping_to_layer.current_step = 0

        # Detect new step when we hit layer 0 again (reset to early layers)
        if layer_idx == 0:  # Early layer indicates new step
            add_smart_swapping_to_layer.current_step += 1
            add_smart_swapping_to_layer.calls_this_step = 0
            if VERBOSE:
                print(f" New sampling step {add_smart_swapping_to_layer.current_step}")

        if not hasattr(add_smart_swapping_to_layer, 'total_forward_calls'):
            add_smart_swapping_to_layer.total_forward_calls = 0
            add_smart_swapping_to_layer.current_step = 0
            add_smart_swapping_to_layer.calls_this_step = 0

        add_smart_swapping_to_layer.total_forward_calls += 1
        add_smart_swapping_to_layer.calls_this_step += 1

        ##DEBUG STATS MEM OP
        if VERBOSE and add_smart_swapping_to_layer.total_forward_calls % 20 == 0:
            global current_model
            if 'current_model' in globals():
                print_memory_optimization_analysis(current_model, layers, args)

        # no need for it just tell you its working.
        # if VERBOSE and layer_idx % 10 == 0:  # Every 10th layer
        #     current_step = getattr(add_smart_swapping_to_layer, 'current_step', 0)
        #     print(f"Layer {layer_idx} step {current_step}: device={device_cache.get_device(layer_idx)}")

        # Step progress logging.
        # if VERBOSE and layer_idx % 20 == 0:
        #     current_step = getattr(add_smart_swapping_to_layer, 'current_step', 0)
        #     if (add_smart_swapping_to_layer.prefetch_hits + add_smart_swapping_to_layer.prefetch_misses) > 0:
        #         hit_rate = add_smart_swapping_to_layer.prefetch_hits / (
        #                 add_smart_swapping_to_layer.prefetch_hits + add_smart_swapping_to_layer.prefetch_misses) * 100
        #         print(
        #             f" Prefetch hit rate: {hit_rate:.1f}% ({add_smart_swapping_to_layer.prefetch_hits}/{add_smart_swapping_to_layer.prefetch_hits + add_smart_swapping_to_layer.prefetch_misses})")


        # THREADING: Update current layer for background thread
        if args.threading:
            add_smart_swapping_to_layer.current_layer_idx = layer_idx

        if PROFILE:
            nvtx.range_push(f"Layer_{layer_idx}_{'Forward'}")
        try:

            if not add_smart_swapping_to_layer.current_forward_logged:
                current_step = getattr(add_smart_swapping_to_layer, 'current_step', 0)
                if VERBOSE:
                    print(f" Forward pass step {current_step}: layers {min(cpu_swappable_layers)}-{max(cpu_swappable_layers)}")
                add_smart_swapping_to_layer.current_forward_logged = True

            if not hasattr(add_smart_swapping_to_layer, 'stats_initialized'):
                add_smart_swapping_to_layer.stats_initialized = True
                add_smart_swapping_to_layer.prefetch_hits = 0
                add_smart_swapping_to_layer.prefetch_misses = 0


            # 2. SMART LAYER MANAGEMENT
            if layer_idx in cpu_swappable_layers:
                if PROFILE:
                    nvtx.range_push(f"Smart_Management_L{layer_idx}")
                try:
                    current_device = device_cache.get_device(layer_idx)
                    layer_already_on_gpu = (current_device.type == 'cuda')

                    if not layer_already_on_gpu:
                        # THREADING: If threading enabled, wait briefly for background thread
                        if args.threading:
                            # print(f" Layer {layer_idx} not ready, waiting for background thread...")
                            for _ in range(50):
                                time.sleep(0.000001) #do not touch. very needed for sync
                                if device_cache.get_device(layer_idx).type == 'cuda':
                                    layer_already_on_gpu = True
                                    print(f" Background thread caught up!")
                                    break

                        # If still not ready, run fallback
                        if not layer_already_on_gpu:

                            #Use lock to prevent race condition
                            if args.threading:
                                with add_smart_swapping_to_layer.gpu_lock:
                                    needed_layers = calculate_needed_layers(layer_idx, args.prefetch)

                                    if args.selective_packing:
                                        cleaned = cleanup_excess_layers_packed(needed_layers, args.packing_threshold_mb)
                                        fetched = fetch_missing_layers_packed(needed_layers, args.packing_threshold_mb)
                                    else:
                                        cleaned = cleanup_excess_layers(needed_layers)
                                        fetched = fetch_missing_layers(needed_layers)
                            else:
                                needed_layers = calculate_needed_layers(layer_idx, args.prefetch)
                                if args.selective_packing:
                                    cleaned = cleanup_excess_layers_packed(needed_layers, args.packing_threshold_mb)
                                    fetched = fetch_missing_layers_packed(needed_layers, args.packing_threshold_mb)
                                else:
                                    cleaned = cleanup_excess_layers(needed_layers)
                                    fetched = fetch_missing_layers(needed_layers)

                                device_cache.mark_moved(layer_idx, device_cache.get_layer_device(layers[layer_idx]))
                                final_device = device_cache.get_device(layer_idx)

                    if layer_already_on_gpu:
                        add_smart_swapping_to_layer.prefetch_hits += 1
                        # if VERBOSE:
                        #     print(f" ✅ Layer {layer_idx} prefetch hit")
                    else:
                        add_smart_swapping_to_layer.prefetch_misses += 1
                        # if VERBOSE:
                        #     print(f" ⚠️ Layer {layer_idx} prefetch miss, fetching...")

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

            # Wait for any pending GPU transfers before executing the layer
            if hasattr(layer, '_packed_block'):
                for event in layer._packed_block.gpu_events.values():
                    torch.cuda.current_stream().wait_event(event)

            # 4. COMPUTATION
            if PROFILE:
                nvtx.range_push(f"Compute_L{layer_idx}")
            try:

                if args.cuda_streams and hasattr(add_smart_swapping_to_layer, 'compute_stream'):
                    with torch.cuda.stream(add_smart_swapping_to_layer.compute_stream):
                        # if VERBOSE:
                        layer_compute_start = time.time()

                        if args.compute_casting != "disabled":
                            # Compute casting: use specified dtype
                            dtype_map = {'fp32': torch.float32, 'bf16': torch.bfloat16, 'fp16': torch.float16}
                            cast_dtype = dtype_map[args.compute_casting]

                            with torch.autocast(device_type='cuda', dtype=cast_dtype, enabled=True):
                                fix_inference_tensor_parameters(layer)
                                # result = original_forward(x, *tuple(new_args), **new_kwargs)
                                # Call with preserved argument structure
                                if is_flux_call:
                                    if fwd_args:
                                        result = original_forward(*new_args, **new_kwargs)
                                        return result
                                    else:
                                        return original_forward(**new_kwargs)
                                else:
                                    # WAN/SDXL: original logic
                                    return original_forward(x, *new_args, **new_kwargs)

                        elif args.autocast and args.autocast != 'fp32':
                            if args.autocast in ['f8_e4m3', 'f8_e5m2']:
                                # Use Transformer Engine for FP8
                                import transformer_engine.pytorch as te
                                with te.fp8_autocast():
                                    fix_inference_tensor_parameters(layer)
                                    result = original_forward(x, *tuple(new_args), **new_kwargs)
                            else:
                                # Regular PyTorch autocast for FP16/BF16
                                dtype_map = {'fp16': torch.float16, 'bf16': torch.bfloat16}
                                autocast_dtype = dtype_map[args.autocast]
                                with torch.autocast(device_type='cuda', dtype=autocast_dtype):
                                    fix_inference_tensor_parameters(layer)
                                    result = original_forward(x, *tuple(new_args), **new_kwargs)
                        else:
                            fix_inference_tensor_parameters(layer)
                            # result = original_forward(x, *tuple(new_args), **new_kwargs)
                            # Call with preserved argument structure
                            if is_flux_call:
                                if fwd_args:
                                    result = original_forward(*new_args, **new_kwargs)
                                    return result
                                else:

                                    return original_forward(**new_kwargs)
                            else:
                                return original_forward(x, *new_args, **new_kwargs)

                        # if VERBOSE:
                        layer_compute_end = time.time()
                        layer_compute_time = layer_compute_end - layer_compute_start
                        add_smart_swapping_to_layer.layer_compute_times.append(layer_compute_time)
                        return result
                else:
                    # if VERBOSE:
                    layer_compute_start = time.time()

                    if args.compute_casting != "disabled":
                        # Compute casting: use specified dtype
                        dtype_map = {'fp32': torch.float32, 'bf16': torch.bfloat16, 'fp16': torch.float16}
                        cast_dtype = dtype_map[args.compute_casting]

                        with torch.autocast(device_type='cuda', dtype=cast_dtype, enabled=True):
                            fix_inference_tensor_parameters(layer)
                            # result = original_forward(x, *tuple(new_args), **new_kwargs)
                            # Call with preserved argument structure
                            if is_flux_call:
                                if fwd_args:
                                    result = original_forward(*new_args, **new_kwargs)
                                    return result
                                else:

                                    return original_forward(**new_kwargs)
                            else:
                                # WAN/SDXL: original logic
                                return original_forward(x, *new_args, **new_kwargs)

                    elif args.autocast and args.autocast != 'fp32':
                        if args.autocast in ['f8_e4m3', 'f8_e5m2']:
                            # Use Transformer Engine for FP8
                            import transformer_engine.pytorch as te
                            with te.fp8_autocast():
                                fix_inference_tensor_parameters(layer)
                                result = original_forward(x, *tuple(new_args), **new_kwargs)
                        else:
                            # Regular PyTorch autocast for FP16/BF16
                            dtype_map = {'fp16': torch.float16, 'bf16': torch.bfloat16}
                            autocast_dtype = dtype_map[args.autocast]
                            with torch.autocast(device_type='cuda', dtype=autocast_dtype):
                                fix_inference_tensor_parameters(layer)
                                result = original_forward(x, *tuple(new_args), **new_kwargs)
                    else:
                        fix_inference_tensor_parameters(layer)
                        # result = original_forward(x, *tuple(new_args), **new_kwargs)
                        # Call with preserved argument structure
                        if is_flux_call:
                            if fwd_args:
                                result = original_forward(*new_args, **new_kwargs)
                                return result
                            else:
                                return original_forward(**new_kwargs)
                        else:
                            # WAN/SDXL: original logic
                            return original_forward(x, *new_args, **new_kwargs)

                # if VERBOSE:
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
                "initial_gpu_layers": ("INT",
                                       {"default": 2, "min": 0, "max": 100, "tooltip": TOOLTIPS["initial_gpu_layers"]}),
                "final_gpu_layers": ("INT",
                                     {"default": 2, "min": 0, "max": 100, "tooltip": TOOLTIPS["final_gpu_layers"]}),
                "prefetch": ("INT", {"default": 1, "min": 0, "max": 100, "tooltip": TOOLTIPS["prefetch"]}),
                "threading": ("BOOLEAN", {"default": False, "tooltip": TOOLTIPS["threading"]}),
                "event_sync": ("BOOLEAN", {"default": False, "tooltip": TOOLTIPS["event_sync"]}),
                "cuda_streams": ("BOOLEAN", {"default": False, "tooltip": TOOLTIPS["cuda_streams"]}),
                # "batch_move": ("BOOLEAN", {"default": False, "tooltip": TOOLTIPS["batch_move"]}),
                "mega_tensor": ("BOOLEAN", {"default": False, "tooltip": "Pack multiple layers into mega tensor for batch transfer"}),
                "mega_tensor_size": ("INT", {"default": 3, "min": 2, "max": 50,
                                             "tooltip": "Number of layers to pack into each mega tensor. Should be the same or equal to Prefetch"}),
                "selective_packing": ("BOOLEAN", {"default": False, "tooltip": TOOLTIPS["selective_packing"]}),
                "packing_threshold_mb": ("INT", {"default": 64, "min": 0, "max": 2048, "tooltip": "Size threshold in MB for packed transfers when selective packing is enabled"}),
                "verbose": ("BOOLEAN", {"default": True, "tooltip": TOOLTIPS["verbose"]}),
                "compile": ("BOOLEAN", {"default": False, "tooltip": TOOLTIPS["compile"]}),
            },
            "optional": {
                "gpu_layer_indices": ("STRING", {"default": "", "tooltip": TOOLTIPS["gpu_layers"]}),
                #Precision tools have been disabled due to confliction with comfy auto precision and casting. need active testing but they are non essential

                # "compute_casting": (["disabled", "fp32", "bf16", "fp16"],
                #                     {"default": "disabled",
                #                      "tooltip": "Cast to higher precision for computation only. Model stays in original dtype for storage/transfer, temporarily upcasts during forward pass for better accuracy"}),
                "cast_target": ("STRING", {"default": "", "tooltip": TOOLTIPS["cast_target"]}),
                # "autocast": (["", 'auto', "fp16", "bf16", "f8_e4m3", "f8_e5m2"],
                #              {"default": "",}),
                # "mixed_precision": (["auto", "f32", "bf16", "f16"],
                #                     {"default": "auto", "tooltip": TOOLTIPS["mixed_precision"]}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)

    FUNCTION = "load_model_with_swapping"
    CATEGORY = "loaders"
    TITLE = "Dynamic Swapping Loader"


    def load_model_with_swapping(self, model, layers, initial_gpu_layers, final_gpu_layers,
                                 prefetch, threading, event_sync, cuda_streams, mega_tensor, mega_tensor_size,
                                 selective_packing,packing_threshold_mb, verbose, compile, compute_casting=False, gpu_layer_indices="",
                                 cast_target="", autocast="", mixed_precision="auto"):

        # Force cleanup before starting
        # torch.cuda.empty_cache()
        # gc.collect()
        # torch.cuda.synchronize()

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
        args.threading = threading
        args.cuda_streams = cuda_streams
        # args.batch_move = batch_move
        args.batch_move = mega_tensor #keeps existing code in line. name changed to not be confused with batch size
        args.mega_tensor_size = mega_tensor_size
        args.selective_packing = selective_packing
        args.packing_threshold_mb = packing_threshold_mb
        args.event_sync = event_sync
        args.verbose = verbose
        global VERBOSE
        VERBOSE = args.verbose
        args.compile = compile

        args.device_sync_mode = 'off'
        args.sync_only_on_resume = False
        args.mixed_precision = mixed_precision

        # Handle string parameters
        args.gpu_layers = gpu_layer_indices.strip() if gpu_layer_indices.strip() else None
        args.cast_target = cast_target.strip() if cast_target.strip() else None
        args.autocast = False #autocast if autocast else None

        args.compute_casting = "disabled"#compute_casting

        if compute_casting:
            args.autocast = False


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
                print(f"⚠️  Invalid cast_target format: '{cast_target}'. Expected: 'from_dtype to_dtype'")

        if mega_tensor:
            casting_handler = CastingHandler()
            casting_handler.convert_for_mega_tensor_compatibility(layers)

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

            if args.selective_packing:
                print(" Packing CPU-swappable layers for optimized transfers...")
                for i in cpu_swappable_layers:
                    try:
                        # Pass pinned flag to PackedCPUBlock
                        packed_layers[i] = PackedCPUBlock(layers[i])
                        # print(f"✓ Layer {i} packed: {packed_layers[i].total_elements} elements")
                    except Exception as e:
                        print(f" Failed to pack layer {i}: {e}")
            else:
                print(" Selective packing disabled - using direct transfers only")

            print("Calculating layer sizes...")
            global layer_sizes_mb
            for i, layer in enumerate(layers):
                layer_sizes_mb[i] = sum(p.numel() * p.element_size() for p in layer.parameters()) / (1024 * 1024)
                if VERBOSE:
                    print(f"   Layer {i}: {layer_sizes_mb[i]:.1f}MB")

            if cast_target or mega_tensor:
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

            if args.compile:
                # Try torch.compile version
                @torch.compile
                def compiled_forward( *layer_args, **layer_kwargs):
                    return original_forward( *layer_args, **layer_kwargs)

                def create_resident_forward(layer_idx, original_forward, compiled_fn):
                    current_args = args
                    @wraps(original_forward)
                    def resident_forward( *args_tuple, **kwargs):

                        # if VERBOSE:
                        layer_compute_start = time.time()

                        try:

                            if current_args.compute_casting != "disabled":
                                dtype_map = {'fp32': torch.float32, 'bf16': torch.bfloat16, 'fp16': torch.float16}
                                cast_dtype = dtype_map[args.compute_casting]

                                with torch.autocast(device_type='cuda', dtype=cast_dtype, enabled=True):
                                    result = original_forward( *args_tuple, **kwargs)
                            else:
                                result = compiled_fn( *args_tuple, **kwargs)
                        except Exception as e:
                            print(f" Compile failed for layer {layer_idx}: {e}")
                            result = original_forward( *args_tuple, **kwargs)

                        # if VERBOSE:
                        layer_compute_end = time.time()
                        layer_compute_time = layer_compute_end - layer_compute_start
                        add_smart_swapping_to_layer.layer_compute_times.append(layer_compute_time)
                        return result

                    return resident_forward

                layer.forward = create_resident_forward(layer_idx, original_forward, compiled_forward)

            else:
                # Original working version (no compile)
                def create_resident_forward(layer_idx, original_forward):
                    current_args = args
                    @wraps(original_forward)
                    def resident_forward( *args_tuple, **kwargs):

                        # if VERBOSE:
                        layer_compute_start = time.time()


                        if current_args.compute_casting != "disabled":
                            dtype_map = {'fp32': torch.float32, 'bf16': torch.bfloat16, 'fp16': torch.float16}
                            cast_dtype = dtype_map[args.compute_casting]

                            with torch.autocast(device_type='cuda', dtype=cast_dtype, enabled=True):
                                result = original_forward( *args_tuple, **kwargs)
                        else:
                            result = original_forward( *args_tuple, **kwargs)

                        # result = original_forward( *args_tuple, **kwargs)

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

"""
Universal Mixed Precision Handler by obisin
=================================

Drop-in utility to handle gradient operations with mixed precision (float8, bfloat16, float32)
across any model architecture. Patches PyTorch's gradient functions to handle unsupported
dtype operations gracefully.

Usage:
    from utils.mixed_precision_handler import patch_mixed_precision
    patch_mixed_precision()

    # or with user control:
    patch_mixed_precision(mixed_precision_target='f32')
"""
class CastingHandler:
    def __init__(self):
        self.original_clip_grad_norm = None
        self.dtype_conversion_map = {
            torch.float8_e4m3fn: torch.bfloat16,
            torch.float8_e5m2: torch.bfloat16,
        }
        self.safe_dtypes = {torch.float32, torch.float16, torch.bfloat16}
        self.is_patched = False

    def detect_model_highest_precision(self, layers):
        """Find the single highest precision dtype in the model"""
        all_dtypes = set()
        for layer in layers:
            for param in layer.parameters():
                all_dtypes.add(param.dtype)

        # Precision hierarchy (highest to lowest)
        precision_order = [torch.float64, torch.float32, torch.bfloat16, torch.float16,
                           getattr(torch, 'float8_e4m3fn', None)]

        # Find the highest precision actually used
        for dtype in precision_order:
            if dtype and dtype in all_dtypes:
                return dtype
        return torch.float32  # fallback


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

    def replace_with_4bit_layer(self, layer):
        """Replace torch.nn.Linear with bitsandbytes Linear4bit, skip LoRA layers"""
        try:
            # import bitsandbytes as bnb

            for name, module in layer.named_modules():
                if isinstance(module, torch.nn.Linear):
                    # Skip LoRA layers - they cause dropout issues with 4-bit
                    if any(lora_term in name.lower() for lora_term in ['lora', 'adapter']):
                        # print(f"   Skipping LoRA layer: {name}")
                        continue

                    parent_names = name.split('.')
                    if any(any(lora_term in part.lower() for lora_term in ['lora', 'adapter'])
                           for part in parent_names):
                        continue

                    print(f"   Converting layer: {name}")
                    linear_4bit = bnb.nn.Linear4bit(
                        module.in_features,
                        module.out_features,
                        bias=module.bias is not None,
                        compute_dtype=torch.float16,
                        quant_type='nf4'
                    )

                    linear_4bit.load_state_dict(module.state_dict())
                    linear_4bit = linear_4bit.to('cuda')

                    # Replace in parent module
                    parent_name = '.'.join(name.split('.')[:-1])
                    module_name = name.split('.')[-1]
                    parent = layer
                    for part in parent_name.split('.'):
                        if part:
                            parent = getattr(parent, part)
                    setattr(parent, module_name, linear_4bit)

            return layer
        except Exception as e:
            print(f"️ 4-bit replacement failed: {e}")
            return layer


    def precast_all_layers(self, cpu_swappable_layers, layers, cast_target=None):
        """Cast all layer dtypes once at startup using existing cast_to_dtype"""
        if not cast_target:
            print("⚠️  No cast_target specified, skipping precasting")
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

    def convert_for_mega_tensor_compatibility(self, layers):
        """Convert F8 dtypes to F16 for mega tensor compatibility"""
        print(" Converting F8 layers to F16 for mega tensor compatibility...")
        converted_layers = 0

        # Check if any F8 dtypes exist
        has_f8 = False
        for layer in layers:
            for param in layer.parameters():
                if param.dtype in [torch.float8_e4m3fn, torch.float8_e5m2]:
                    has_f8 = True
                    break
            if has_f8:
                break

        if not has_f8:
            print("✓ No F8 dtypes found - mega tensor compatible")
            return 0

        # Convert only F8 layers to F16
        for i, layer in enumerate(layers):
            layer_converted = False

            for param in layer.parameters():
                if param.dtype in [torch.float8_e4m3fn, torch.float8_e5m2]:
                    param.data = param.data.to(torch.float16)
                    layer_converted = True

            for buffer in layer.buffers():
                if buffer.dtype in [torch.float8_e4m3fn, torch.float8_e5m2]:
                    buffer.data = buffer.data.to(torch.float16)
                    layer_converted = True

            if layer_converted:
                converted_layers += 1

        print(f"✓ Converted {converted_layers} F8 layers to F16 for mega tensor compatibility")
        return converted_layers
