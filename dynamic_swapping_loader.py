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
        self.selective_packing = 0   # Size threshold in MB for using packed transfers (default: 64MB)
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
    "threading": "Enable background threading for automatic layer management",
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

swap_stats = {'to_gpu': 0, 'to_cpu': 0}
transfer_events = {}
transfer_stats = {
    'to_gpu_times': [], 'to_cpu_times': [], 'to_gpu_speeds': [], 'to_cpu_speeds': [],
    'current_step_gpu_times': [], 'current_step_cpu_times': [],
    'current_step_gpu_speeds': [], 'current_step_cpu_speeds': []
}
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
    return layer_sizes_mb.get(idx, 0)

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

        transfer_time = 0

        if VERBOSE:
            start_time = time.time()
            layer_size_mb = get_cached_layer_size_mb(idx)
        layer.to(CPU_DEVICE)
        add_smart_swapping_to_layer.cleanup_stream = torch.cuda.Stream()
        swap_stats['to_cpu'] += 1
        if VERBOSE:
            end_time = time.time()
            transfer_time = end_time - start_time

        if transfer_time > 0 and VERBOSE:
            speed_mbps = layer_size_mb / transfer_time

            # Track current step stats
            transfer_stats['current_step_cpu_times'].append(transfer_time)
            transfer_stats['current_step_cpu_speeds'].append(speed_mbps)

        device_cache.mark_moved(idx, torch.device('cpu'))

        # if idx % 10 == 0:
        #     print(f"   Layer {idx} → CPU, memory: {torch.cuda.memory_allocated(0) / 1e9:.1f}GB")

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

            if VERBOSE:
                start_time = time.time()
                layer_size_mb = get_cached_layer_size_mb(idx)

            layer.to(GPU_DEVICE, non_blocking=True)

            if VERBOSE:
                end_time = time.time()
                transfer_time = end_time - start_time

            if transfer_time > 0 and VERBOSE:
                speed_mbps = layer_size_mb / transfer_time
                # Track current step stats
                transfer_stats['current_step_gpu_times'].append(transfer_time)
                transfer_stats['current_step_gpu_speeds'].append(speed_mbps)

            event_based_sync("gpu_transfer", idx)
            device_cache.mark_moved(idx, torch.device('cuda'))
            swap_stats['to_gpu'] += 1
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


def cleanup_excess_layers(keep_layers):
    """Remove layers from GPU that are not in keep_layers set"""
    # current_step = getattr(add_smart_swapping_to_layer, 'current_sampling_step', 0)

    if PROFILE:
        nvtx.range_push(f"Cleanup_Excess_{len(cpu_swappable_layers - keep_layers)}layers")
    try:
        if args.batch_move:
            # Batch approach (your current code)
            layers_to_remove = []
            for idx in cpu_swappable_layers:
                if (idx < len(layers) and
                        idx not in keep_layers and
                        device_cache.get_device(idx).type == 'cuda'):
                    layers_to_remove.append(idx)


            cleaned_count = batch_safe_move_to_cpu(layers_to_remove)
        else:
            # Individual approach (fallback)
            cleaned_count = 0
            layers_to_remove = []
            for idx in cpu_swappable_layers:
                if (idx < len(layers) and
                        idx not in keep_layers and
                        device_cache.get_device(idx).type == 'cuda'):
                    layers_to_remove.append(idx)
                    safe_move_to_cpu(layers[idx], idx)
                    cleaned_count += 1

        return cleaned_count
    finally:
        if PROFILE:
            nvtx.range_pop()


def fetch_missing_layers(needed_layers):
    """Ensure all needed layers are on GPU"""
    current_step = getattr(add_smart_swapping_to_layer, 'current_sampling_step', 0)

    layers_to_fetch = [idx for idx in needed_layers
                       if (idx < len(layers) and
                           idx in cpu_swappable_layers and
                           device_cache.get_device(idx).type == 'cpu')]


    if PROFILE:
        nvtx.range_push(f"Fetch_Missing_{len(needed_layers)}layers")
    try:
        # Your existing fetch logic...
        fetched_count = 0
        if args.batch_move:
            fetched_count = batch_safe_move_to_gpu(layers_to_fetch)
        else:
            for idx in layers_to_fetch:
                success = safe_move_to_gpu(layers[idx], idx)
                if success:
                    fetched_count += 1


        return fetched_count
    finally:
        if PROFILE:
            nvtx.range_pop()

def batch_safe_move_to_gpu_packed(layer_indices, threshold_mb=64):
    """Batch move to GPU with selective packing"""
    if not layer_indices:
        return 0

    moved_count = 0
    large_layers = []
    small_layers = []

    # Separate layers by size
    for idx in layer_indices:
        if (idx < len(layers) and
                device_cache.get_device(idx).type == 'cpu'):

            layer_size_mb = get_cached_layer_size_mb(idx)

            if layer_size_mb > threshold_mb and idx in packed_layers:
                large_layers.append(idx)
            else:
                small_layers.append(idx)

    # Handle large layers with packed transfer
    if large_layers:
        for idx in large_layers:
            success = safe_move_to_gpu_packed(layers[idx], idx)
            if success:
                moved_count += 1

    # Handle small layers with direct transfer
    if small_layers:
        for idx in small_layers:
            success = safe_move_to_gpu(layers[idx], idx)
            if success:
                moved_count += 1

    return moved_count


def batch_safe_move_to_cpu_packed(layer_indices, threshold_mb=64):
    """Batch move to CPU with selective packing"""
    if not layer_indices:
        return 0

    moved_count = 0
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
                # print(f"🔄 Background thread detected step {current_step}, preloading early layers")
                last_seen_step = current_step

                # Preload for new step using actual prefetch setting
                first_swappable = min(cpu_swappable_layers)
                next_step_needed = calculate_needed_layers(first_swappable, args.prefetch)

                for layer_idx in next_step_needed:
                    if device_cache.get_device(layer_idx).type == 'cpu':
                        safe_move_to_gpu_threaded(layers[layer_idx], layer_idx)

                #OVERLY AGGRESSIVE DSEBUG VERSION
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
        for idx in layer_indices:
            if (idx < len(layers) and
                    device_cache.get_device(idx).type == 'cuda'):

                # Measure transfer timing
                transfer_time = 0
                if VERBOSE:
                    start_time = time.time()
                    layer_size_mb = get_cached_layer_size_mb(idx)
                layers[idx].to(CPU_DEVICE)
                if VERBOSE:
                    end_time = time.time()
                    transfer_time = end_time - start_time

                if transfer_time > 0 and VERBOSE:
                    speed_mbps = layer_size_mb / transfer_time

                    # Track current step stats
                    transfer_stats['current_step_cpu_times'].append(transfer_time)
                    transfer_stats['current_step_cpu_speeds'].append(speed_mbps)
                device_cache.mark_moved(idx, torch.device('cpu'))
                swap_stats['to_cpu'] += 1
                moved_count += 1

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
        for idx in layer_indices:
            if (idx < len(layers) and
                    device_cache.get_device(idx).type == 'cpu'):

                layer = layers[idx]
                transfer_time = 0

                if VERBOSE:
                    start_time = time.time()
                    layer_size_mb = get_cached_layer_size_mb(idx)

                layer.to(GPU_DEVICE, non_blocking=True)


                if VERBOSE:
                    end_time = time.time()
                    transfer_time = end_time - start_time

                event_based_sync("gpu_transfer", idx)
                if transfer_time > 0 and VERBOSE:
                    speed_mbps = layer_size_mb / transfer_time
                    # Track current step stats
                    transfer_stats['current_step_gpu_times'].append(transfer_time)
                    transfer_stats['current_step_gpu_speeds'].append(speed_mbps)

                device_cache.mark_moved(idx, torch.device('cuda'))
                swap_stats['to_gpu'] += 1
                moved_count += 1

        return moved_count
    finally:
        if PROFILE:
            nvtx.range_pop()

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

        if VERBOSE:
            start_time = time.time()
            layer_size_mb = get_cached_layer_size_mb(idx)

        if idx in packed_layers:
            # Use packed transfer (automatically uses pinned if enabled)
            packed_layers[idx].unpack_to_gpu(layer, GPU_DEVICE)
        else:
            # Fallback to normal transfer
            layer.to(GPU_DEVICE, non_blocking=True)

        transfer_time = 0

        event_based_sync("gpu_transfer", idx)
        if VERBOSE:
            end_time = time.time()
            transfer_time = end_time - start_time
        if transfer_time > 0 and VERBOSE:
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

        if VERBOSE:
            start_time = time.time()
            layer_size_mb = get_cached_layer_size_mb(idx)
        layer.to(CPU_DEVICE)
        if VERBOSE:
            end_time = time.time()
            transfer_time = end_time - start_time

        if transfer_time > 0 and VERBOSE:
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
        add_smart_swapping_to_layer.streams_initialized = True
        add_smart_swapping_to_layer.copy_stream = torch.cuda.Stream()
        add_smart_swapping_to_layer.compute_stream = torch.cuda.Stream()
        print(" CUDA Streams enabled for copy-compute overlap")

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

        if VERBOSE and not hasattr(add_smart_swapping_to_layer, 'step_timing_initialized'):
            add_smart_swapping_to_layer.step_timing_initialized = True
            add_smart_swapping_to_layer.step_start_time = time.time()
            add_smart_swapping_to_layer.current_step = 0

        if not hasattr(add_smart_swapping_to_layer, 'total_forward_calls'):
            add_smart_swapping_to_layer.total_forward_calls = 0
            add_smart_swapping_to_layer.current_step = 0
            add_smart_swapping_to_layer.calls_this_step = 0

        add_smart_swapping_to_layer.total_forward_calls += 1
        add_smart_swapping_to_layer.calls_this_step += 1

        # Detect new step when we hit layer 0 again (reset to early layers)
        if layer_idx <= 2:  # Early layer indicates new step
            if add_smart_swapping_to_layer.calls_this_step > len(layers):  # Processed full model
                add_smart_swapping_to_layer.current_step += 1
                add_smart_swapping_to_layer.calls_this_step = 0

                if VERBOSE:
                    print(f"🔄 New sampling step {add_smart_swapping_to_layer.current_step} detected")

        # At the start of swapped_forward
        if VERBOSE and layer_idx % 10 == 0:  # Every 10th layer
            current_step = getattr(add_smart_swapping_to_layer, 'current_step', 0)
            print(f"Layer {layer_idx} step {current_step}: device={device_cache.get_device(layer_idx)}")

        # Step progress logging
        if VERBOSE and layer_idx % 20 == 0:
            current_step = getattr(add_smart_swapping_to_layer, 'current_step', 0)
            if (add_smart_swapping_to_layer.prefetch_hits + add_smart_swapping_to_layer.prefetch_misses) > 0:
                hit_rate = add_smart_swapping_to_layer.prefetch_hits / (
                        add_smart_swapping_to_layer.prefetch_hits + add_smart_swapping_to_layer.prefetch_misses) * 100
                print(
                    f" Prefetch hit rate: {hit_rate:.1f}% ({add_smart_swapping_to_layer.prefetch_hits}/{add_smart_swapping_to_layer.prefetch_hits + add_smart_swapping_to_layer.prefetch_misses})")


        # THREADING: Update current layer for background thread
        if args.threading:
            add_smart_swapping_to_layer.current_layer_idx = layer_idx

        if PROFILE:
            nvtx.range_push(f"Layer_{layer_idx}_{'Forward'}")
        try:

            # Initialize global flags if they don't exist
            # if not hasattr(add_smart_swapping_to_layer, 'current_forward_logged'):
            #     add_smart_swapping_to_layer.current_forward_logged = False

            # # Summary logging at the start - only once per pass type
            if not add_smart_swapping_to_layer.current_forward_logged:
                current_step = getattr(add_smart_swapping_to_layer, 'current_step', 0)
                if VERBOSE:
                    print(
                        f" Forward pass step {current_step}: layers {min(cpu_swappable_layers)}-{max(cpu_swappable_layers)}")
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
                            print(f" Layer {layer_idx} not ready, waiting for background thread...")
                            for _ in range(50):
                                time.sleep(0.000001) #do not touch. very needed for sync
                                if device_cache.get_device(layer_idx).type == 'cuda':
                                    layer_already_on_gpu = True
                                    print(f" Background thread caught up!")
                                    break

                        # If still not ready, run fallback
                        if not layer_already_on_gpu:

                            # THREADING: Use lock to prevent race condition
                            if args.threading:
                                with add_smart_swapping_to_layer.gpu_lock:
                                    needed_layers = calculate_needed_layers(layer_idx, args.prefetch)

                                    if args.selective_packing:
                                        cleaned = cleanup_excess_layers_packed(needed_layers, args.selective_packing)
                                        fetched = fetch_missing_layers_packed(needed_layers, args.selective_packing)
                                    else:
                                        cleaned = cleanup_excess_layers(needed_layers)
                                        fetched = fetch_missing_layers(needed_layers)
                            else:
                                # No threading, run normally
                                needed_layers = calculate_needed_layers(layer_idx, args.prefetch)
                                if args.selective_packing:
                                    cleaned = cleanup_excess_layers_packed(needed_layers, args.selective_packing)
                                    fetched = fetch_missing_layers_packed(needed_layers, args.selective_packing)
                                else:
                                    cleaned = cleanup_excess_layers(needed_layers)
                                    fetched = fetch_missing_layers(needed_layers)

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

                if layer_already_on_gpu:
                    add_smart_swapping_to_layer.prefetch_hits += 1
                else:
                    add_smart_swapping_to_layer.prefetch_misses += 1

                device = device_cache.get_device(layer_idx)
                gpu_success = device.type == 'cuda'
            else:
                # Layer not in cpu_swappable_layers (permanent resident)
                device = device_cache.get_device(layer_idx)
                gpu_success = device.type == 'cuda'

            # Handle GPU failure case
            if not gpu_success:
                print(f" Layer {layer_idx} failed to be on GPU, forcing aggressive cleanup...")
                # Nuclear cleanup - evict ALL layers except this one
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

            # # Debug output every 10th layer
            # if layer_idx % 10 == 0:  # Every 10th layer
            #     current_step = getattr(add_smart_swapping_to_layer, 'current_step', 0)
            #     print(f"Layer {layer_idx} step {current_step}: device={device_cache.get_device(layer_idx)}")


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


            # DISABLE FLUX DETECTION FOR TESTING
            # is_flux_call = False  # Force to False

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
                        if VERBOSE:
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

                        if VERBOSE:
                            layer_compute_end = time.time()
                            layer_compute_time = layer_compute_end - layer_compute_start
                            add_smart_swapping_to_layer.layer_compute_times.append(layer_compute_time)
                        return result
                else:
                    if VERBOSE:
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

                if VERBOSE:
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
                                       {"default": 2, "min": 0, "max": 50, "tooltip": TOOLTIPS["initial_gpu_layers"]}),
                "final_gpu_layers": ("INT",
                                     {"default": 2, "min": 0, "max": 50, "tooltip": TOOLTIPS["final_gpu_layers"]}),
                "prefetch": ("INT", {"default": 1, "min": 0, "max": 100, "tooltip": TOOLTIPS["prefetch"]}),
                "threading": ("BOOLEAN", {"default": False, "tooltip": TOOLTIPS["threading"]}),
                "event_sync": ("BOOLEAN", {"default": False, "tooltip": TOOLTIPS["event_sync"]}),
                "cuda_streams": ("BOOLEAN", {"default": False, "tooltip": TOOLTIPS["cuda_streams"]}),
                "batch_move": ("BOOLEAN", {"default": False, "tooltip": TOOLTIPS["batch_move"]}),
                "selective_packing": ("INT",
                                      {"default": 0, "min": 0, "max": 128, "tooltip": TOOLTIPS["selective_packing"]}),
                "verbose": ("BOOLEAN", {"default": True, "tooltip": TOOLTIPS["verbose"]}),
                "compile": ("BOOLEAN", {"default": False, "tooltip": TOOLTIPS["compile"]}),
            },
            "optional": {
                "gpu_layer_indices": ("STRING", {"default": "", "tooltip": TOOLTIPS["gpu_layers"]}),
                #Precision tools have been disabled due to confliction with comfy auto precision and casting


                # "compute_casting": (["disabled", "fp32", "bf16", "fp16"],
                #                     {"default": "disabled",
                #                      "tooltip": "Cast to higher precision for computation only. Model stays in original dtype for storage/transfer, temporarily upcasts during forward pass for better accuracy"}),
                # "cast_target": ("STRING", {"default": "", "tooltip": TOOLTIPS["cast_target"]}),
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
                                 prefetch, threading, event_sync, cuda_streams, batch_move,
                                 selective_packing, verbose, compile, compute_casting=False, gpu_layer_indices="",
                                 cast_target="", autocast="", mixed_precision="auto"):

        # Force cleanup before starting
        torch.cuda.empty_cache()
        gc.collect()

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
        args.batch_move = batch_move
        args.selective_packing = selective_packing
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
        args.cast_target = False #cast_target.strip() if cast_target.strip() else None
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
                print("Applying ENHANCED smart GPU allocation + dynamic swapping...")
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

            # PHASE 1: Determine which layers go where based on specified layers or initial/final counts
            for i, layer in enumerate(layers):
                if hasattr(layer, 'to'):
                    if specified_gpu_layers is not None:
                        # Use specified layer indices for GPU placement
                        if i in specified_gpu_layers:
                            try:
                                layer.to(GPU_DEVICE)
                                gpu_resident_layers.add(i)
                                if VERBOSE:
                                    print(f"Layer {i} (specified) -> GPU permanent")
                            except RuntimeError as e:
                                print(f" CRITICAL: Cannot fit specified layer {i} on GPU!")
                                print(f"GPU memory may be insufficient. Consider removing layer {i} from --gpu_layers")
                                raise e
                        else:
                            # Not in specified list, make swappable
                            layer.to(CPU_DEVICE)
                            cpu_swappable_layers.add(i)
                            if VERBOSE:
                                print(f"Layer {i} (not specified) -> CPU swappable")

                    else:
                        # Use original initial/final logic as fallback
                        if i < initial_gpu_layers:
                            # Initial layers on GPU permanently
                            try:
                                layer.to(GPU_DEVICE)
                                gpu_resident_layers.add(i)
                                if VERBOSE:
                                    print(f"Layer {i} (initial) -> GPU permanent")
                            except RuntimeError as e:
                                print(f"GPU exhausted at layer {i}, moving to CPU with swapping")
                                layer.to(CPU_DEVICE)
                                cpu_swappable_layers.add(i)

                        elif i >= (len(layers) - final_gpu_layers):
                            # Final layers on GPU permanently
                            try:
                                layer.to(GPU_DEVICE)
                                gpu_resident_layers.add(i)
                                if VERBOSE:
                                    print(f"Layer {i} (final) -> GPU permanent")
                            except RuntimeError as e:
                                print(f"CRITICAL: Cannot fit final layer {i} on GPU!")
                                raise e
                        else:
                            # Middle layers on CPU with swapping capability
                            layer.to(CPU_DEVICE)
                            cpu_swappable_layers.add(i)
                            print(f"Layer {i} (middle) -> CPU swappable")

            print(f"✓ {len(gpu_resident_layers)} layers permanently on GPU: {sorted(gpu_resident_layers)}")
            print(f"✓ {len(cpu_swappable_layers)} layers on CPU with smart swapping: {sorted(cpu_swappable_layers)}")


            if cast_target:
                casting_handler = CastingHandler()
                casting_handler.precast_all_layers(cpu_swappable_layers, layers)

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

            print(" Pre-calculating layer sizes...")
            layer_sizes_mb = {}
            for i, layer in enumerate(layers):
                layer_sizes_mb[i] = sum(p.numel() * p.element_size() for p in layer.parameters()) / (1024 * 1024)
                if VERBOSE:
                    print(f"   Layer {i}: {layer_sizes_mb[i]:.1f}MB")

            if args.cast_target:
                print(f"\n=== LAYER CASTING DTYPES ===")
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

        for layer_idx in cpu_swappable_layers:
            add_smart_swapping_to_layer(
                layers[layer_idx],
                layer_idx,
                layers,
                gpu_resident_layers,
                cpu_swappable_layers)

        if VERBOSE:
            print("Adding compute timing to permanent GPU layers...")
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

                        if VERBOSE:
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

                        if VERBOSE:
                            layer_compute_end = time.time()
                            layer_compute_time = layer_compute_end - layer_compute_start
                            add_smart_swapping_to_layer.layer_compute_times.append(layer_compute_time)
                        return result

                    return resident_forward

                layer.forward = create_resident_forward(layer_idx, original_forward, compiled_forward)
                if VERBOSE:
                    print(f" Added compiled timing to permanent GPU layer {layer_idx}")

            else:
                # Original working version (no compile)
                def create_resident_forward(layer_idx, original_forward):
                    current_args = args
                    @wraps(original_forward)
                    def resident_forward( *args_tuple, **kwargs):

                        if VERBOSE:
                            layer_compute_start = time.time()


                        if current_args.compute_casting != "disabled":
                            dtype_map = {'fp32': torch.float32, 'bf16': torch.bfloat16, 'fp16': torch.float16}
                            cast_dtype = dtype_map[args.compute_casting]

                            with torch.autocast(device_type='cuda', dtype=cast_dtype, enabled=True):
                                result = original_forward( *args_tuple, **kwargs)
                        else:
                            result = original_forward( *args_tuple, **kwargs)

                        # result = original_forward( *args_tuple, **kwargs)

                        if VERBOSE:
                            layer_compute_end = time.time()
                            layer_compute_time = layer_compute_end - layer_compute_start
                            add_smart_swapping_to_layer.layer_compute_times.append(layer_compute_time)
                        return result

                    return resident_forward

                layer.forward = create_resident_forward(layer_idx, original_forward)
                if VERBOSE:
                    print(f"✓ Added timing to permanent GPU layer {layer_idx}")

                # if step % 5 == 0:
                #     print(f"\n=== STEP {step} STATS ===")
                #     print(f"Expected total steps: {model_engine.total_steps}")
                #     if (add_smart_swapping_to_layer.prefetch_hits + add_smart_swapping_to_layer.prefetch_misses) > 0:
                #         hit_rate = add_smart_swapping_to_layer.prefetch_hits / (
                #                     add_smart_swapping_to_layer.prefetch_hits + add_smart_swapping_to_layer.prefetch_misses) * 100
                #         print(f"Prefetch hit rate: {hit_rate:.1f}% ({add_smart_swapping_to_layer.prefetch_hits}/{add_smart_swapping_to_layer.prefetch_hits + add_smart_swapping_to_layer.prefetch_misses})")
                #     print("========================")


        if VERBOSE:
            print("✅ Dynamic swapping successfully integrated with ComfyUI inference")

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


    def precast_all_layers(self, cpu_swappable_layers, layers):
        """Cast all layer dtypes once at startup using existing cast_to_dtype"""
        print(" Pre-casting layer dtypes (one-time setup)...")
        casted_layers = 0

        for i, layer in enumerate(layers):
            if i in cpu_swappable_layers:  # Only cast swappable layers
                layer_casted = False

                for name, param in layer.named_parameters():
                    # Use your existing function but stay on current device
                    new_param = self.cast_to_dtype(param, param.device)  # Keep on same device
                    if new_param.dtype != param.dtype:
                        param.data = new_param.data
                        layer_casted = True

                if layer_casted:
                    casted_layers += 1

        print(f"✓ Pre-casted {casted_layers} layers using existing cast_to_dtype function")
