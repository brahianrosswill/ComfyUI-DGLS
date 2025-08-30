# ComfyUI - DGLS (Dynamic GPU Layer Swapping) 

Smart dynamic layer swapping between GPU and CPU for optimal inference performance with comprehensive mixed precision handling and copy-compute overlap optimization. Enables running much larger models on limited VRAM setups.

NOTE: I am am actively bug testing this atm. The code is working, however I havent promoted this or written about it due to some tests I want to carry out. README is for an old version adn will be updated shortly

Currently only working with diffusion models. So any workflow with 'Load Diffusion Model' offical node should be fine with this swapped for it. OmniGen not working atm. 

## Features

* **Inference Optimized**: Designed specifically for ComfyUI diffusion model inference
* **Mixed Precision Support**: Handles multiple dtypes and precision casting
* **Threading and Copy-Compute Overlap**: Background thread management with overlapped memory transfers
* **Predictive Prefetching**: Intelligently predicts and preloads layers before they're needed
* **Packed Transfers**: Optimizes large layer transfers using contiguous memory packing
* **CUDA Streams**: Uses CUDA streams to overlap memory transfers with computation
* **Model Compatibility**: Supports Cosmos, Flux, Wan2.1, Wan2.2, HunyuanVideo, and generic transformer blocks.

This memory management system maintains inference performance while dramatically reducing VRAM requirements through intelligent layer orchestration.

## Performance

Achieved a 30% reduction in speed compared to other offloading techniques in comfy. 

## System Requirements

**GPU Compatibility**: CUDA-capable GPUs, tested on RTX 2060 and RTX 2080 Ti

**Memory Requirements**:
- **RAM**: 16GB minimum, 32GB recommended for large models
- **RAM Speed**: 3200MHz+ recommended for optimal transfer speeds
- **VRAM**: 6GB+ (enables inference of models that normally require 16GB+)

## Installation
### ComfyUI Custom Nodes Installation

1. Navigate to your ComfyUI custom nodes directory:
```bash
cd ComfyUI/custom_nodes/
```

2. Clone or copy the DGLS files:
```bash
git clone https://github.com/obisin/dgls-comfyui
# or manually place the files in a new folder
```

3. Restart ComfyUI - the nodes should appear in the loaders category

### Required Files
- `dynamic_swapping_loader.py` - Main swapping logic and ComfyUI node
- `dgls_model_loader.py` - Model loader with layer extraction

## How It Works

DGLS implements intelligent layer management during inference by maintaining a **sliding window** of active layers on GPU while keeping others on CPU.

### Inference Process
```
Sampling Step:  [GPU: layers 0,1,2,20,21] [CPU: layers 3,4,5,...,19]
                ↓ Permanent residents + sliding window
Forward Pass:   [GPU: layers 0,1,2,3,4] [CPU: layers 5,6,...,19,20,21]
                ↓ Compute layer 2, prefetch layer 4
Next Layer:     [GPU: layers 0,1,3,4,5] [CPU: layers 2,6,7,...,19,20,21]
                ↓ Evict layer 2, compute layer 3, prefetch layer 5
```

**Key Principles**:
- **Permanent Residents**: First/last layers stay on GPU for stability
- **Sliding Window**: Middle layers swap dynamically based on computation needs
- **Prefetching**: Next layers loaded while current layer computes
- **Background **: Synchronous transfers overlap with computation

## Node Usage

### DGLS Model Loader Node

**Inputs**:
- `model_name`: Select from available diffusion models
- `model_type`: Choose model architecture type (if model not on list pick default)
- `cast_dtype`: Target dtype for model weights
- `verbose`: Enable detailed logging

**Outputs**:
- `model`: Loaded ComfyUI model
- `layers`: Extracted layers for swapping

### DGLS Swapping Loader Node

**Required Inputs**:
- `model`: Model from DGLS Model Loader
- `layers`: Layers from DGLS Model Loader
- `initial_gpu_layers`: Number of initial layers to keep on GPU (1-10)
- `final_gpu_layers`: Number of final layers to keep on GPU (1-10)
- `prefetch`: Number of layers to prefetch ahead (0-4)

**Optional Performance Settings**:
- `threading`: Enable background threading for better overlap (May not work with all models or systems)
- `cuda_streams`: Enable CUDA streams (requires more VRAM)
- `batch_move`: Move multiple layers simultaneously
- `selective_packing`: Size threshold for packed transfers (MB)
- `event_sync`: Use CUDA events for better synchronization
- `compile`: Enable torch.compile for permanent GPU layers

**Advanced Options**:
- `gpu_layer_indices`: Specify exact layers to keep on GPU (comma-separated)
- `verbose`: Enable detailed timing and transfer information

## Configuration Examples

### Conservative Setup (6GB VRAM)
```
initial_gpu_layers: 2
final_gpu_layers: 2
prefetch: 1
threading: False
event_sync: True
```

### Balanced Setup (8-12GB VRAM)
```
initial_gpu_layers: 3
final_gpu_layers: 3
prefetch: 2
threading: True
batch_move: True
selective_packing: 64
event_sync: True
```

### High Performance (16GB+ VRAM)
```
initial_gpu_layers: 5+
final_gpu_layers: 5+
prefetch: 3+
threading: True
cuda_streams: True
batch_move: True
selective_packing: 128
compile: True
event_sync: True
```

## Workflow Integration

Works in place of the **Load Diffusion Model** official comfy node.

### Basic Workflow
1. **DGLS Model Loader** → Load your diffusion model
2. **Dynamic Swapping Loader** → Apply swapping configuration
3. **Connect to samplers** → Use swapping-enabled model for inference

## Arguments Reference

NOTE: All settings can be stacked together!

### Core Settings

**`initial_gpu_layers` / `final_gpu_layers`**
- **Purpose**: Number of layers to keep permanently on GPU
- **Recommendation**: Start with 2-3 each, increase if you have more VRAM

**`prefetch`**
- **Purpose**: Number of layers to preload ahead of current computation
- **Recommendation**: 1-2 for most setups, higher with more VRAM

**`gpu_layer_indices`**
- **Format**: Comma-separated numbers (e.g., "0,1,2,18,19,20")
- **Purpose**: Precise control over GPU-resident layers
- **Use**: Overrides initial/final settings when specified

### Performance Optimization

**`threading`**
- **Purpose**: Background thread for automatic layer management
- **Benefit**: Better overlap of transfers with computation
- **Note**: If you're using a smaller model or one with a fast compute per layer, threading may be counter productive as the swap will lag behind the fast compute. Instead increase resident GPU layers and prefetch
- **Note**: May cause instability in some systems

**`cuda_streams`**
- **Purpose**: CUDA streams for copy-compute overlap
- **Requirement**: Additional VRAM for stream buffers
- **Benefit**: Maximum performance when VRAM allows
- **Note**: May not work with all models, Flux can sometimes have issues

**`batch_move`**
- **Purpose**: Move multiple layers simultaneously
- **Benefit**: Better for GPUs with higher bandwidth

**`selective_packing`**
- **Purpose**: Pack large layers for optimized transfers

**`event_sync`**
- **Purpose**: Use CUDA events instead of basic synchronization
- **Benefit**: Better performance in most cases
- **Recommendation**: Enable unless experiencing issues

**`compile`**
- **Purpose**: torch.compile optimization for permanent GPU layers
- **Requirement**: PyTorch 2.0+
- **Benefit**: Additional speedup for resident layers

## Troubleshooting

First try restarting Comfy after an error or OOM.

### Memory Issues

**"CUDA out of memory" during startup**:
- Reduce `initial_gpu_layers` and `final_gpu_layers` to 1
- Disable `cuda_streams` or `selective_packing`
- Use lower `prefetch` values

### Performance Issues

**Slow inference**:
- Enable `threading` and `event_sync`
- Increase `prefetch` if you have spare VRAM
- Try `batch_move` for newer GPUs

**Low prefetch efficiency**:
- Reduce `prefetch` value
- Enable `threading` for background management
- Check VRAM usage and reduce permanent layers

### Stability Issues

**Crashes with threading**:
- Disable `threading`
- Use conservative `prefetch` values
- Enable `verbose` to debug timing issues

**Device synchronization errors**:
- Disable `cuda_streams`

## Technical Details

### Performance Optimizations
- **Hot Path Functions**: Module-level functions with local variable binding
- **Minimal Call Overhead**: Direct function calls without abstraction layers
- **Factory Pattern**: Clean interfaces with fast execution
- **Cache-Aware Design**: Efficient device state tracking

### Memory Management
- **Three-Tier Hierarchy**: GPU residents, sliding window, CPU storage
- **Packed Transfers**: Contiguous memory for better PCIe utilization
- **Event-Based Sync**: CUDA events for precise timing control

## Limitations

**Single GPU Only**: Designed for single GPU inference setups- Will not conflict if you set your vae or clip to another GPU for storage though.

**ComfyUI Specific**: Optimized for ComfyUI's inference patterns and model structures

**Model Architecture**: Some complex multi-branch models may need manual layer specification


## Attribution & License

**Dynamic Swapping Method**: Original research and implementation by **obisin**

**MIT License** - See LICENSE file for full details
