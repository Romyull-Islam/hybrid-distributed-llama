# Hybrid Distributed Llama

> Built on [distributed-llama](https://github.com/b4rtaz/distributed-llama) by Bartłomiej Tadych (MIT). This repository adds memory-aware device selection, local `mmap`-based model loading on each node, and a small memory-check utility, aimed at running LLM inference across a heterogeneous cluster (for example a Mac Mini M1 together with Raspberry Pi 5 boards).

Hybrid Distributed Llama is an inference engine for large language models (LLMs) that runs across a hybrid cluster of devices. By splitting the model weights and computation across nodes, it targets resource-constrained setups such as a Mac Mini M1 combined with Raspberry Pi 5 boards.

## What this repository changes

Compared with the upstream base (commit `56b96ad`, "feat: optional network turbo"):

- **`device_manager.py`** (new): estimates per-node memory with `psutil` (currently the local machine's available memory is used for every listed device), picks the smallest power-of-two number of nodes whose memory fits the model size plus 20% overhead, and builds the `--workers` argument for `dllama`.
- **`device-check.cpp`** (new): prints the node's free memory in MB (uses `sysinfo` on Linux and `GlobalMemoryStatusEx` on Windows). Built by `make`.
- **`launch.py`**: each model entry carries an approximate size; new `--devices hostname,ip,port ...` option; files are downloaded only when missing; worker addresses are appended to the generated `dllama` command.
- **`src/app.cpp`, `src/llm.cpp`, `src/llm.hpp`**: the model header, weights and tokenizer are read from a local `mmap`-ed file (`loadLlmHeaderFromMemory`, `loadLlmNetWeightFromMemory`, `Tokenizer(void*)`).
- **`src/nn/nn-network.cpp`**: the root node no longer streams weight slices to workers over the network; each node is expected to load its own weights locally.
- **`src/dllama.cpp`, `src/app.hpp`, `src/tokenizer.cpp`**: reworked inference/chat loop and tokenizer to match the memory-based loading.
- **`Makefile`**: rewritten (C++17, `build/` directory, `device-check` target).

**Status:** research prototype, work in progress. In the current code the worker-side loader (`NnWorkerWeightReader::read()`) is a placeholder and `dllama`'s `main` runs only the `inference`/`chat` modes, so the multi-device path is not complete yet.

## Features

- **Distributed Inference** (from upstream): Split LLM inference across multiple devices to combine their memory and compute.
- **Local Model Loading**: Each node loads the model from its own local storage using `mmap` instead of receiving weights from the root node over the network.
- **Dynamic Device Selection**: Automatically select devices based on memory capacity using `device_manager.py`.
- **ARM targets**: Intended for ARM devices such as Mac Mini M1 and Raspberry Pi 5.
- **Chat and Inference Modes**: Interactive chat mode or batch inference with configurable steps.
- **Vulkan Support** (from upstream): Optional GPU acceleration with Vulkan (define `DLLAMA_VULKAN` during compilation, requires Vulkan SDK and shader compilation).

## Repository Structure

- `src/`: Core C++ source files (`app.cpp`, `dllama.cpp`, `nn-network.cpp`, etc.).
- `device_manager.py`: Python script for dynamic device selection based on memory capacity.
- `launch.py`: Script to download models, configure devices, and launch Hybrid Distributed Llama.
- `device-check.cpp`: Utility to check device memory capacity.
- `Makefile`: Build configuration for compiling the project.
- `models/`: Directory where model and tokenizer files are stored (created by `launch.py`).

## Installation

### Prerequisites

#### Mac Mini M1

- **OS**: macOS (e.g., Ventura or later)
- **Tools**:
  - Xcode Command Line Tools: `xcode-select --install`
  - Homebrew: `/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"`
  - Python 3: `brew install python3`
  - CMake: `brew install cmake`
  - GCC: `brew install gcc`
- **Python Packages**: `pip3 install psutil`
- **Optional (Vulkan)**: Install Vulkan SDK if using `DLLAMA_VULKAN`.

#### Raspberry Pi 5

- **OS**: Raspberry Pi OS (64-bit, Bookworm)
- **Tools**:
  - Update: `sudo apt update && sudo apt upgrade`
  - Build Tools: `sudo apt install build-essential cmake python3 python3-pip`
- **Python Packages**: `pip3 install psutil`
- **Optional (Vulkan)**: Install Vulkan libraries (`libvulkan-dev`) and `glslc` if using `DLLAMA_VULKAN`.

### Build Instructions

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/Romyull-Islam/hybrid-distributed-llama.git
   cd hybrid-distributed-llama
   ```

2. **Build the Project**:

   - Default build:

     ```bash
     make
     ```
   - With Vulkan support:

     ```bash
     make DLLAMA_VULKAN=1
     ```
   - With debug mode:

     ```bash
     make DEBUG=1
     ```
   - On Mac Mini M1, if you encounter issues, update the `Makefile` to use `g++-12`:

     ```makefile
     CXX = g++-12
     ```
   - On Raspberry Pi 5, if C++17 features fail, install `g++-10`:

     ```bash
     sudo apt install g++-10
     ```

     Update `Makefile`:

     ```makefile
     CXX = g++-10
     ```

3. **Verify Device Capacity**:

   ```bash
   ./device-check
   ```

   This prints the free memory of the device in MB. Note that `device-check.cpp` uses `sysinfo`, which is Linux-only (Windows is also handled); it does not build on macOS as-is.

## Usage

### Single Device

To run Hybrid Distributed Llama on a single device (e.g., Mac Mini M1):

```bash
python3 launch.py llama3_1_8b_instruct_q40 --run
```

This downloads the Llama 3.1 8B model (4.5 GB) to `models/llama3_1_8b_instruct_q40/` and starts the chat mode.

For a smaller model on Raspberry Pi 5 (e.g., 8 GB RAM):

```bash
python3 launch.py llama3_2_1b_instruct_q40 --run
```

### Multi-Device Setup

To run across multiple devices (e.g., Mac Mini M1 as root, Raspberry Pi 5 as worker):

1. Ensure all devices have the repository and model files in the same path.

   ```bash
   scp -r models/llama3_1_8b_instruct_q40 pi@192.168.0.2:/path/to/hybrid-distributed-llama/models/
   ```
2. Run on the root node (Mac Mini):

   ```bash
   python3 launch.py llama3_1_8b_instruct_q40 --devices macmini,192.168.0.1,9999 pi5,192.168.0.2,9999 --run
   ```

### Available Models

- `llama3_1_8b_instruct_q40` (4.5 GB)
- `llama3_1_405b_instruct_q40` (225 GB, requires multiple devices)
- `llama3_2_1b_instruct_q40` (1.2 GB)
- `llama3_2_3b_instruct_q40` (2 GB)
- `llama3_3_70b_instruct_q40` (35 GB)
- `deepseek_r1_distill_llama_8b_q40` (4.5 GB)

## Contributing

Contributions are welcome! Please open an issue or submit a pull request on GitHub.

## License

MIT License, inherited from upstream distributed-llama (© Bartłomiej Tadych). See `LICENSE` for details.
