# Hybrid Distributed Llama

Hybrid Distributed Llama is a high-performance inference engine for large language models (LLMs), designed to run across a hybrid cluster of devices. By distributing the model weights and computation, it enables efficient inference on resource-constrained setups, such as a combination of devices like Mac Mini M1 and Raspberry Pi 5. More devices mean faster inference!

## Features

- **Distributed Inference**: Split LLM inference across multiple devices to leverage combined computational power.
- **Local Model Loading**: Models are loaded from local storage using `mmap` for improved performance and reduced network overhead.
- **Dynamic Device Selection**: Automatically select devices based on memory capacity using `device_manager.py`.
- **Optimized for ARM**: Supports ARM architectures like Mac Mini M1 and Raspberry Pi 5.
- **Chat and Inference Modes**: Interactive chat mode or batch inference with configurable steps.
- **Vulkan Support**: Optional GPU acceleration with Vulkan (define `DLLAMA_VULKAN` during compilation, requires Vulkan SDK and shader compilation).

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

   This outputs the available memory (e.g., `Total memory: 16384 MB` on a 16 GB Mac Mini M1).

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

Hybrid Distributed Llama is licensed under the MIT License. See `LICENSE` for details.
