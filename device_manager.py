import psutil
import socket
import os
from typing import List, Tuple
import math

class Device:
    def __init__(self, hostname: str, ip: str, port: int, memory: int):
        self.hostname = hostname
        self.ip = ip
        self.port = port
        self.memory = memory  # Memory in MB

    def __str__(self):
        return f"{self.hostname} ({self.ip}:{self.port}, {self.memory} MB)"

class DeviceManager:
    def __init__(self, devices: List[Tuple[str, str, int]] = None):
        """Initialize with a list of (hostname, ip, port) or auto-detect local device."""
        self.devices = []
        if devices:
            for hostname, ip, port in devices:
                memory = self.get_memory_capacity()
                self.devices.append(Device(hostname, ip, port, memory))
        else:
            # Auto-detect local device
            hostname = socket.gethostname()
            ip = socket.gethostbyname(hostname)
            port = 9999  # Default port
            memory = self.get_memory_capacity()
            self.devices.append(Device(hostname, ip, port, memory))
        # Sort devices by memory capacity (descending)
        self.devices.sort(key=lambda x: x.memory, reverse=True)

    def get_memory_capacity(self) -> int:
        """Get available memory in MB."""
        return psutil.virtual_memory().available // (1024 * 1024)

    def select_devices(self, model_size_mb: int, max_nodes: int) -> List[Device]:
        """Select the minimum number of devices to fit the model."""
        # Add 20% overhead for runtime buffers
        model_size_mb = int(model_size_mb * 1.2)
        nodes = 1
        while nodes <= max_nodes and nodes <= len(self.devices):
            memory_per_node = model_size_mb / nodes
            # Check if the top `nodes` devices have enough memory
            selected = self.devices[:nodes]
            if all(device.memory >= memory_per_node for device in selected):
                return selected
            nodes *= 2  # Distributed Llama requires 2^n nodes
        raise Exception(f"No combination of {max_nodes} or fewer devices can handle model size {model_size_mb} MB")

    def get_worker_args(self, selected_devices: List[Device]) -> str:
        """Generate worker addresses for dllama command."""
        if len(selected_devices) == 1:
            return ""  # No workers needed for single device
        worker_args = " ".join(f"{device.ip}:{device.port}" for device in selected_devices[1:])
        return f"--workers {worker_args}"

    def distribute_model_files(self, model_path: str, tokenizer_path: str, devices: List[Device]):
        """Verify model and tokenizer files are available on each device."""
        for device in devices:
            if not os.path.exists(model_path) or not os.path.exists(tokenizer_path):
                raise Exception(f"Model or tokenizer not found on {device}: {model_path}, {tokenizer_path}")