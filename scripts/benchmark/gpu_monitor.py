"""GPU monitoring utilities. Falls back gracefully if pynvml unavailable."""

import subprocess
import torch


class GPUMonitor:
    """Monitor GPU utilization and memory on Windows via nvidia-smi."""

    def __init__(self, device_index=0):
        self.device_index = device_index
        self._has_smi = self._check_nvidia_smi()
        if not self._has_smi:
            print("WARNING: nvidia-smi not available, GPU utilization monitoring disabled")

    def _check_nvidia_smi(self):
        try:
            subprocess.run(
                ["nvidia-smi", "--version"],
                capture_output=True, timeout=5,
            )
            return True
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False

    def get_utilization(self):
        """Return GPU utilization as a float 0-1, or -1 if unavailable."""
        if not self._has_smi:
            return -1.0
        try:
            result = subprocess.run(
                ["nvidia-smi",
                 "--query-gpu=utilization.gpu",
                 "--format=csv,noheader,nounits",
                 f"--id={self.device_index}"],
                capture_output=True, text=True, timeout=5,
            )
            return float(result.stdout.strip()) / 100.0
        except Exception:
            return -1.0

    def get_memory_used_mb(self):
        """Return CUDA memory currently allocated in MB."""
        return torch.cuda.memory_allocated(self.device_index) / 1024 ** 2

    def get_peak_memory_mb(self):
        """Return peak CUDA memory allocated in MB."""
        return torch.cuda.max_memory_allocated(self.device_index) / 1024 ** 2

    def get_memory_total_mb(self):
        """Return total GPU memory in MB."""
        return torch.cuda.get_device_properties(self.device_index).total_memory / 1024 ** 2

    def get_competing_processes(self):
        """Check for other GPU processes. Returns list of (pid, name, mem_mb)."""
        if not self._has_smi:
            return []
        try:
            result = subprocess.run(
                ["nvidia-smi",
                 "--query-compute-apps=pid,process_name,used_gpu_memory",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5,
            )
            import os
            my_pid = os.getpid()
            procs = []
            for line in result.stdout.strip().split("\n"):
                if not line.strip():
                    continue
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 3:
                    pid = int(parts[0])
                    if pid != my_pid:
                        procs.append((pid, parts[1], parts[2]))
            return procs
        except Exception:
            return []

    def check_and_warn(self, threshold=0.90):
        """Check utilization and warn if below threshold. Returns utilization."""
        util = self.get_utilization()
        if 0 <= util < threshold:
            print(
                f"  WARNING: GPU utilization {util*100:.0f}% < {threshold*100:.0f}% threshold"
            )
            procs = self.get_competing_processes()
            if procs:
                print(f"  Competing GPU processes found:")
                for pid, name, mem in procs:
                    print(f"    PID {pid}: {name} ({mem} MB)")
        return util
