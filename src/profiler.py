"""Resource profiling for RQ2 (space) and RQ3 (efficiency / energy).

- Peak GPU memory via torch.cuda.max_memory_allocated() / max_memory_reserved()
- Wall-clock duration, samples/sec and tokens/sec
- Energy in Wh from NVML: the GPU's cumulative energy counter when supported,
  otherwise power samples integrated over time
- Time-to-First-Token (TTFT) and decode throughput for a single generate() call
- Trainable-parameter ratio of a PEFT model
"""

import csv
import threading
import time
from pathlib import Path

import torch
from transformers import StoppingCriteria, StoppingCriteriaList

try:
    import pynvml
except ImportError:  # energy is reported as None without NVML
    pynvml = None

GIB = 1024**3


class EnergyMeter:
    """GPU energy in Wh between start() and stop()."""

    def __init__(self, device_index=0, interval_s=0.1):
        self.device_index = device_index
        self.interval_s = interval_s
        self.handle = None
        self.method = None
        self._thread = None
        self._stop = threading.Event()
        self._joules = 0.0
        self._start_mj = None

    def start(self):
        if pynvml is None:
            return
        pynvml.nvmlInit()
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(self.device_index)
        try:
            self._start_mj = pynvml.nvmlDeviceGetTotalEnergyConsumption(self.handle)
            self.method = "nvml_energy_counter"
        except pynvml.NVMLError:
            self.method = "nvml_power_sampling"
            self._stop.clear()
            self._thread = threading.Thread(target=self._sample, daemon=True)
            self._thread.start()

    def _sample(self):
        last = time.perf_counter()
        while not self._stop.is_set():
            time.sleep(self.interval_s)
            now = time.perf_counter()
            watts = pynvml.nvmlDeviceGetPowerUsage(self.handle) / 1000.0
            self._joules += watts * (now - last)
            last = now

    def stop(self):
        """Return energy in Wh (None if NVML is unavailable)."""
        if self.handle is None:
            return None
        if self.method == "nvml_energy_counter":
            joules = (pynvml.nvmlDeviceGetTotalEnergyConsumption(self.handle) - self._start_mj) / 1000.0
        else:
            self._stop.set()
            self._thread.join()
            joules = self._joules
        pynvml.nvmlShutdown()
        return joules / 3600.0


class GPUProfiler:
    """Context manager that records peak VRAM, duration, throughput and energy.

    Usage:
        with GPUProfiler() as prof:
            ... training loop ...
            prof.add(samples=bs, tokens=n_tokens)
        prof.results  # dict
    """

    def __init__(self, device_index=0):
        self.device_index = device_index
        self.energy = EnergyMeter(device_index)
        self.samples = 0
        self.tokens = 0
        self.results = {}

    def add(self, samples=0, tokens=0):
        self.samples += samples
        self.tokens += tokens

    def __enter__(self):
        self.cuda = torch.cuda.is_available()  # False only for CPU smoke tests
        if self.cuda:
            torch.cuda.synchronize(self.device_index)
            torch.cuda.reset_peak_memory_stats(self.device_index)
            self.energy.start()
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        if self.cuda:
            torch.cuda.synchronize(self.device_index)
        duration = time.perf_counter() - self._t0
        energy_wh = self.energy.stop() if self.cuda else None
        self.results = {
            "duration_s": round(duration, 3),
            "peak_vram_allocated_gib": round(torch.cuda.max_memory_allocated(self.device_index) / GIB, 3) if self.cuda else None,
            "peak_vram_reserved_gib": round(torch.cuda.max_memory_reserved(self.device_index) / GIB, 3) if self.cuda else None,
            "samples": self.samples,
            "tokens": self.tokens,
            "samples_per_s": round(self.samples / duration, 3) if duration else None,
            "tokens_per_s": round(self.tokens / duration, 2) if duration else None,
            "energy_wh": round(energy_wh, 3) if energy_wh is not None else None,
            "energy_method": self.energy.method,
            "gpu_name": torch.cuda.get_device_name(self.device_index) if self.cuda else "cpu",
        }
        return False


def cap_vram(gib, device=0):
    """Limit PyTorch's allocator to `gib` GiB on a larger GPU, so a run that would not fit
    a consumer card (e.g. 24 GB) fails with OOM here too. Returns the fraction set, or None."""
    if not gib or not torch.cuda.is_available():
        return None
    total = torch.cuda.get_device_properties(device).total_memory
    fraction = min(1.0, gib * GIB / total)
    torch.cuda.set_per_process_memory_fraction(fraction, device)
    return round(fraction, 4)


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


class _FirstTokenTimer(StoppingCriteria):
    """Records the time at which the first new token has been produced."""

    def __init__(self):
        self.first_token_time = None

    def __call__(self, input_ids, scores, **kwargs):
        if self.first_token_time is None:
            _sync()
            self.first_token_time = time.perf_counter()
        return torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)


@torch.no_grad()
def measure_generation(model, input_ids, attention_mask=None, **generate_kwargs):
    """Run one generate() call and return (output_ids, timing dict).

    TTFT covers the prefill forward pass plus the first decoding step.
    Decode throughput excludes the first token.
    """
    timer = _FirstTokenTimer()
    _sync()
    t0 = time.perf_counter()
    output = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        stopping_criteria=StoppingCriteriaList([timer]),
        **generate_kwargs,
    )
    _sync()
    total = time.perf_counter() - t0
    new_tokens = output.shape[1] - input_ids.shape[1]
    ttft = (timer.first_token_time - t0) if timer.first_token_time else total
    decode_time = total - ttft
    return output, {
        "ttft_s": round(ttft, 4),
        "generation_s": round(total, 4),
        "new_tokens": int(new_tokens),
        "tokens_per_s": round(new_tokens / total, 2) if total else None,
        "decode_tokens_per_s": round((new_tokens - 1) / decode_time, 2) if new_tokens > 1 and decode_time > 0 else None,
    }


def trainable_parameters(model):
    """Trainable vs. total parameters and adapter size in MB."""
    trainable = total = 0
    trainable_bytes = 0
    for p in model.parameters():
        n = p.numel()
        # bitsandbytes 4-bit weights are packed two per byte in a uint8 tensor
        if p.__class__.__name__ == "Params4bit":
            n *= 2
        total += n
        if p.requires_grad:
            trainable += n
            trainable_bytes += n * p.element_size()
    return {
        "trainable_params": trainable,
        "total_params": total,
        "trainable_pct": round(100 * trainable / total, 4) if total else None,
        "adapter_mb": round(trainable_bytes / 1024**2, 2),
    }


def append_csv(path, row):
    """Append one result row to a CSV, writing the header on first use."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if new_file:
            writer.writeheader()
        writer.writerow(row)
