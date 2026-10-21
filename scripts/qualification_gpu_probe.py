#!/usr/bin/env python3
"""Finite CUDA allocation/copy/free smoke used by the live handoff harness.

This helper makes no timing or performance claim.  It allocates one bounded
device buffer, writes a deterministic byte pattern, reads it back, verifies it,
and frees it before exiting.
"""

from __future__ import annotations

import argparse
import ctypes
import ctypes.util
import json
import os
from pathlib import Path
import re
import sys
import uuid


MAX_ALLOCATION_BYTES = 256 * 1024 * 1024
MIN_ALLOCATION_BYTES = 1024 * 1024


def validate_probe_request(gpu_uuid: str, memory_bytes: int, visible: str | None) -> None:
    if not isinstance(gpu_uuid, str) or not re.fullmatch(r"GPU-[A-Fa-f0-9-]{16,64}", gpu_uuid):
        raise ValueError("gpu_uuid must be a stable NVIDIA GPU UUID")
    if isinstance(memory_bytes, bool) or not isinstance(memory_bytes, int):
        raise ValueError("memory_bytes must be an integer")
    if not MIN_ALLOCATION_BYTES <= memory_bytes <= MAX_ALLOCATION_BYTES:
        raise ValueError("memory_bytes must be between 1 MiB and 256 MiB")
    if visible != gpu_uuid:
        raise ValueError("CUDA_VISIBLE_DEVICES must name exactly the granted GPU UUID")


def _cuda_error(cuda: ctypes.CDLL, code: int, operation: str) -> None:
    if code == 0:
        return
    cuda.cudaGetErrorString.argtypes = [ctypes.c_int]
    cuda.cudaGetErrorString.restype = ctypes.c_char_p
    message = cuda.cudaGetErrorString(code)
    detail = message.decode("utf-8", errors="replace") if message else f"CUDA error {code}"
    raise RuntimeError(f"{operation} failed: {detail}")


def runtime_visible_device_uuid(cuda: object, *, driver: object | None = None) -> str:
    """Resolve ordinal zero's physical UUID through the CUDA driver ABI.

    CUDA Runtime has no ``cudaDeviceGetUuid`` symbol. Once the runtime has
    selected ordinal zero and initialized its primary context, the driver API
    reports the device bound to that context and its physical UUID. This binds
    the receipt to the actual CUDA device, rather than trusting visibility
    environment variables or a requested UUID.

    ``driver`` is injectable so the ctypes ABI contract can be tested without
    loading hardware libraries or accessing a GPU.
    """
    cuda.cudaSetDevice.argtypes = [ctypes.c_int]
    cuda.cudaSetDevice.restype = ctypes.c_int
    cuda.cudaFree.argtypes = [ctypes.c_void_p]
    cuda.cudaFree.restype = ctypes.c_int
    _cuda_error(cuda, cuda.cudaSetDevice(0), "cudaSetDevice")
    # cudaFree(0) initializes the runtime primary context for ordinal zero.
    _cuda_error(cuda, cuda.cudaFree(None), "cudaFree(0) context initialization")

    if driver is None:
        driver_name = ctypes.util.find_library("cuda") or "libcuda.so.1"
        driver = ctypes.CDLL(driver_name)
    driver.cuInit.argtypes = [ctypes.c_uint]
    driver.cuInit.restype = ctypes.c_int
    driver.cuCtxGetDevice.argtypes = [ctypes.POINTER(ctypes.c_int)]
    driver.cuCtxGetDevice.restype = ctypes.c_int
    get_uuid = getattr(driver, "cuDeviceGetUuid_v2", None)
    if get_uuid is None:
        get_uuid = getattr(driver, "cuDeviceGetUuid", None)
    if get_uuid is None:
        raise RuntimeError("CUDA driver exports neither cuDeviceGetUuid_v2 nor cuDeviceGetUuid")
    get_uuid.argtypes = [ctypes.c_void_p, ctypes.c_int]
    get_uuid.restype = ctypes.c_int

    def check_driver(code: int, operation: str) -> None:
        if code == 0:
            return
        detail = f"CUDA driver error {code}"
        get_error = getattr(driver, "cuGetErrorString", None)
        if get_error is not None:
            get_error.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_char_p)]
            get_error.restype = ctypes.c_int
            message = ctypes.c_char_p()
            if get_error(code, ctypes.byref(message)) == 0 and message.value:
                detail = message.value.decode("utf-8", errors="replace")
        raise RuntimeError(f"{operation} failed: {detail}")

    check_driver(driver.cuInit(0), "cuInit")
    device = ctypes.c_int()
    check_driver(driver.cuCtxGetDevice(ctypes.byref(device)), "cuCtxGetDevice")
    raw_uuid = ctypes.create_string_buffer(16)
    check_driver(get_uuid(raw_uuid, device.value), "cuDeviceGetUuid")
    return "GPU-" + str(uuid.UUID(bytes=raw_uuid.raw))


def run_probe(gpu_uuid: str, memory_bytes: int, *, cuda_library: str = "libcudart.so") -> dict[str, object]:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    validate_probe_request(gpu_uuid, memory_bytes, visible)
    cuda = ctypes.CDLL(cuda_library)
    cuda.cudaGetDeviceCount.argtypes = [ctypes.POINTER(ctypes.c_int)]
    cuda.cudaGetDeviceCount.restype = ctypes.c_int
    cuda.cudaMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
    cuda.cudaMalloc.restype = ctypes.c_int
    cuda.cudaMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
    cuda.cudaMemcpy.restype = ctypes.c_int
    cuda.cudaFree.argtypes = [ctypes.c_void_p]
    cuda.cudaFree.restype = ctypes.c_int

    count = ctypes.c_int()
    _cuda_error(cuda, cuda.cudaGetDeviceCount(ctypes.byref(count)), "cudaGetDeviceCount")
    if count.value != 1:
        raise RuntimeError(f"expected one visible CUDA device, found {count.value}")
    actual_uuid = runtime_visible_device_uuid(cuda)
    if actual_uuid.casefold() != gpu_uuid.casefold():
        raise RuntimeError("CUDA device UUID does not match the parent grant")

    device = ctypes.c_void_p()
    _cuda_error(cuda, cuda.cudaMalloc(ctypes.byref(device), memory_bytes), "cudaMalloc")
    host = ctypes.create_string_buffer(memory_bytes)
    returned = ctypes.create_string_buffer(memory_bytes)
    seed = bytes((index * 37 + 11) & 0xFF for index in range(4096))
    for offset in range(0, memory_bytes, len(seed)):
        amount = min(len(seed), memory_bytes - offset)
        ctypes.memmove(ctypes.addressof(host) + offset, seed, amount)
    try:
        _cuda_error(cuda, cuda.cudaMemcpy(device, host, memory_bytes, 1), "cudaMemcpy host-to-device")
        _cuda_error(cuda, cuda.cudaMemcpy(returned, device, memory_bytes, 2), "cudaMemcpy device-to-host")
        if returned.raw != host.raw:
            raise RuntimeError("CUDA round-trip data verification failed")
    finally:
        _cuda_error(cuda, cuda.cudaFree(device), "cudaFree")
    return {
        "schema_version": 1,
        "status": "passed",
        "test": "finite-cuda-allocation-copy-free",
        "gpu_uuid": actual_uuid,
        "memory_bytes": memory_bytes,
        "verified_bytes": memory_bytes,
        "pid": os.getpid(),
        "performance_claim": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument("--memory-bytes", type=int, required=True)
    args = parser.parse_args(argv)
    try:
        result = run_probe(args.gpu_uuid, args.memory_bytes)
    except Exception as exc:
        print(json.dumps({"status": "failed", "error": type(exc).__name__, "message": str(exc)[:300]}))
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
