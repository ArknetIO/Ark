# Execution Domains

The `[domain]` attribute defines the execution context, threading model, and memory access rights of a function.

## 1. Domain Types

### Host
* **Attribute:** `fn` or `fn[host]`
* **Context:** Main Application Thread.
* **Execution:** Serial.
* **Role:** Orchestration, memory allocation, I/O, and dispatch.
* **Memory:** Full access to system RAM.

### CPU Kernel
* **Attribute:** `fn[cpu]`
* **Context:** Worker Thread Pool.
* **Execution:** Parallel Loop (SPMD).
* **Role:** Data preprocessing, complex logic, branching algorithms.
* **Memory:** Access to system RAM (passed via arguments).

### GPU Kernel
* **Attribute:** `fn[gpu]`
* **Context:** Graphics Processing Unit (CUDA/Metal/HIP).
* **Execution:** Massively Parallel SIMD.
* **Role:** Matrix arithmetic, heavy floating-point throughput.
* **Memory:** Strict access to VRAM (tagged `@gpu`).

## 2. Invocation Rules

Control flow is restricted to maintain synchronization safety.

| Caller | Callee | Type | Syntax |
| :--- | :--- | :--- | :--- |
| **Host** | **Host** | Direct Call | `let res = func(arg);` |
| **Host** | **Device** | Async Launch | `Grid <- Kernel(Args) as Token;` |
| **Device** | **Device** | Inline Call | `let val = func(arg);` |
| **Device** | **Host** | **Illegal** | N/A |

## 3. ABI Transformation

* **Host:** Standard C ABI.
* **Kernels:** Transformed into a "Runtime Wrapper" that accepts a generic `void*` argument pack and iterates over the grid dimension.



## 4. Unified Example

This example demonstrates a pipeline where the **Host** allocates memory, the **CPU** normalizes data, and the **GPU** performs heavy calculation.

```ark
// 1. [CPU] Pre-processing Kernel
// Normalizes input data (runs on Worker Threads)
fn[cpu] normalize(val: f32, max: f32) -> f32 {
    if (val > max) { return max; }
    return val / max;
}

// 2. [GPU] Compute Kernel
// Performs heavy arithmetic (runs on Device Cores)
fn[gpu] matrix_scale(val: f32, scale: f32) -> f32 {
    return val * scale + 1.0;
}

// 3. [Host] Orchestrator
fn[host] main() {
    let size = 1024;
    
    // Allocation: RAM vs VRAM
    let cpu_buf = alloc<f32>(size);
    let gpu_buf = alloc<f32>(size) @gpu:0;

    // A. Launch CPU Pre-processing
    // 'cpu_buf' is the grid. 
    cpu_buf <- normalize(cpu_buf, 255.0) as t_cpu;

    // B. Launch GPU Computation (Async)
    // We can dispatch this immediately; it will queue on the device.
    gpu_buf <- matrix_scale(gpu_buf, 2.0) as t_gpu;

    // C. Synchronization
    // Host waits for both to complete
    await t_cpu;
    await t_gpu;

    print "Pipeline complete.";
}