# Writing Kernels

A **Kernel** is a specialized function designed to be executed in parallel across a dataset. In ArkLang, kernels follow the **SPMD** (Single Program, Multiple Data) model: you write the logic for a single element, and the compiler expands it to process the entire array.

## 1. Defining a Kernel

Kernels are defined using the `fn` keyword with a specific domain attribute.

```ark
// CPU Kernel: Runs on worker threads
fn[cpu] my_cpu_kernel(...) { ... }

// GPU Kernel: Runs on the graphics device
fn[gpu] my_gpu_kernel(...) { ... }

```

### The Implicit Loop

Unlike a standard host function, a kernel does not contain a loop to iterate over data. Instead, the kernel **is** the body of the loop. The iteration count is determined by the size of the "Grid" (the destination array) provided at the call site.

## 2. Kernel Arguments

A kernel typically accepts two types of arguments:

1. **The Grid Element (Implicitly Indexed):**
The first argument usually represents the value of the grid at the current index `i`.
2. **Uniforms (Scalars):**
Additional arguments (like integers, floats, or structs) are "broadcast" to all threads. Every thread sees the same value.

```ark
// 'val' is the element at Grid[i]
// 'scale' is a constant scalar (10.0) passed from the host
fn[cpu] scale_kernel(val: f64, scale: f64) -> f64 {
    return val * scale;
}

```

## 3. Return Values & Output

Kernels in ArkLang are often "Map" operations. The return value of the kernel is automatically written back to the destination grid at the current index.

* **Input:** `Grid[i]`
* **Computation:** Kernel Logic
* **Return:** Written to `Grid[i]`

## 4. Restrictions

Because kernels run in a restricted parallel environment, they have specific limitations compared to Host functions:

* **No I/O:** You cannot use `print`, write to files, or open sockets.
* **No Allocations:** You cannot use `alloc` or `allocof`. All memory must be allocated by the Host and passed in.
* **No Global Side Effects:** Kernels should only modify their assigned output element. Modifying global variables or other indices without synchronization leads to race conditions.
* **No Host Calls:** A Kernel cannot call a `fn[host]`. It can only call other helper functions marked with the same domain (`fn[cpu]` or `fn[gpu]`).

## 5. Example: Image Brightness

This example demonstrates a CPU kernel that adjusts the brightness of an image buffer.

```ark
// 1. The Kernel
// Logic for a SINGLE pixel.
// The compiler handles the 0..N loop.
fn[cpu] brightness(pixel: f32, amount: f32) -> f32 {
    let res = pixel + amount;
    
    // Clamp result to 0.0 - 1.0 range
    if (res > 1.0) { return 1.0; }
    if (res < 0.0) { return 0.0; }
    
    return res;
}

// 2. The Host
fn[host] main() {
    let num_pixels = 1920 * 1080;
    let image_data = allocof<f32>(num_pixels);

    // Initialize image_data with values...

    // Launch:
    // The kernel will run 2,073,600 times in parallel.
    image_data <- brightness(image_data, 0.1) as t_process;

    await t_process;
}

```
