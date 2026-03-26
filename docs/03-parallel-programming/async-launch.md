# Async Launch

ArkLang provides a dedicated syntax for dispatching kernels to a device (CPU/GPU) without blocking the host thread.

## 1. Syntax

The launch statement consists of three parts: the **Destination Grid**, the **Kernel Call**, and the **Sync Token**.

```ark
Destination <- KernelName(Arguments...) as Token;

```

* **Destination (`<-`):** The array or buffer that defines the iteration space (Grid) and receives the results.
* **Kernel Call:** The function invocation. The kernel must be defined with `fn[cpu]` or `fn[gpu]`.
* **Token (`as`):** A handle used to synchronize execution later.

## 2. Execution Flow

1. **Dispatch:** The host schedules the kernel for execution on the target domain.
2. **Continue:** The launch statement returns immediately. The host proceeds to the next line of code.
3. **Parallel Work:** The kernel runs in the background (on worker threads or GPU).
4. **Synchronization:** The host blocks and waits for completion using `await`.

## 3. Synchronization

The `await` keyword ensures memory consistency. Accessing the destination buffer before awaiting the token results in undefined behavior (race conditions).

```ark
// Block until 't1' is finished
await t1;

```

## 4. Example

```ark
fn[host] main() {
    let data = allocof<f32>(1024);

    // Launch: 'data' is the grid (0..1024)
    // 'process_kernel' runs on [cpu]
    data <- process_kernel(data, 2.0) as t1;

    print "Kernel is running in background...";

    // Wait for completion
    await t1;
    
    // Safe to read 'data' now
    print data[0];
}

fn[cpu] process_kernel(val: f32, scale: f32) -> f32 {
    return val * scale;
}

```

