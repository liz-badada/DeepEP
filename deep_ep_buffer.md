## deep_ep/buffer.py
```mermaid
sequenceDiagram
    participant User
    participant Buffer
    participant deep_ep_cpp
    participant CUDA
    participant NVSHMEM

    %% Initialization
    User->>Buffer: __init__(group, num_nvl_bytes, num_rdma_bytes, ...)
    Buffer->>deep_ep_cpp: Buffer(rank, group_size, ...)
    Buffer->>NVSHMEM: Initialize NVSHMEM (if needed)
    Buffer->>deep_ep_cpp: sync(device_ids, ipc_handles, root_unique_id)

    %% Dispatch Flow
    rect rgb(200, 220, 255)
        Note over User,Buffer: Normal Dispatch Path
        User->>Buffer: dispatch(x, handle, ...)
        alt Internode Communication
            Buffer->>deep_ep_cpp: internode_dispatch(...)
            deep_ep_cpp->>CUDA: Execute CUDA Kernels
            deep_ep_cpp->>NVSHMEM: RDMA Communication
        else Intranode Communication
            Buffer->>deep_ep_cpp: intranode_dispatch(...)
            deep_ep_cpp->>CUDA: Execute CUDA Kernels
        end
    end

    %% Combine Flow
    rect rgb(220, 200, 255)
        Note over User,Buffer: Normal Combine Path
        User->>Buffer: combine(x, handle, ...)
        alt Internode Communication
            Buffer->>deep_ep_cpp: internode_combine(...)
            deep_ep_cpp->>CUDA: Execute CUDA Kernels
            deep_ep_cpp->>NVSHMEM: RDMA Communication
        else Intranode Communication
            Buffer->>deep_ep_cpp: intranode_combine(...)
            deep_ep_cpp->>CUDA: Execute CUDA Kernels
        end
    end

    %% Low Latency Path
    rect rgb(255, 220, 200)
        Note over User,Buffer: Low Latency Path
        User->>Buffer: clean_low_latency_buffer(...)
        Buffer->>deep_ep_cpp: clean_low_latency_buffer(...)
        
        User->>Buffer: low_latency_dispatch(x, topk_idx, ...)
        Buffer->>deep_ep_cpp: low_latency_dispatch(...)
        deep_ep_cpp->>NVSHMEM: IBGDA Communication
        
        User->>Buffer: low_latency_combine(x, topk_idx, ...)
        Buffer->>deep_ep_cpp: low_latency_combine(...)
        deep_ep_cpp->>NVSHMEM: IBGDA Communication
    end

    %% Layout Calculation
    rect rgb(200, 255, 220)
        Note over User,Buffer: Layout Calculation
        User->>Buffer: get_dispatch_layout(topk_idx, ...)
        Buffer->>deep_ep_cpp: get_dispatch_layout(...)
        deep_ep_cpp-->>Buffer: Return layout information
        Buffer-->>User: Return layout tensors and event
    end
```

## csrc/config.hpp
```mermaid
sequenceDiagram
    participant Client
    participant Config
    participant LowLatencyLayout
    participant LowLatencyBuffer

    Client->>Config: 创建配置(num_sms, tokens...)
    Config-->>Client: 返回配置实例

    Client->>Config: get_nvl_buffer_size_hint(hidden_bytes, num_ranks)
    Config-->>Client: 返回NVL缓冲区大小

    Client->>Config: get_rdma_buffer_size_hint(hidden_bytes, num_ranks)
    Config-->>Client: 返回RDMA缓冲区大小

    Client->>LowLatencyLayout: 创建布局(rdma_buffer, tokens, hidden, ranks, experts)
    LowLatencyLayout->>LowLatencyBuffer: 初始化两个缓冲区(odd/even)
    LowLatencyLayout-->>Client: 返回布局实例

    Client->>LowLatencyBuffer: clean_meta()
    LowLatencyBuffer-->>Client: 返回清理元数据
```