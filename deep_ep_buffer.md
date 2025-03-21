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

    Client->>Config: Create Config(num_sms, tokens...)
    Config-->>Client: Return config instance

    Client->>Config: get_nvl_buffer_size_hint(hidden_bytes, num_ranks)
    Config-->>Client: Return NVL buffer size

    Client->>Config: get_rdma_buffer_size_hint(hidden_bytes, num_ranks)
    Config-->>Client: Return RDMA buffer size

    Client->>LowLatencyLayout: Create Layout(rdma_buffer, tokens, hidden, ranks, experts)
    LowLatencyLayout->>LowLatencyBuffer: Initialize two buffers(odd/even)
    LowLatencyLayout-->>Client: Return layout instance

    Client->>LowLatencyBuffer: clean_meta()
    LowLatencyBuffer-->>Client: Return cleaned metadata
```

### NVL Buffer Size (when num_ranks > 0, align to 128 bytes)
[get_nvl_buffer_size_hint](https://github.com/liz-badada/DeepEP/blob/deepep_study/csrc/config.hpp#L45-L65)
<!-- ```math
\begin{aligned}
\text{NVL\_Buffer\_Size} = \frac{((C \times R_{nvl} \times S_{total}) + 127 ) \times 128}{128}
\end{aligned}
```
where:
```math
\begin{aligned}
& C = \text{num\_channels} = \frac{\text{num\_sms}}{2} \\
& R_{nvl} = \min(\text{num\_ranks}, \text{NUM\_MAX\_NVL\_PEERS}) \\
& S_{total} = (2R_{rdma} + 3) \times \text{sizeof(int)} + T_{recv} \times (S_{data} + S_{meta} + S_{topk} + S_{scale}) \\
& R_{rdma} = \max(\frac{\text{num\_ranks}}{\text{NUM\_MAX\_NVL\_PEERS}}, 1) \\
& T_{recv} = \text{num\_max\_nvl\_chunked\_recv\_tokens} \\
& S_{data} = \text{hidden\_bytes} \\
& S_{meta} = \text{source\_meta\_bytes} \\
& S_{topk} = 128 \times (\text{sizeof(int64\_t)} + \text{sizeof(float)}) \\
& S_{scale} = 128 \times \text{sizeof(float)}
\end{aligned}
``` -->
![nvl_buffer_size](./figures/nvl_buffer_size.png)

### RDMA Buffer Size (when num_ranks ≤ NUM_MAX_NVL_PEERS, align to 128 bytes)
[get_rdma_buffer_size_hint](https://github.com/liz-badada/DeepEP/blob/deepep_study/csrc/config.hpp#L67-L91)
<!-- ```math
\begin{aligned}
& \text{RDMA\_Buffer\_Size} = \frac{((C \times R_{rdma} \times 2S_{total}) + 127 ) \times 128}{128}
\end{aligned}
```
where:
```math
\begin{aligned}
& C = \text{num\_channels} = \frac{\text{num\_sms}}{2} \\
& R_{rdma} = \frac{\text{num\_ranks}}{\text{NUM\_MAX\_NVL\_PEERS}} \\
& S_{total} = (2N_{nvl} + 2) \times \text{sizeof(int)} + \quad T_{recv} \times (S_{data} + S_{meta} + S_{topk} + S_{scale} + S_{int4}) \\
& N_{nvl} = \text{NUM\_MAX\_NVL\_PEERS} \\
& T_{recv} = \text{num\_max\_rdma\_chunked\_recv\_tokens} \\
& S_{data} = \text{hidden\_bytes} \\
& S_{meta} = \text{source\_meta\_bytes} \\
& S_{topk} = 128 \times (\text{sizeof(int64\_t)} + \text{sizeof(float)}) \\
& S_{scale} = 128 \times \text{sizeof(float)} \\
& S_{int4} = \text{sizeof(int4)}
\end{aligned}
``` -->
![rdma_buffer_size](./figures/rdma_buffer_size.png)

### Notes
    - All calculation results are aligned to 128 bytes
    - RDMA buffer size includes bidirectional communication ($\times 2$)
    - Both buffers contain space for control information, data, metadata, TopK, and scale factors