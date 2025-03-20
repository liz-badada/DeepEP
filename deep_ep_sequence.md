# DeepEP

## code
```sh
|-- csrc
|   |-- CMakeLists.txt
|   |-- config.hpp
|   |-- deep_ep.cpp
|   |-- deep_ep.hpp
|   |-- event.hpp
|   `-- kernels
|       |-- CMakeLists.txt
|       |-- api.cuh
|       |-- buffer.cuh
|       |-- configs.cuh
|       |-- exception.cuh
|       |-- ibgda_device.cuh
|       |-- internode.cu
|       |-- internode_ll.cu
|       |-- intranode.cu
|       |-- launch.cuh
|       |-- runtime.cu
|       `-- utils.cuh
|-- deep_ep
|   |-- __init__.py
|   |-- __pycache__
|   |   |-- __init__.cpython-310.pyc
|   |   |-- buffer.cpython-310.pyc
|   |   `-- utils.cpython-310.pyc
|   |-- buffer.py
|   `-- utils.py
|-- tests
|   |-- __pycache__
|   |   |-- test_low_latency.cpython-38.pyc
|   |   `-- utils.cpython-38.pyc
|   |-- test_internode.py
|   |-- test_intranode.py
|   |-- test_low_latency.py
|   `-- utils.py
`-- third-party
    |-- README.md
    `-- nvshmem.patch
```

## start from python api
```mermaid
sequenceDiagram
    participant Main
    participant TestLoop
    participant TestMain
    participant Buffer
    participant Dispatch
    participant Combine
    participant Benchmark

    Main->>TestLoop: spawn multiple processes
    Note over TestLoop: Initialize distributed setup
    
    TestLoop->>Buffer: Create buffer with RDMA size
    TestLoop->>TestMain: Run test with parameters
    
    rect rgb(200, 200, 255)
        Note over TestMain: Test Dispatch & Combine
        TestMain->>Buffer: low_latency_dispatch
        Buffer-->>Dispatch: Process data
        Note over Dispatch: Handle topk operations
        Dispatch-->>TestMain: Return packed data
        
        TestMain->>Buffer: low_latency_combine
        Buffer-->>Combine: Process data
        Note over Combine: Combine results
        Combine-->>TestMain: Return combined data
    end
    
    rect rgb(200, 255, 200)
        Note over TestMain: Benchmark Operations
        TestMain->>Benchmark: Measure dispatch bandwidth
        TestMain->>Benchmark: Measure combine bandwidth
        Benchmark-->>TestMain: Return timing results
    end
    
    TestMain-->>TestLoop: Return hash value
    
    loop Optional Pressure Test
        TestLoop->>TestMain: Repeat test with different seeds
        TestMain-->>TestLoop: Verify hash consistency
    end
    
    TestLoop-->>Main: Complete test
```