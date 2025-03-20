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

## test_internode.py
```mermaid
sequenceDiagram
    participant Main
    participant TestLoop
    participant TestMain
    participant Buffer
    participant DistributedGroup

    Main->>TestLoop: spawn multiple processes
    Note over TestLoop: Initialize with local_rank & num_local_ranks

    TestLoop->>DistributedGroup: init_dist()
    TestLoop->>Buffer: create Buffer instance

    TestLoop->>TestMain: test_main()
    
    Note over TestMain: Initialize test parameters
    TestMain->>TestMain: Generate random test data
    
    rect rgb(200, 200, 255)
        Note over TestMain: Test Dispatch Phase
        TestMain->>Buffer: get_dispatch_layout
        Buffer-->>TestMain: return layout info
        
        loop For different modes
            Note over TestMain: Test with different configurations
            TestMain->>Buffer: dispatch()
            Buffer-->>TestMain: receive dispatched data
            
            alt if not with_topk
                TestMain->>Buffer: cached dispatch
                Buffer-->>TestMain: receive cached data
            end
            
            TestMain->>Buffer: combine()
            Buffer-->>TestMain: return combined data
            TestMain->>TestMain: verify results
        end
    end
    
    rect rgb(200, 255, 200)
        Note over TestMain: Performance Tuning Phase
        loop For different configurations
            TestMain->>Buffer: tune dispatch performance
            Buffer-->>TestMain: performance metrics
        end
        
        loop For different configurations
            TestMain->>Buffer: tune combine performance
            Buffer-->>TestMain: performance metrics
        end
    end

    opt If test_ll_compatibility
        TestMain->>Buffer: clean_low_latency_buffer
        TestMain->>TestMain: test_low_latency
    end
```

## test_intranode.py
```mermaid
sequenceDiagram
    participant Main
    participant TestLoop
    participant TestMain
    participant Buffer
    participant Dispatch
    participant Combine

    Main->>TestLoop: spawn multiple processes
    TestLoop->>TestLoop: init_dist()
    TestLoop->>Buffer: create Buffer
    TestLoop->>TestMain: test_main()
    
    Note over TestMain: Initialize test data:<br/>- Generate random tensors<br/>- Calculate expert meta<br/>- Calculate rank layout

    TestMain->>Buffer: get_dispatch_layout()
    
    loop Test Different Configurations
        Note over TestMain: Test combinations of:<br/>- previous_mode<br/>- async_mode<br/>- data types (FP8/BF16)<br/>- with/without top-k
        
        TestMain->>Dispatch: buffer.dispatch()
        Note over Dispatch: Process dispatch operation
        Dispatch-->>TestMain: return recv_x, indices, weights
        
        alt without top-k
            TestMain->>Dispatch: cached dispatch
        end
        
        TestMain->>Combine: buffer.combine()
        Note over Combine: Process combine operation
        Combine-->>TestMain: return combined results
        
        TestMain->>TestMain: verify results
    end
    
    Note over TestMain: Performance Tuning
    loop Tune Dispatch
        TestMain->>Dispatch: test different configurations
        Note over TestMain: Find best SM and chunk size
    end
    
    loop Tune Combine
        TestMain->>Combine: test different configurations
        Note over TestMain: Find best SM and chunk size
    end
```

## test_low_latency.py
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