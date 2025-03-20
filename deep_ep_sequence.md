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
    participant Python
    participant FlashAPTPython as flash_mla_inference.py
    participant FlashAPI as flash_api.cpp
    participant FlashFwdMLAMetaData as flash_fwd_mla_metadata.cu
    participant FlashFwdMLABF16SM90 as flash_fwd_mla_bf16_sm90.cu

    Python ->> FlashAPTPython: get_mla_metadata()
    FlashAPTPython ->> FlashAPI: get_mla_metadata()
    FlashAPI ->> FlashFwdMLAMetaData: get_mla_metadata_func()
    FlashFwdMLAMetaData ->> FlashFwdMLAMetaData: get_mla_metadata_kernel()
    FlashFwdMLAMetaData -->> FlashFwdMLAMetaData: return Mla_metadata_params
    FlashFwdMLAMetaData -->> FlashAPI: {tile_scheduler_metadata, num_splits}
    FlashAPI -->> Python: {tile_scheduler_metadata, num_splits}

    Python ->> FlashAPTPython: flash_mla_with_kvcache()
    FlashAPTPython ->> FlashAPI: mha_fwd_kvcache_mla()
    FlashAPI ->> FlashAPI: tensor checks and reshaping
    FlashAPI ->> FlashFwdMLABF16SM90: run_mha_fwd_splitkv_mla()
    FlashFwdMLABF16SM90 ->> FlashFwdMLABF16SM90: run_flash_splitkv_fwd_mla()
    FlashFwdMLABF16SM90 ->> FlashFwdMLABF16SM90: flash_fwd_splitkv_mla_kernel()
    FlashFwdMLABF16SM90 ->> FlashFwdMLABF16SM90: flash_fwd_splitkv_mla_combine_kernel()
    FlashFwdMLABF16SM90 -->> FlashAPI: {attention outputs, softmax logsumexp}
    FlashAPI -->> Python: {attention outputs, softmax logsumexp}
```