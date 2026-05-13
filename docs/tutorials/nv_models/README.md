# Use NVIDIA Open Models for PaperQA
- [nvidia/nemotron-parse](https://build.nvidia.com/nvidia/nemotron-parse) -> PDF parser
- [nvidia/llama-3.2-nv-embedqa-1b-v2](https://build.nvidia.com/nvidia/llama-3_2-nv-embedqa-1b-v2) -> embedding (can also be used from unference endpoint)
- [nvidia/nemotron-nano-12b-v2-vl](https://build.nvidia.com/nvidia/nemotron-3-nano-30b-a3b) -> enrichment_llm, summary_llm, optionally: llm, agent_llm

*The right-side terms match paper-qa `Settings` / `ParsingSettings` field names.*

You can try the models above interactively for free at [build.nvidia.com](https://build.nvidia.com) (click the model name links above).

## Why use them?
- **Open** models. 
- The [NVIDIA Developer Program](https://developer.nvidia.com/developer-program) is free to join and gives you free access to [NVIDIA NIM](https://developer.nvidia.com/nim?sortBy=developer_learning_library%2Fsort%2Ffeatured_in.nim%3Adesc%2Ctitle%3Aasc) for research, development, and testing.

## How to run
I tested the steps below on 4x-A100/H100 instances on [NVIDIA Brev](https://docs.nvidia.com/brev/latest/).

### 1. Launch NVIDIA NIMs
Follow [launch_NIMs.md](./launch_NIMs.md) (script: [`launch_nim.sh`](./launch_nim.sh)) to run all NIMs with Docker. Then run [`test_APIs.py`](./test_APIs.py) to confirm the locally-hosted NIM API endpoints are working.

Example with self-hosted parse + VLM and NVIDIA inference API for embeddings:

```bash
NVIDIA_INFERENCE_KEY=... python test_APIs.py \
    --mode all \
    --parse-base-url http://localhost:8002/v1 \
    --vlm-base-url http://localhost:8004/v1 \
    --embedding-base-url https://inference-api.nvidia.com/v1 \
    --vlm-model nvidia/nemotron-nano-12b-v2-vl
```

### 2. Install PaperQA
Follow [install_PQA.md](./install_PQA.md) (script: [`install_pqa.sh`](./install_pqa.sh)) to install in editable mode.

### 3. Run PaperQA with locally-hosted NIMs
Run [`test_PQA.py`](./test_PQA.py) to configure `Settings` for the locally-hosted NIM API endpoints and verify that PaperQA methods work with them.

Example with self-hosted parse + VLM, inference API for embeddings, and a
specific model for the main LLM and agent LLM:

```bash
NVIDIA_INFERENCE_KEY=... python test_PQA.py \
    --parse-base-url http://localhost:8002/v1 \
    --embedding-base-url https://inference-api.nvidia.com/v1 \
    --vlm-base-url http://localhost:8004/v1 \
    --vlm-model nvidia/nemotron-nano-12b-v2-vl \
    --llm-model nvidia/nvidia/nemotron-3-super-v3 \
    --llm-base-url https://inference-api.nvidia.com/v1 \
    --agent-llm-model nvidia/nvidia/nemotron-3-super-v3 \
    --agent-llm-base-url https://inference-api.nvidia.com/v1 \
    --trace
```

### Caveats
- Work in progress: integrating nemotron-nano-3 as the main llm and agent_llm. Stay tuned.
- I originally tried using NVIDIA-hosted API endpoints on build.nvidia.com, but some NIMs have long wait queues due to popularity, so self-hosting is recommended.
