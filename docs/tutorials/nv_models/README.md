# Use NVIDIA Open Models for PaperQA
- [nvidia/nemotron-parse](https://build.nvidia.com/nvidia/nemotron-parse) -> PDF parser
- [nvidia/llama-3.2-nv-embedqa-1b-v2](https://build.nvidia.com/nvidia/llama-3_2-nv-embedqa-1b-v2) -> embedding
- [nvidia/nemotron-nano-12b-v2-vl](https://build.nvidia.com/nvidia/nemotron-3-nano-30b-a3b) -> enrichment_llm, summary_llm, llm, agent_llm

*The right-side terms match paper-qa `Settings` / `ParsingSettings` field names.*

You can try the models above interactively for free at [build.nvidia.com](https://build.nvidia.com) (click the model name links above).

## Why use them?
- **Open** models. 
- The [NVIDIA Developer Program](https://developer.nvidia.com/developer-program) is free to join and gives you free access to [NVIDIA NIM](https://developer.nvidia.com/nim?sortBy=developer_learning_library%2Fsort%2Ffeatured_in.nim%3Adesc%2Ctitle%3Aasc) for research, development, and testing.

## How to run
I tested the steps below on 4×-A100/H100 instances on [NVIDIA Brev](https://docs.nvidia.com/brev/latest/).

### 1. Launch NVIDIA NIMs
Follow [launch_NIMs.sh](./launch_NIMs.sh) to run all NIMs with Docker. Then run [test_APIs.ipynb](./test_APIs.ipynb) to confirm the locally-hosted NIM API endpoints are working.

### 2. Install PaperQA
Follow [install_PQA.sh](./install_PQA.sh) to install in editable mode and create a Jupyter kernel for the next step.

### 3. Run PaperQA with locally-hosted NIMs
Run [test_PQA_singlePDF.ipynb](./test_PQA_singlePDF.ipynb) and [test_PQA_multiPDFs.ipynb](./test_PQA_multiPDFs.ipynb) with the Jupyter kernel you created in the previous step. In each notebook, you’ll configure `Settings` for the locally-hosted NIM API endpoints and verify that several PaperQA methods work with them.

### Caveats
- Work in progress: integrating nemotron-nano-3 as the main llm and agent_llm. Stay tuned.
- I originally tried using NVIDIA-hosted API endpoints on build.nvidia.com, but some NIMs have long wait queues due to popularity, so self-hosting is recommended 😎.
