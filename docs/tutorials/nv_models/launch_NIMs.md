# Launching NIM Containers

## Prerequisites

1. **NVIDIA API key** — generate one at <https://build.nvidia.com/>.
2. **Docker with NVIDIA runtime** installed (`nvidia-container-toolkit`).
3. Export your key so the containers can access it:

```bash
export NGC_API_KEY="<your-api-key>"
```

4. Log in to the NGC registry:

```bash
docker login -u '$oauthtoken' -p "$NGC_API_KEY" nvcr.io
```

## First-time setup: move Docker + containerd storage to ephemeral disk

Cloud instances often have a small root filesystem (`/`) that fills up quickly
when pulling large NIM images. Run this **once** to move Docker and containerd
storage to `/ephemeral` (typically a large attached volume):

```bash
./launch_nim.sh setup-docker
```

This will:
1. Bind-mount `/tmp` to ephemeral storage (prevents TensorRT engine builds from filling `/`)
2. Move containerd root to `/ephemeral/containerd`
3. Move Docker data-root to `/ephemeral/docker-data`
4. Clean up old `/var/lib/docker` and `/var/lib/containerd` to reclaim space on `/`
5. Restart both daemons

To use a custom path:

```bash
./launch_nim.sh --docker-root /mnt/data/docker setup-docker
```

> **Note:** Restarting Docker stops all running containers. Images will need to
> be re-pulled to the new location.

## Helper script — `launch_nim.sh`

A reusable script that wraps `docker run` for any NIM container.

```
./launch_nim.sh [setup-docker] --name NAME --gpus GPUS --port PORT [OPTIONS] IMAGE
```

| Flag | Description | Default |
|------|-------------|---------|
| `-n, --name` | Container name | *(required)* |
| `-g, --gpus` | GPU device(s): `0`, `"1,2"`, or `"all"` | *(required)* |
| `-p, --port` | Host port mapped to container port 8000 | *(required)* |
| `-l, --log` | Log file path | `<name>.log` |
| `-s, --shm` | Shared memory size | `16GB` |
| `-c, --cache` | Local NIM model cache directory | `~/ephemeral/.cache/nim` |
| `-d, --docker-root` | Docker data-root for `setup-docker` | `/ephemeral/docker-data` |

## Launching the NIMs for PaperQA

### 1. Nemotron-Parse (document parsing)

```bash
./launch_nim.sh \
    --name parse \
    --gpus 0 \
    --port 8002 \
    nvcr.io/nim/nvidia/nemotron-parse:latest
```

#### Multiple parse instances (round-robin load balancing)

Nemotron-Parse is a single-GPU model — passing multiple GPUs to one container
won't help. To scale parsing throughput, launch **one instance per GPU** on
different ports:

```bash
./launch_nim.sh --name parse0 --gpus 0 --port 8002 nvcr.io/nim/nvidia/nemotron-parse:latest
./launch_nim.sh --name parse1 --gpus 1 --port 8003 nvcr.io/nim/nvidia/nemotron-parse:latest
./launch_nim.sh --name parse2 --gpus 2 --port 8004 nvcr.io/nim/nvidia/nemotron-parse:latest
./launch_nim.sh --name parse3 --gpus 3 --port 8005 nvcr.io/nim/nvidia/nemotron-parse:latest
```

Then pass all endpoints as a comma-separated list. The `nim_runner.py`
round-robin will distribute per-page requests across them:

```bash
PQA_PARSE_API_BASE=http://localhost:8002/v1,http://localhost:8003/v1,http://localhost:8004/v1,http://localhost:8005/v1 \
PQA_INDEX_CONCURRENCY=4 \
...
```

> **Tip:** Bump `PQA_INDEX_CONCURRENCY` to match the number of parse instances
> so multiple PDFs are parsed in parallel, fully utilizing all instances.

### 2. Embedding model

```bash
./launch_nim.sh \
    --name embedding \
    --gpus 1 \
    --port 8003 \
    nvcr.io/nim/nvidia/llama-3.2-nv-embedqa-1b-v2:latest
```

### 3. VLM (vision-language model)

The VLM is larger and benefits from more shared memory. To use multiple GPUs,
pass a comma-separated list:

```bash
./launch_nim.sh \
    --name vlm \
    --gpus "1,2,3" \
    --port 8004 \
    --shm 32GB \
    nvcr.io/nim/nvidia/nemotron-nano-12b-v2-vl:latest
```

For a single GPU, just pass the device index:

```bash
./launch_nim.sh \
    --name vlm \
    --gpus 2 \
    --port 8004 \
    --shm 32GB \
    nvcr.io/nim/nvidia/nemotron-nano-12b-v2-vl:latest
```

> **Tip:** The embedding model is small enough to share a GPU with the VLM.
> For example, on a 4×H100 machine: parse on GPU 0, embedding + VLM on GPUs 1-3:
>
> ```bash
> ./launch_nim.sh --name embedding --gpus 1 --port 8003 nvcr.io/nim/nvidia/llama-3.2-nv-embedqa-1b-v2:latest
> ./launch_nim.sh --name vlm --gpus "1,2,3" --port 8004 --shm 32GB nvcr.io/nim/nvidia/nemotron-nano-12b-v2-vl:latest
> ```

## Verifying containers are ready

Each container logs to `<name>.log` by default. Check if a NIM is ready:

```bash
tail -f parse.log       # Ctrl+C to stop tailing
tail -f embedding.log
tail -f vlm.log
```

You can also verify the health endpoint once the container reports ready:

```bash
curl -s http://localhost:8002/v1/health/ready   # parse
curl -s http://localhost:8003/v1/health/ready   # embedding
curl -s http://localhost:8004/v1/health/ready   # vlm
```

## Stopping containers

```bash
docker stop parse embedding vlm
```

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `No space left on device` during pull | Root disk full | Run `./launch_nim.sh setup-docker` to move storage to ephemeral |
| `No space left on device` during TRT engine build | Container `/tmp` on root disk | Already fixed — script mounts ephemeral tmp into each container |
| `stat /tmp: no such file or directory` | `/tmp` missing on host | Run `./launch_nim.sh setup-docker` (binds `/tmp` to ephemeral) |
| `cannot set both Count and DeviceIDs` | Docker `--gpus` quoting bug | Already fixed in script |
| `Free memory on device ... less than desired` | Other NIM already on that GPU | Adjust `--gpus` to use free GPUs (check `nvidia-smi`) |
| `failed to lease content ... blob not found` | Deleted containerd data without restart | `sudo systemctl restart containerd docker` |
