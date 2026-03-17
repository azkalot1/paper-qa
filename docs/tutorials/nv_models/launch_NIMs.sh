# Generate your NVIDIA API key from https://build.nvidia.com/
docker login -u \$oauthtoken -p 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 nvcr.io

export NGC_API_KEY=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
export LOCAL_NIM_CACHE=~/ephemeral/.cache/nim
mkdir -p "$LOCAL_NIM_CACHE"
chmod -R 777 "$LOCAL_NIM_CACHE"

#------- Nemotron-Parse -------------------
# One GPU per NIM
export NIM_PARSE_GPU=0

nohup docker run --rm --name parse \
--gpus "device=${NIM_PARSE_GPU}" \
--shm-size=16GB \
-e NGC_API_KEY \
-v "$LOCAL_NIM_CACHE:/opt/nim/.cache" \
-u 0:0 \
-p 8002:8000 \
nvcr.io/nim/nvidia/nemotron-parse:latest > parse.log 2>&1 &

# Follow the log to check if the NIM is ready
tail -f parse.log

#------- embedding -------------------
export NIM_EMBEDDING_GPU=1

nohup docker run --rm --name embedding \
  --gpus "device=${NIM_EMBEDDING_GPU}" \
  --shm-size=16GB \
  -e NGC_API_KEY \
  -v "$LOCAL_NIM_CACHE:/opt/nim/.cache" \
  -u 0:0 \
  -p 8003:8000 \
  nvcr.io/nim/nvidia/llama-3.2-nv-embedqa-1b-v2:latest > embedding.log 2>&1 &

tail -f embedding.log

#------------- VLM -------------------
export NIM_VLM_GPU=2

nohup docker run --rm --name vlm \
  --gpus "device=${NIM_VLM_GPU}" \
  --shm-size=32GB \
  -e NGC_API_KEY \
  -v "$LOCAL_NIM_CACHE:/opt/nim/.cache" \
  -u 0:0 \
  -p 8004:8000 \
  nvcr.io/nim/nvidia/nemotron-nano-12b-v2-vl:latest > vlm.log 2>&1 &

tail -f vlm.log