# Reproduction guide

**English** | [中文](reproduce_zh.md)

This guide covers the services, data, and commands for the main results.
Every command is run from the repository root.

## 1. Environments

Three Python environments keep the dependencies apart:

| Environment | Purpose | Install |
|---|---|---|
| main | DiscernMem, MAS hosts, benchmarks | [README](../README.md#installation) (Python 3.10, Linux) |
| serving | LLM and embedding endpoints | vLLM 0.18.0; commands below |
| retrieval | Wikipedia retrieval for TriviaQA and PopQA | follow the retriever environment in the [Search-R1](https://github.com/PeterGriffinJin/Search-R1) README |

The main environment was validated with Python 3.10 and PyTorch 2.10.
`requirements.txt` specifies its dependencies, including libraries used by
KodCode tests. Create this environment using [the README](../README.md#installation).

Install the vLLM version used for model serving in a separate environment:

```bash
conda create -n discernmem-serving python=3.10 pip -y
conda activate discernmem-serving
python -m pip install vllm==0.18.0
```

Install the main and service dependencies separately. Create the retrieval
environment following Search-R1. Start the LLM and embedding endpoints in the
serving environment and retrieval in its own environment. Switch back to the
main environment to run experiments: `conda activate discernmem`.

KodCode executes generated Python and dataset tests in local subprocesses.
These subprocesses are not a security sandbox; run them in an isolated
container or machine without sensitive credentials.

## 2. Services

Choose idle GPUs for each service using `CUDA_VISIBLE_DEVICES`. GPU memory
fractions apply to the selected GPU; account for all services sharing that GPU.
The Search-R1 e5 index alone is about 60 GiB on disk and is first loaded into
system RAM. With `--faiss_gpu`, the upstream server shards it across the visible
GPUs using float16; reserve sufficient GPU memory separately from the LLM.
Downloading shards, concatenating the index, and unpacking the corpus also
require substantial extra disk space.

### LLM (port 8100)

```bash
vllm serve Qwen/Qwen3-4B-Instruct-2507 \
  --host 127.0.0.1 --port 8100 \
  --max-model-len 131072 \
  --gpu-memory-utilization 0.85 \
  --max-num-seqs 16 \
  --max-num-batched-tokens 16384 \
  --enable-chunked-prefill \
  --enforce-eager
```

### Embeddings (port 8002)

```bash
vllm serve Qwen/Qwen3-Embedding-0.6B \
  --host 127.0.0.1 --port 8002 \
  --max-model-len 16224 \
  --gpu-memory-utilization 0.10 \
  --max-num-seqs 32 \
  --max-num-batched-tokens 16384 \
  --enable-chunked-prefill \
  --enforce-eager
```

### Retrieval for TriviaQA and PopQA (port 8000)

In a clone of Search-R1, download the e5 index and the 2018 Wikipedia corpus
with its `scripts/download.py` (see its README), then start the server:

```bash
python search_r1/search/retrieval_server.py \
  --index_path /path/to/e5_Flat.index \
  --corpus_path /path/to/wiki-18.jsonl \
  --topk 3 \
  --retriever_name e5 \
  --retriever_model intfloat/e5-base-v2 \
  --faiss_gpu
```

The server listens on all interfaces without authentication; keep port 8000
behind a firewall. If GPU memory cannot hold the index, omit `--faiss_gpu` to
keep the index on the CPU (search is slower); the upstream encoder still uses
a GPU. Before launch, check that `wiki-18.jsonl` contains JSON lines: some
distributions require extracting a tar archive after decompression.

### Health checks

```bash
curl -fsS http://127.0.0.1:8100/v1/models
curl -fsS http://127.0.0.1:8002/v1/models
curl -fsS http://127.0.0.1:8000/retrieve \
  -H 'Content-Type: application/json' \
  -d '{"queries":["Who wrote Hamlet?"],"topk":3,"return_scores":true}'
```

### `.env`

```bash
cp .env.example .env
```

| Variable | Value |
|---|---|
| `LLM_BASE_URL`, `LLM_MODEL`, `LLM_API_KEY` | chat endpoint, the served model name, its key (`EMPTY` for vLLM) |
| `DISCERNMEM_EMBED_BASE_URL`, `DISCERNMEM_EMBED_MODEL`, `DISCERNMEM_EMBED_API_KEY` | embedding endpoint, served model name, key |
| `SEARCH_URL` (optional) | retrieval endpoint; default `http://127.0.0.1:8000/retrieve` |

If the models are served from local paths, set `LLM_MODEL` and
`DISCERNMEM_EMBED_MODEL` to those paths, since vLLM names a model after the
path it was loaded from.

## 3. Data

| Benchmark | Source | Tasks | Selection |
|---|---|---:|---|
| TriviaQA | `mandarjoshi/trivia_qa`, `rc.wikipedia.nocontext` | 6,993 | `validation[1000:]`, in order |
| PopQA | `akariasai/PopQA` | 7,267 | `test[7000:]`, stratified by relation type (seed 42) |
| KodCode | `KodCode/KodCode-Light-RL-10K` | 2,000 | seeded split (seed 42), test portion |
| PDDL | AgentBoard (bundled in `data/pddl/`) | 60 | gripper 20, blockworld 10, barman 20, tyreworld 10 |

The Hugging Face datasets download on first use. Tasks are processed in a
fixed order; each task is scored before its experience can update memory.

To reuse an existing dataset cache, set `HF_DATASETS_CACHE` to its directory.
For fully cached runs, also set `HF_HUB_OFFLINE=1` and `HF_DATASETS_OFFLINE=1`.
Offline mode does not download missing data: the cache must already contain
the dataset names and configurations listed above.

## 4. Main results (12 settings)

```bash
for benchmark in triviaqa popqa kodcode pddl; do
  for mas in autogen camel macnet; do
    python scripts/run.py --benchmark "$benchmark" --mas "$mas"
  done
done
```

Each command runs one complete task stream and creates
`results/<benchmark>_<mas>_full/`. The settings applied to every run come from
the `protocol` section of `configs/benchmarks/<benchmark>.yaml`:

| | TriviaQA | PopQA | KodCode | PDDL |
|---|---|---|---|---|
| tasks | 6,993 | 7,267 | 2,000 | 60 |
| max turns | 5 | 5 | 1 | 30 |
| evaluator | answer match | answer match | pytest, 30 s per task | AgentBoard PDDLGym, strict action check |
| task wall-clock limit | none | none | none | none |
| replay workers / replay timeout | 8 / 600 s | 8 / 600 s | 8 / 600 s | 8 / 600 s |

Runs are independent and may execute concurrently when the LLM endpoint has
capacity. Use `--dry-run` to print the expanded command and the
method-config fingerprint before launching.

## 5. Results and reports

```text
results/<run_id>/
  configs/<benchmark>_<mas>_discernmem.yaml
  <benchmark>/<mas>/discernmem/
    results/eval_results.json      pass_rate: mean task score in [0, 1]
    results/task_results.jsonl     per-task scores
    logs/run.log
    memory/discernmem/records/     authoritative memory store
    memory/discernmem/reports/     index.md and per-memory reports
```

The task score is answer accuracy for TriviaQA and PopQA, the pytest pass
indicator for KodCode, and the progress rate (highest fraction of goal
conditions satisfied) for PDDL.

Memory statistics and reports for a finished run:

```bash
leaf=results/pddl_autogen_full/pddl/autogen/discernmem
config=results/pddl_autogen_full/configs/pddl_autogen_discernmem.yaml
python scripts/report.py --storage-dir "$leaf/memory/discernmem" --config "$config"
python scripts/materialize_views.py --storage-dir "$leaf/memory/discernmem" --config "$config" --complete
```

## 6. Re-running and options

- A run refuses to write into a non-empty result directory. Choose a new
  `--run_id`, or pass `-- --overwrite_run` to replace it.
- `--set KEY=VALUE` overrides any method field in
  `configs/discernmem/default.yaml`.
- Anything after `--` is forwarded to `scripts/run_benchmarks.py`; run
  `python scripts/run_benchmarks.py --help` for the full list.
