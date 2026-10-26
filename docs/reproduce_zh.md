# 复现指南

[English](reproduce.md) | **中文**

本文档说明主结果所需的服务、数据和命令。所有命令都在仓库根目录下运行。

## 1. 环境

建议用三个 Python 环境分别隔离依赖：

| 环境 | 用途 | 安装 |
|---|---|---|
| main | DiscernMem、MAS 宿主、各基准 | [README](../README_zh.md#安装)（Python 3.10，Linux） |
| serving | LLM 与 embedding 服务 | vLLM 0.18.0；见下方命令 |
| retrieval | TriviaQA 和 PopQA 的 Wikipedia 检索 | 按 [Search-R1](https://github.com/PeterGriffinJin/Search-R1) README 中 retriever 环境的说明安装 |

主程序环境的关键验证版本为 Python 3.10、PyTorch 2.10，完整依赖见
`requirements.txt`（包含 KodCode 测试需要的库）。主程序环境的创建命令见
[README](../README_zh.md#安装)。

模型服务使用独立环境；在其中安装已使用的 vLLM 版本：

```bash
conda create -n discernmem-serving python=3.10 pip -y
conda activate discernmem-serving
python -m pip install vllm==0.18.0
```

主程序环境与服务环境分别安装依赖。检索环境按上游 Search-R1 的说明创建；
在服务环境中启动 LLM 和 embedding，在检索环境中启动检索服务，
运行实验时切回主程序环境：`conda activate discernmem`。

KodCode 会在本地子进程中执行生成的 Python 代码和数据集测试。子进程不构成
安全沙箱；请在不含敏感凭据的隔离容器或机器中运行。

## 2. 服务

通过 `CUDA_VISIBLE_DEVICES` 为各服务选择空闲 GPU。显存比例针对所选 GPU，
同卡部署时需计算所有服务的总占用。Search-R1 的 e5 索引文件本身约 60 GiB，
会先读入系统内存；启用 `--faiss_gpu` 后，上游服务使用 float16 将索引分片到
可见 GPU，需与 LLM 分别预留显存。下载分片、拼接索引及解压语料还需要额外磁盘空间。

### LLM（端口 8100）

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

### Embedding（端口 8002）

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

### TriviaQA 和 PopQA 的检索服务（端口 8000）

在 Search-R1 仓库中用 `scripts/download.py` 下载 e5 索引和 2018 年 Wikipedia 语料（见其 README），然后启动服务：

```bash
python search_r1/search/retrieval_server.py \
  --index_path /path/to/e5_Flat.index \
  --corpus_path /path/to/wiki-18.jsonl \
  --topk 3 \
  --retriever_name e5 \
  --retriever_model intfloat/e5-base-v2 \
  --faiss_gpu
```

该服务监听所有网卡且没有鉴权，请用防火墙限制 8000 端口的访问。显存不足时可
去掉 `--faiss_gpu`，让索引留在 CPU（搜索更慢）；上游的查询编码器仍使用 GPU。
启动前检查 `wiki-18.jsonl` 是否为真正的逐行 JSON；部分下载分发需要在解压后
继续提取 tar 包中的语料文件。

### 连通性检查

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

| 变量 | 取值 |
|---|---|
| `LLM_BASE_URL`、`LLM_MODEL`、`LLM_API_KEY` | chat 服务地址、服务端模型名、密钥（vLLM 填 `EMPTY`） |
| `DISCERNMEM_EMBED_BASE_URL`、`DISCERNMEM_EMBED_MODEL`、`DISCERNMEM_EMBED_API_KEY` | embedding 服务地址、模型名、密钥 |
| `SEARCH_URL`（可选） | 检索服务地址，默认 `http://127.0.0.1:8000/retrieve` |

如果模型是从本地路径加载的，vLLM 会用该路径作为模型名，这时 `LLM_MODEL` 和 `DISCERNMEM_EMBED_MODEL` 也要填成对应路径。

## 3. 数据

| 基准 | 来源 | 任务数 | 选取方式 |
|---|---|---:|---|
| TriviaQA | `mandarjoshi/trivia_qa`，`rc.wikipedia.nocontext` | 6,993 | `validation[1000:]`，按原顺序 |
| PopQA | `akariasai/PopQA` | 7,267 | `test[7000:]`，按关系类型分层采样（seed 42） |
| KodCode | `KodCode/KodCode-Light-RL-10K` | 2,000 | 固定种子划分（seed 42）的 test 部分 |
| PDDL | AgentBoard（附带在 `data/pddl/`） | 60 | gripper 20、blockworld 10、barman 20、tyreworld 10 |

Hugging Face 数据集在首次使用时下载。任务按固定顺序处理，每个任务先评分，之后其经验才能更新记忆。

复用已有数据集缓存时，可将 `HF_DATASETS_CACHE` 指向缓存目录。缓存完整后，
还可设置 `HF_HUB_OFFLINE=1` 和 `HF_DATASETS_OFFLINE=1` 进行离线运行。
离线模式不会下载缺失数据；缓存必须包含上表指定的数据集名称和配置。

## 4. 主结果（12 组设置）

```bash
for benchmark in triviaqa popqa kodcode pddl; do
  for mas in autogen camel macnet; do
    python scripts/run.py --benchmark "$benchmark" --mas "$mas"
  done
done
```

每条命令跑完一整条任务流，结果写入 `results/<benchmark>_<mas>_full/`。每次运行使用的设置来自 `configs/benchmarks/<benchmark>.yaml` 的 `protocol` 段：

| | TriviaQA | PopQA | KodCode | PDDL |
|---|---|---|---|---|
| 任务数 | 6,993 | 7,267 | 2,000 | 60 |
| 最大轮数 | 5 | 5 | 1 | 30 |
| 评测方式 | 答案匹配 | 答案匹配 | pytest，每题 30 秒 | AgentBoard PDDLGym，strict 动作检查 |
| 单任务墙钟限制 | 无 | 无 | 无 | 无 |
| 重放 worker / 重放超时 | 8 / 600 秒 | 8 / 600 秒 | 8 / 600 秒 | 8 / 600 秒 |

各次运行互相独立，LLM 服务有余量时可以并行。启动前可以用 `--dry-run` 查看展开后的命令和方法配置指纹。

## 5. 结果与报告

```text
results/<run_id>/
  configs/<benchmark>_<mas>_discernmem.yaml
  <benchmark>/<mas>/discernmem/
    results/eval_results.json      pass_rate：[0, 1] 区间内的平均任务得分
    results/task_results.jsonl     每个任务的得分
    logs/run.log
    memory/discernmem/records/     权威记忆库
    memory/discernmem/reports/     index.md 及每条记忆的报告
```

任务得分的含义：TriviaQA 和 PopQA 为答案正确率，KodCode 为 pytest 是否全部通过，PDDL 为进度率（达成的目标条件比例的最大值）。

为已完成的运行生成记忆统计和报告：

```bash
leaf=results/pddl_autogen_full/pddl/autogen/discernmem
config=results/pddl_autogen_full/configs/pddl_autogen_discernmem.yaml
python scripts/report.py --storage-dir "$leaf/memory/discernmem" --config "$config"
python scripts/materialize_views.py --storage-dir "$leaf/memory/discernmem" --config "$config" --complete
```

## 6. 重跑与其他选项

- 结果目录非空时运行会被拒绝。请换一个 `--run_id`，或传 `-- --overwrite_run` 覆盖。
- `--set KEY=VALUE` 可覆盖 `configs/discernmem/default.yaml` 中的任意方法字段。
- `--` 之后的参数会转给 `scripts/run_benchmarks.py`；完整参数列表见 `python scripts/run_benchmarks.py --help`。
