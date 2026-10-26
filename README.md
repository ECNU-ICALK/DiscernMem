# DiscernMem

**English** | [中文](README_zh.md)

Code for *Remembering Is Not Enough: Refining Memories and Learning When to Use
Them in Multi-Agent Systems*.

![Overview of DiscernMem](docs/figures/overview.svg)

*Overview of DiscernMem. (a) Memory use in one task cycle: retrieval, team
execution and scoring, and new experience. (b) DMR: paired replay, scope and
defect diagnosis, and validated repair. (c) BLM: applicability learning,
lifecycle management, and role-specific retrieval. Updates affect only
subsequent tasks.*

DiscernMem is a training-free memory framework for LLM multi-agent systems. Each
memory is a behavioral rule owned by one role. After every scored task,
DiscernMem decides whether a harmful reuse reflects a defect in the rule or a
rule applied in the wrong context, and responds accordingly:

- **Defect-Aware Memory Refinement (DMR)** replays the task with only one memory
  supplied to its owner and with no memory at all, diagnoses content defects
  from the trajectories, and accepts a repair only if it does not regress on the
  triggering task and on earlier tasks.
- **Boundary-Learning Meta-Memory (BLM)** records each memory's uses and
  measured effects and learns, from the owner's pre-action context, when the
  role should apply or avoid it.
- New experience is extracted per role, matched against existing memories by
  identity, and calibrated on similar earlier tasks. Memories move through
  provisional, active, and inactive states.

Every task is scored before its experience can update memory; updates affect
later tasks only.

Supported multi-agent systems: **AutoGen**, **CAMEL**, **MacNet**.
Supported benchmarks: **TriviaQA**, **PopQA**, **KodCode**, and **PDDL**
(AgentBoard).

## Repository layout

```text
discernmem/           the method
  centralized_adapter.py   interface used by the MAS hosts (per-role retrieval, task completion)
  controller.py            post-task loop: replay, diagnosis, repair, meta-memory, admission
  replay.py scheduling.py  paired replays (single memory vs. no memory)
  usage_assessor.py counterexamples.py   applicability check and defect attribution
  repair_generator.py cpe.py validation.py   repair candidates and cross-task validation
  meta_memory.py meta_summarizer.py effect.py   use/avoid conditions and effect records
  pipeline.py admission_controller.py identity_judge.py semantic_index.py   extraction and admission
  retrieval.py retrieval_selector.py   role-filtered retrieval and selection
  state_engine.py          lifecycle (provisional / active / inactive)
  storage.py               JSON memory store and human-readable reports
  config.py                method configuration
  prompt/                  method prompts and JSON schemas
mas/                  AutoGen, CAMEL, and MacNet hosts, agent and message graph
  <mas>/prompts/prompt.py           also holds this MAS's guidance for memory methods (MEMORY_*)
data/                 benchmark environments, data loaders, and interaction loops
  pddl/agentboard_pddlgym/ vendored AgentBoard PDDLGym (see Third-party code and data)
common/               LLM/embedding clients, QA retrieval settings, launcher, statistics
  memory_guidance.py   task guidance per benchmark and lookup of MAS/task guidance for memory methods
configs/
  benchmarks/          per-benchmark runner settings (`protocol` section)
  discernmem/          method defaults, ablation variants, sweep definitions
scripts/
  run.py               run one experiment
  run_sweep.py         run the ablation group or the hyper-parameter sweeps
  run_benchmarks.py    low-level runner (all options)
  report.py materialize_views.py   metrics and reports from a finished memory store
docs/reproduce.md     full reproduction guide
docs/figures/         overview figure shown above
```

## Installation

Use Linux and Python 3.10. Create the main environment with conda:

```bash
conda create -n discernmem python=3.10 pip -y
conda activate discernmem
python -m pip install -r requirements.txt
python -m pip check
```

The main environment was validated with Python 3.10 and PyTorch 2.10; dependency
versions are specified in `requirements.txt`. Run all commands from the
repository root. No package installation is needed. Model serving and retrieval
use separate environments, described in [the reproduction guide](docs/reproduce.md#1-environments).

## Services

DiscernMem talks to OpenAI-compatible endpoints:

| Service | Model | Used by |
|---|---|---|
| Chat completions | Qwen3-4B-Instruct-2507 | all agents and all method LLM calls |
| Embeddings | Qwen3-Embedding-0.6B | memory retrieval, identity matching, validation-task selection |
| Retrieval (`POST /retrieve`) | Search-R1 e5 over the 2018 Wikipedia dump | TriviaQA and PopQA only |

`docs/reproduce.md` gives the exact commands for starting each service.
Then copy the environment template and fill in the endpoints:

```bash
cp .env.example .env
```

`LLM_MODEL` and `DISCERNMEM_EMBED_MODEL` must match the model names served by
the endpoints.

## Quick start

```bash
# One complete setting
python scripts/run.py --benchmark kodcode --mas camel

# All twelve main-result settings
for benchmark in triviaqa popqa kodcode pddl; do
  for mas in autogen camel macnet; do
    python scripts/run.py --benchmark "$benchmark" --mas "$mas"
  done
done
```

Run these commands in the main `discernmem` environment after starting the
services and configuring `.env`. Benchmark settings and result processing are
described in [the reproduction guide](docs/reproduce.md).

Arguments after `--` are passed unchanged to `scripts/run_benchmarks.py`.

## Outputs

Each run writes to `results/<run_id>/`:

```text
results/<run_id>/
  configs/<benchmark>_<mas>_discernmem.yaml    frozen method configuration
  <benchmark>/<mas>/discernmem/
    results/eval_results.json     final summary (pass_rate = mean task score in [0, 1])
    results/task_results.jsonl    one line per task
    logs/run.log                  full log
    memory/discernmem/records/    authoritative memory store (JSON)
    memory/discernmem/reports/    human-readable memory reports (index.md)
```

## Configuration

- `configs/benchmarks/<benchmark>.yaml`: dataset, generation, and interaction
  settings; the `protocol` section holds the runner settings used by
  `scripts/run.py` (task count, turns, environment mode, replay workers, ...).
- `configs/discernmem/default.yaml`: the full method.

The task manifest is generated at launch from the benchmark protocol, so every
configuration file stays short. `--dry-run` prints the resulting fingerprint.

## Third-party code and data

- `data/pddl/agentboard_pddlgym/` is an unmodified copy of the PDDLGym package
  and PDDL problems distributed with [AgentBoard](https://github.com/hkust-nlp/AgentBoard)
  (AgentBoard code: Apache-2.0; AgentBoard data: GPL-2.0; PDDLGym, from which it
  derives: MIT). `data/pddl/test.jsonl` holds the 60 PDDL test items of the
  AgentBoard dataset (GPL-2.0). License texts are in
  `data/pddl/agentboard_pddlgym/LICENSES/`.
- KodCode-Light-RL-10K (CC BY-NC 4.0), TriviaQA (Apache 2.0), and PopQA (from an
  MIT-licensed repository) are downloaded from the Hugging Face Hub on first use.
- QA retrieval uses the [Search-R1](https://github.com/PeterGriffinJin/Search-R1)
  retrieval server with its e5 index over the 2018 Wikipedia dump.

## License

This project's own code is released under the [MIT License](LICENSE),
Copyright (c) 2026 DiscernMem Contributors. Parts of the code are
adapted from the ConMem code release under the MIT license. Bundled third-party
code and data keep their own licenses, and downloaded datasets keep theirs; see
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
