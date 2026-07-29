# ContractFlow

[English](#english) | [中文](#中文)

## English

ContractFlow is a research implementation of contract-guided long-form generation. Structured contracts coordinate planning, generation, review, verification, and localized repair across textbook generation, LongGenBench-style structured writing, and LongBench-Write transfer experiments.

### Included

- Contract schemas, sequence checks, evidence extraction, and verification.
- Full, direct, single-agent, and ablation pipelines.
- LongGenBench planning, generation, review, repair, and evaluation utilities.
- LongBench-Write transfer and optimized long-writing utilities.
- Blind pairwise evaluation and aggregate metrics.
- Deterministic mock execution and unit tests.
- Empty environment, task, and experiment templates.

### Not Included

- Credentials, tokens, private endpoints, or provider URLs.
- Filled model names, budgets, retry counts, concurrency values, or task counts.
- Generated outputs, checkpoints, logs, local paths, paper files, or author metadata.
- Third-party source trees or benchmark datasets.

### Installation

```bash
python -m venv .venv
python -m pip install -e .
```

### Local Configuration

Copy `.env.example` to `.env` and fill the required fields locally. Every published field is intentionally blank. The completed `.env` file is ignored by Git.

### Deterministic Smoke Test

The core pipeline can be exercised without credentials:

```bash
python -m learnbyai_research.cli generate \
  --task path/to/task.json \
  --pipeline full \
  --output runs/smoke.json \
  --mock
```

Use `tasks/textbook_task.template.json` as the task schema. Fill all required values locally before execution.

### LongGenBench

The LongGen runner does not contain experiment-specific values. All task counts, worker counts, batch sizes, and token budgets are required command-line arguments:

```bash
python scripts/run_longgenbench.py --help
```

### LongBench-Write

Create a local configuration from `configs/longbench_write.template.json`, then run:

```bash
python scripts/run_longbench_write_transfer.py --help
```

### Evaluation

```bash
python -m learnbyai_research.cli judge --help
python -m learnbyai_research.cli aggregate --help
python -m learnbyai_research.cli contract-metrics --help
python -m learnbyai_research.cli task-spec-metrics --help
```

### Tests And Release Audit

```bash
python -m pytest
python scripts/validate_release.py
```

The release audit scans tracked files for secret-like values, external URLs, generated artifacts, and forbidden local configuration paths.

### Anonymous Supplement Packaging

Create the supplementary archive from a clean checkout without the `.git` directory. Do not include author names, affiliations, acknowledgements, submission identifiers, or links to online repositories in an anonymous submission package.

## 中文

ContractFlow 是一个面向长文本生成的契约引导研究实现。系统通过结构化契约协调规划、生成、审查、验证和局部修复，覆盖教材生成、LongGenBench 风格结构化写作与 LongBench-Write 迁移实验。

### 已包含

- 契约结构、顺序检查、证据提取与验证。
- 完整流程、直接生成、单智能体基线与消融流程。
- LongGenBench 的规划、生成、审查、修复和评测工具。
- LongBench-Write 迁移与优化长文本生成工具。
- 盲评成对比较和聚合指标。
- 确定性 Mock 执行与单元测试。
- 空白环境变量、任务和实验配置模板。

### 未包含

- 密钥、令牌、私有接口地址或服务商 URL。
- 已填写的模型名、预算、重试次数、并发数或任务数量。
- 生成结果、检查点、日志、本机路径、论文文件或作者信息。
- 第三方源码目录或 benchmark 数据集。

### 安装

```bash
python -m venv .venv
python -m pip install -e .
```

### 本地配置

将 `.env.example` 复制为 `.env`，并仅在本地填写所需字段。公开模板中的所有参数均有意留空，完成后的 `.env` 已被 Git 忽略。

### 确定性冒烟测试

核心流程可以在没有密钥的情况下运行：

```bash
python -m learnbyai_research.cli generate \
  --task path/to/task.json \
  --pipeline full \
  --output runs/smoke.json \
  --mock
```

任务结构见 `tasks/textbook_task.template.json`，运行前需在本地填写所有必需字段。

### LongGenBench

LongGen 启动脚本不保存本次实验的具体值。任务数量、worker 数、batch 大小和 token 预算都必须由命令行显式提供：

```bash
python scripts/run_longgenbench.py --help
```

### LongBench-Write

先根据 `configs/longbench_write.template.json` 创建本地配置，再运行：

```bash
python scripts/run_longbench_write_transfer.py --help
```

### 评测

```bash
python -m learnbyai_research.cli judge --help
python -m learnbyai_research.cli aggregate --help
python -m learnbyai_research.cli contract-metrics --help
python -m learnbyai_research.cli task-spec-metrics --help
```

### 测试与发布检查

```bash
python -m pytest
python scripts/validate_release.py
```

发布检查会扫描已跟踪文件中的疑似密钥、外部 URL、生成产物和禁止提交的本地配置路径。

### 匿名补充材料打包

匿名补充材料应从干净检出中创建，且不包含 `.git` 目录。匿名提交包中不要出现作者姓名、单位、致谢、投稿编号或在线仓库链接。
