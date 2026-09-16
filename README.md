# ContractFlow

**A verifiable contract framework for constrained long-form generation.**

[English](#english) | [中文](#中文)

## English

### Overview

ContractFlow is a research framework for long-form generation under distributed
constraints. It converts a natural-language request into persistent,
segment-level contracts, generates the document in resumable batches, verifies
the resulting segments, and repairs only the parts that fail their obligations.

The repository accompanies the paper:

> **ContractFlow: A Verifiable Contract Framework for Constrained Long-Form Generation**

The current experimental scope has two parts:

- **LongGenBench**, the main experiment for measuring instruction persistence
  in structured long outputs at 16K and 32K.
- **MoreLongWrite**, a transfer experiment covering six families of natural
  long-writing tasks at 16K, 32K, and 64K.

### Method

ContractFlow treats long generation as stateful execution rather than a single
completion:

1. Parse the public task prompt and construct a global execution plan.
2. Compile single-instance, range, and periodic requirements into persistent
   segment contracts.
3. Generate indexed segments in batches while carrying forward local context.
4. Check structural completeness and review segments against their contracts.
5. Repair failed segments without rewriting already valid content.
6. Assemble the final document and retain checkpoints and call metadata for
   reproducible analysis.

### LongGenBench

LongGenBench is the primary experimental setting. The code supports:

- `direct`: one-shot generation from the original task prompt.
- `contractflow`: contracts and review/repair enabled.
- `agentwrite`: plan-then-write baseline.
- `cogwriter`: official CogWriter workflow with runtime task adaptation.
- `self_refine`: one or more full-document refinement iterations initialized
  from the corresponding Direct output.

The controlled mechanism study is exposed as a `2 x 2` design:

| Variant | Contracts | Review/Repair |
| --- | --- | --- |
| `contractflow` | Yes | Yes |
| `no_contract` | No | Yes |
| `no_review_repair` | Yes | No |
| `neither` | No | No |

The no-contract variants retain segmented generation but omit contract
compilation and contract-guided semantic checks. The no-review/repair variants
stop after drafting and preserve any incomplete structure for evaluation.

### MoreLongWrite

MoreLongWrite evaluates transfer beyond indexed synthetic constraints. It
contains natural-language tasks from academic and technical writing, education
and training, popular science, professional writing, news and investigation,
and literary and creative writing.

The transfer workflow compares `direct`, `agentwrite`, and `contractflow`.
ContractFlow selects an expository, academic/technical, or narrative adapter,
creates section contracts from the original prompt, and applies bounded local
review and repair.

### Repository Layout

```text
learnbyai_research/
  longgenbench.py                    ContractFlow and LongGen task utilities
  longgenbench_agentwrite.py         AgentWrite adaptation
  longgenbench_cogwriter_official.py Official CogWriter integration
  longgenbench_self_refine.py        Self-Refine adaptation
  longgenbench_official_eval.py      Official-flow evaluation utilities
  longbench_write_transfer.py        MoreLongWrite task and method adapters
  morelongwrite_optimized.py         MoreLongWrite ContractFlow workflow
  adaptive_generation.py             Budget-aware generation utilities
  llm.py                             OpenAI-compatible model client
scripts/
  run_longgenbench.py                Direct, ContractFlow, and 2 x 2 ablations
  run_longgenbench_baselines.py      LongGen comparison systems
  run_morelongwrite.py               MoreLongWrite transfer experiments
  validate_release.py                Release integrity checks
configs/
  longgen.template.json
  morelongwrite.template.json
tests/
```

### Installation

Python 3.10 or later is required.

```bash
python -m venv .venv
python -m pip install -e .
```

### Model Roles

Model clients are configured locally by role:

| Role | Purpose |
| --- | --- |
| `AUTHOR` | Planning and initial generation |
| `REVIEW` | Contract review and issue diagnosis |
| `REPAIR` | Localized revision |
| `JUDGE` | Post-hoc evaluation |

Configure model clients through role-based environment variables. The available
variable names are listed in `.env.example`; experiment settings can be supplied
through the command-line interfaces and configuration templates.

### Running Experiments

Inspect the complete required arguments for the LongGen primary systems and
mechanism ablations:

```bash
python scripts/run_longgenbench.py --help
```

Inspect the LongGen baseline runner:

```bash
python scripts/run_longgenbench_baselines.py --help
```

Inspect the MoreLongWrite transfer runner:

```bash
python scripts/run_morelongwrite.py --help
```

For Self-Refine, generate the matching `direct` artifact first. AgentWrite and
CogWriter integrations expect their upstream source trees to be prepared
locally when required by the selected workflow.

### Evaluation and Testing

LongGen evaluation utilities preserve indexed-block parsing and the released
Once, Range, Periodic, completion, and aggregation flow. MoreLongWrite records
writing quality and deterministic length compliance separately.

```bash
python -m pytest
python scripts/validate_release.py
```

### License

ContractFlow is licensed under the [PolyForm Noncommercial License 1.0.0](LICENSE).
It is made available for **non-commercial academic research, education,
evaluation, reproducibility studies, and other non-commercial purposes** in
accordance with that license.

**Commercial use is not permitted without separate prior written authorization
from the copyright holders.** This includes incorporating ContractFlow or
substantial portions of its source code into a commercial product or paid
service, commercial deployment, sublicensing, sale, or other commercial
exploitation.

If you use ContractFlow in academic work, please cite the corresponding paper
and this repository. For commercial licensing or collaboration, please contact
the copyright holders.

---

## 中文

### 项目简介

ContractFlow 是一个面向约束长文本生成的可验证契约框架。它先把自然语言任务
转化为可持续维护的分段契约，再以可恢复的批次生成文档，对结果进行结构检查和
语义审查，并只修复未满足约束的局部内容。

本仓库对应论文：

> **ContractFlow: A Verifiable Contract Framework for Constrained Long-Form Generation**

当前实验由两部分组成：

- **LongGenBench**：论文主实验，用于评估 16K 和 32K 长度下分布式指令能否
  持续得到满足。
- **MoreLongWrite**：迁移实验，覆盖六类自然长写作任务以及 16K、32K、64K
  三种目标长度。

### 方法流程

ContractFlow 将长文本生成视为一个有状态的执行过程：

1. 解析公开任务提示，建立全局执行计划。
2. 将单点、区间和周期要求编译为持续生效的分段契约。
3. 分批生成带索引的文本片段，并维护必要的上下文状态。
4. 检查结构完整性，并依据契约审查各个片段。
5. 只修复未通过检查的片段，保留其他已经正确的内容。
6. 组装最终文档，同时保存检查点和调用元数据以支持复现实验。

### LongGenBench 主实验

LongGenBench 是当前论文的核心实验。仓库支持以下系统：

- `direct`：直接根据原始任务提示进行一次性生成。
- `contractflow`：启用契约与审查/修复的完整方法。
- `agentwrite`：先规划、再逐段写作的基线。
- `cogwriter`：基于官方 CogWriter 流程并进行运行时任务适配。
- `self_refine`：从对应的 Direct 输出开始进行整篇反馈与改写。

论文中的机制消融采用 `2 x 2` 设计：

| 变体 | 契约 | 审查/修复 |
| --- | --- | --- |
| `contractflow` | 启用 | 启用 |
| `no_contract` | 关闭 | 启用 |
| `no_review_repair` | 启用 | 关闭 |
| `neither` | 关闭 | 关闭 |

无契约变体仍然采用分段生成，但不执行契约编译和契约驱动的语义检查；关闭
审查/修复后，系统会在初稿生成完成后停止，并将不完整结构原样交给评测流程。

### MoreLongWrite 迁移实验

MoreLongWrite 用于检验 ContractFlow 能否迁移到不带人工结构化约束的自然写作
任务。任务覆盖学术与技术、教育与培训、科普、职场与功能写作、新闻与调查、
文学与创意写作六个类别。

迁移实验比较 `direct`、`agentwrite` 和 `contractflow`。ContractFlow 会根据
原始提示选择说明型、学术技术型或叙事型适配器，建立分段执行契约，并执行
有界的局部审查与修复。

### 代码结构

```text
learnbyai_research/
  longgenbench.py                    ContractFlow 与 LongGen 任务工具
  longgenbench_agentwrite.py         AgentWrite 适配
  longgenbench_cogwriter_official.py 官方 CogWriter 集成
  longgenbench_self_refine.py        Self-Refine 适配
  longgenbench_official_eval.py      官方流程评测工具
  longbench_write_transfer.py        MoreLongWrite 任务与方法适配
  morelongwrite_optimized.py         MoreLongWrite 的 ContractFlow 流程
  adaptive_generation.py             预算感知生成工具
  llm.py                             OpenAI 兼容模型客户端
scripts/
  run_longgenbench.py                主方法、Direct 与 2 x 2 消融
  run_longgenbench_baselines.py      LongGen 对比基线
  run_morelongwrite.py               MoreLongWrite 迁移实验
  validate_release.py                发布完整性检查
configs/
  longgen.template.json
  morelongwrite.template.json
tests/
```

### 安装

需要 Python 3.10 或更高版本。

```bash
python -m venv .venv
python -m pip install -e .
```

### 模型角色

模型客户端按角色在本地配置：

| 角色 | 用途 |
| --- | --- |
| `AUTHOR` | 规划与初始生成 |
| `REVIEW` | 契约审查与问题诊断 |
| `REPAIR` | 局部修复 |
| `JUDGE` | 生成后的独立评测 |

模型客户端通过按角色划分的环境变量进行配置，变量名可参考 `.env.example`；
实验设置可通过命令行入口和配置模板传入。

### 运行实验

查看 LongGen 主实验和机制消融所需参数：

```bash
python scripts/run_longgenbench.py --help
```

查看 LongGen 基线运行参数：

```bash
python scripts/run_longgenbench_baselines.py --help
```

查看 MoreLongWrite 迁移实验参数：

```bash
python scripts/run_morelongwrite.py --help
```

运行 Self-Refine 前，需要先生成对应的 `direct` 结果。AgentWrite 和 CogWriter
流程在需要时会读取本地准备好的上游源码目录。

### 评测与测试

LongGen 评测工具保留索引片段解析，以及 Once、Range、Periodic、Completion
和聚合流程。MoreLongWrite 分别记录写作质量与确定性的长度符合度。

```bash
python -m pytest
python scripts/validate_release.py
```

### 许可声明

ContractFlow 采用 [PolyForm Noncommercial License 1.0.0](LICENSE)。按照该许可证，
本项目可用于**非商业的学术研究、教育、评测、复现实验及其他非商业用途**。

**未经版权持有人另行事先书面授权，不得将 ContractFlow 用于商业用途。**
商业用途包括但不限于：将 ContractFlow 或其实质性代码纳入商业产品或收费服务、
进行商业部署、转授权、出售，或以其他方式进行商业利用。

如在学术工作中使用 ContractFlow，请引用对应论文及本仓库。商业授权或合作事宜
请联系版权持有人。
