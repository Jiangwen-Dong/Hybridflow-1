# 高效的 LLM 协同推理框架 (Efficient LLM Collaboration Inference)

高效 LLM 协同推理框架是一个先进的系统，旨在优化多个大型语言模型 (LLM) 的协同推理过程。通过利用自动任务分解、动态模型分配和并行处理，该系统能够高效地解决单个模型难以应对的复杂问题。

## 核心特性

  - **自动任务分解**: 利用一个成熟的“规划器”(Planner)模型将复杂问题分解为由一系列更小、可管理的子任务组成的有向无环图 (DAG)。
  - **动态模型分配**: 根据每个子任务的预估难度，智能地为其分配最合适的模型（例如，更小、更快的模型或更大、更强大的模型）。
  - **并行任务执行**: 采用多线程调度器并行执行独立的子任务，显著减少了总处理时间。
  - **全面的性能监控**: 跟踪并报告详细的性能指标，包括执行时间、首个令牌生成时间 (TTFT)、令牌使用量和预估成本，为系统效率提供深入的洞察。
  - **批量处理与评估**: 支持对数据集进行批量处理和系统评估，从而实现稳健的模型和系统性能分析。
  - **数据集构建模式**: 内置数据集构建功能，可根据问题和参考答案自动生成高质量的“问题-计划”对，用于规划器模型的微调。
  - **理论性能建模**: 能够根据任务计划和模型性能数据，生成理论上的性能报告（包括甘特图），用于分析和优化任务调度。
  - **基准对比工具**: 提供独立的单一模型测试脚本 (`single_model_only.py`)，用于直接与协同框架的性能进行基准比较。

## 工作原理

该系统通过一个结构化的多阶段流程进行运作：

1.  **规划 (任务分解)**: 用户的查询首先被发送到一个 **规划器模型** (Planner Model)。该模型分析问题并生成一个详细的 XML 格式的执行计划。该计划概述了一系列步骤，每个步骤都包含具体的任务、预估的难度、令牌预算及其对其他步骤的依赖关系。
2.  **调度 (任务分派)**: 系统解析 XML 计划以构建任务依赖图 (DAG)。然后，调度器识别出所有依赖项已满足的任务，并将它们分派以供执行。
3.  **执行 (解决子任务)**: 每个被分派的任务都被发送到一个 **执行器模型** (Executor Model)。系统会根据计划中的 `Difficulty` (难度) 属性，为较简单的任务动态选择一个小型高效的模型，或为较复杂的任务选择一个大型强大的模型。
4.  **并行处理**: 调度器使用线程池并发执行多个独立的子任务。当任务完成时，调度器会检查是否有新的任务因依赖满足而解锁，并将其添加到执行队列中。
5.  **聚合 (生成最终答案)**: 一旦计划中的所有任务都完成，系统会收集每个步骤的结果，并进行最终的聚合，以生成对原始查询的最终答案。
6.  **报告**: 流程结束后，系统会生成一份详细的报告，包括性能指标、成本分析、任务依赖图和量化的评估分数。

## 安装与设置

1.  **安装依赖**

    建议首先创建一个虚拟环境。

    ```bash
    python -m venv venv
    source venv/bin/activate  # 在 Windows 上，使用 `venv\Scripts\activate`
    ```

    从 `requirements.txt` 文件中安装所需的包：

    ```bash
    pip install -r requirements.txt
    ```

2.  **配置 API 密钥和模型**

      - 复制配置文件示例：
        ```bash
        cp config.example.yaml config.yaml
        ```
      - 编辑 `config.yaml` 文件，设置您的模型和 API 凭证。
      - 将您的 API 密钥放置在 `*_key_path` 变量指定的文件中（例如，在一个名为 `usage/` 的文件夹中）。

## 配置文件 (`config.yaml`) 详解

`config.yaml` 是系统的中央控制文件。以下是各部分的详细说明：

```yaml
# 模型配置
models:
  small_model: "qwen2.5-3b-instruct" 
  large_model: "gpt-4o"
  router_model: "qwen3-1.7b"
  enable_threshold: True
  threshold: 5  # 难度大于等于此值时使用大模型
  use_local_router: True  # 是否使用本地部署的规划器模型
  local_router_model: "saves/Qwen3-1.7B-Thinking/full/train_2025-09-25-23-33-09" # 本地模型路径

# API配置
api:
  small_key_path: "usage/qwen"
  large_key_path: "usage/bianxie1"
  router_key_path: "usage/local"
  small_api_base_url: "https://dashscope.aliyuncs.com/compatible-mode/v1"
  large_api_base_url: "https://api.bianxie.ai/v1"
  router_api_base_url: "https://dashscope.aliyuncs.com/compatible-mode/v1"
  local_router_base_url: "http://127.0.0.1:8000/v1" # 本地规划器API

# 系统配置
system:
  workers: 10  # 并行工作线程数

# 查询配置 (用于单一查询模式)
query: "Define all possible orientations and placements of the L-shaped tile within the 2x5 rectangle."

# 数据集配置 (用于数据集评估或构建模式)
dataset:
  enabled: True  # 设为 true 启用数据集模式
  path: "dataset/TestData/MMLU-STEM.json"
  limit: 50 # 可选：限制处理的问题数量
  
  # 数据集构建配置
  build:
    enabled: False  # 设为 true 以构建数据集，而不是评估
    use_models_for_execution: False  # 如果为 false，只生成计划而不执行子任务
    save_thinking: False  # 是否在输出中保存 planner 的 <think> 块

# 评估配置
evaluation:
  enabled: False  # 评估功能的总开关
  planner_enabled: False  # 规划器评估开关
  executor_enabled: False  # 执行器评估开关
  model: "deepseek-chat"  # 用于评估的裁判模型
  key_path: "usage/deepseek2"
  api_base_url: "https://api.deepseek.com"
```

## 使用方法：协同推理框架

系统支持三种主要运行模式：单一查询、数据集评估和数据集构建。

### 1. 单一查询模式

用于处理单个复杂问题。

1.  在 `config.yaml` 中，确保 `dataset: enabled:` 为 `false`。
2.  在 `query:` 字段中定义您的问题。
3.  运行主脚本：
    ```bash
    python main.py --config config.yaml
    ```
4.  详细结果将保存在 `data_reports/single/` 目录中，并按模型配置和时间戳进行组织。

### 2. 数据集评估模式

用于在指定数据集上批量运行并评估系统性能。

1.  在 `config.yaml` 中，启用 `dataset` 部分，并确保 `build: enabled:` 为 `false`。
    ```yaml
    dataset:
      enabled: true
      path: "dataset/TestData/MMLU-STEM.json"
      limit: 50 # 可选
    ```
2.  运行主脚本：
    ```bash
    python main.py
    ```
3.  评估报告将保存在 `data_reports/dataset/` 目录中。报告 `dataset_report.md` 中会包含**任务规划指标**：
      * **平均任务步骤数**: Planner 为每个问题平均分解出的子任务数量。
      * **平均压缩比例**: 任务图的关键路径深度 / 总任务数。**此值越低，代表任务的并行潜力越高**。
      * **平均每步骤Token限制**: Planner 为每个子任务分配的平均令牌预算。

### 3. 数据集构建模式

此模式用于为微调规划器（Planner）生成训练数据。

1.  在 `config.yaml` 中，启用 `dataset` 和 `build` 部分：
    ```yaml
    dataset:
      enabled: true
      path: "path/to/your/source_dataset.json"
      build:
        enabled: true # 激活构建模式
    ```
2.  运行主脚本：`python main.py`
3.  生成的训练数据将保存在 `data_reports/dataset/` 相应的时间戳文件夹中。

## 使用方法：单一模型基准测试

为了验证协同框架的有效性，项目提供了一个独立的基准测试脚本 `single_model_only.py`，用于评估单个模型直接解决问题的性能。

### 1. 单一查询模式 (基准测试)

1.  直接通过命令行参数运行：
    ```bash
    python single_model_only.py --query "你的问题" --model "gpt-4o" --timeout 60
    ```

### 2. 数据集评估模式 (基准测试)

1.  修改 `config.yaml` 中的 `dataset` 部分，指向您要测试的数据集。
2.  通过命令行运行，可以指定模型和超时时间：
    ```bash
    python single_model_only.py --dataset "dataset/your_dataset.json" --limit 10 --model "gpt-4o" --timeout 120
    ```
3.  测试结果将保存在 `data_reports/single_model_results/` 目录下，按模型名称和时间戳分子文件夹。

**命令行参数说明 (`single_model_only.py`)**:

  - `--query`: 要解决的问题。
  - `--dataset`: 数据集文件路径。
  - `--model`: 指定要使用的模型名称 (会覆盖配置文件)。
  - `--limit`: 处理数据集的最大问题数。
  - `--timeout`: 模型请求的超时时间(秒)。
  - `--config`: 配置文件路径 (默认为 `config.yaml`)。

## 辅助工具

### 任务分配分析脚本

在数据集评估模式运行后，您可以使用 `analyze_model_tasks.py` 脚本来分析任务在大小模型之间的分配情况。

**使用方法:**

```bash
# python analyze_model_tasks.py <dataset_results.json路径> [可选：难度阈值]
python analyze_model_tasks.py data_reports/dataset/.../your_timestamp/dataset_results.json 5
```

## 评估框架

为确保系统质量，我们采用了一个严格的评估框架，该框架使用一个强大的 LLM 作为裁判，对 **规划器** (Planner) 和 **执行器** (Executor) 模型进行评估。您可以在 `config.yaml` 的 `evaluation` 部分启用此功能。

## 代码库结构

| 文件 | 描述 |
| --- | --- |
| `main.py` | **协同框架**的主入口点。处理参数解析并根据配置协调工作流程。 |
| `single_model_only.py` | **单一模型基准测试**的独立脚本，用于性能对比。 |
| `config.yaml` | 用于模型、API 密钥、系统设置和数据集路径的中央配置文件。 |
| `execution.py` | 包含协同框架的并行任务执行、API 调用和规划器流式响应处理的核心逻辑。 |
| `dataset_runner.py` | 管理协同框架的数据集批量处理（评估或构建），并生成综合报告。 |
| `evaluation.py` | 实现用于评估规划器和执行器性能的裁判评估框架。 |
| `performance.py` | `PerformanceTracker` 类，用于监控令牌使用、成本和执行时间等指标。 |
| `output_performance.py` | 根据模型特定的延迟和吞吐量数据计算理论性能基准。 |
| `task_metrics.py` | 计算任务依赖图 (DAG) 的结构指标，如深度和压缩率。 |
| `config.py` | 定义 `ModelConfig` 类，用于以编程方式管理模型配置和 API 客户端。 |
| `api_pricing.py` | 提供各种 LLM API 最新定价信息的实用工具模块。 |
| `analyze_model_tasks.py`| 用于分析和报告小模型与大模型之间任务分配情况的脚本。 |
| `utils.py` | 包含如报告路径生成等通用辅助函数。 |
| `log_config.py` | 配置日志记录器，将详细运行信息保存到文件中。 |