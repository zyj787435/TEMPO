## AUTOTOOL论文复现&改进 数据集&对照论文集

### 1 AUTOTOOL论文
论文中的“工具”分为两种，一种是Alfworld、Scienceworld里定义的和环境交互的动作，另一种是Toolquery里面模拟/真实的API调用


### 2 数据集

| 名称 | 具体内容 | 来源 | 格式 | 备注 |
|------|----------|------|------|------|
| **ToolBench** | 大规模真实 API 工具调用数据集，包含 16,000+ 工具与多步调用轨迹，任务以复杂 API 组合调用为主，体现强烈的工具调用惯性与序列依赖 | https://github.com/OpenBMB/ToolBench | JSON（工具 schema + 调用轨迹） | 非常适合构建工具图、分析 tool transition、做图搜索与规划；算力与工程成本较高 |
| **ScienceWorld** | 文本交互式科学实验环境，任务涉及多步操作（观察、移动、操作物体、实验流程），工具调用序列长、参数依赖强 | https://github.com/allenai/ScienceWorld | Text-based simulator（step-by-step interaction） | 图搜索、状态感知、参数依赖实验的黄金数据集；AutoTool 等方法的核心 benchmark |
| **ToolQuery-Academic** | 学术查询类多步工具调用任务，涉及作者、论文、引用关系等 API 查询，强调结构化参数传递 | https://github.com/THUDM/AgentBench/tree/main/ToolQuery | JSON / API query logs | 工具参数结构清晰，适合验证参数图搜索与 API 场景泛化；规模中等，易复现 |
| **AgentBench** | 综合性 LLM Agent 基准测试集，涵盖 Web、数据库、API、多步决策等多种真实场景 | https://github.com/THUDM/AgentBench | 多格式（Web / API / Text） | 覆盖面广，适合验证方法的跨领域泛化能力；不同子任务复杂度差异较大 |
| **AgentBoard** | 多步 Agent 行为评测平台，提供统一的 progress rate 评估指标，包含 ToolQuery、Web、Finance 等子任务 | https://github.com/THUDM/AgentBoard | 标准化任务描述 + 执行日志 | 审稿人友好，评价体系成熟；非常适合效率（LLM calls / token）对比实验 |
| **WebArena** | 真实网页环境中的多步操作任务（搜索、点击、填写表单等），强调长时规划与状态变化 | https://github.com/web-arena-x/webarena | 浏览器交互环境（HTML + DOM） | 强状态依赖、易产生 loop；非常适合 state-aware graph search 与冗余动作消除实验 |
| **ToolQA** | 标准化评测数据集，旨在评估大模型必须通过调用外部工具才能正确回答问题的能力，问题覆盖多种知识格式与复杂程度| https://github.com/night-chen/ToolQA | JSON、按领域和难度划分 | 可作为第三方基准对比多种工具使用方法的准确率 |


### 3 对照论文集

**1.Reflexion** 
https://arxiv.org/abs/2303.11366
https://github.com/noahshinn/reflexion
代理在执行失败后对轨迹进行语言化“反思（Reflection）”，并将总结作为知识加入长期记忆

**2.ART**
https://arxiv.org/abs/2303.09014

**3.ToolNet** *原文对照*
https://arxiv.org/abs/2403.00839v1

**4.ToolPlanner** *原文对照*
https://arxiv.org/html/2409.14826v2
https://github.com/OceannTwT/Tool-Planner

**5.Toolformer**
https://arxiv.org/abs/2302.04761
https://github.com/lucidrains/toolformer-pytorch  *一个pytorch实现*

**6.DFSDT** *原文对照*
https://arxiv.org/abs/2307.16789v2
https://github.com/OpenBMB/ToolBench

**7. Anytool** *原文对照*
https://arxiv.org/abs/2402.04253
https://github.com/dyabel/AnyTool
*工具来源为从RapidAPI Hub里在线搜索相关API*

**8.Toolchain** *原文对照*
https://arxiv.org/abs/2310.13227

**9.LLMCompiler** *原文对照*
https://arxiv.org/abs/2312.04511
https://github.com/SqueezeAILab/LLMCompiler


### 4 
