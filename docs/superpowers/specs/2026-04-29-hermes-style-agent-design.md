# Hermes-Style Agent 设计文档

**日期：** 2026-04-29
**状态：** Approved
**框架：** LangGraph 1.1.10
**语言：** Python 3.11+

---

## 概述

基于 NousResearch/hermes-agent 架构理念，用 LangGraph 手写 StateGraph 构建一个全功能智能体，用于学习 agent 编排的核心流程。终端交互，支持多 LLM 提供商切换。

### 核心特性

- 手写 LangGraph StateGraph，完整 agent loop（ReAct 模式）
- 工具调用系统（8-12 个核心工具，三级风险标记）
- 四层记忆系统（会话上下文、MEMORY.md/USER.md、SQLite FTS5、FAISS 向量检索）
- Human-in-the-loop（危险操作确认、手工输入、质量干预）
- 多 Provider 支持（OpenAI、Anthropic、智谱 AI）
- 简化版自我进化（Nudge Engine + 统一 review）

---

## 1. AgentState & Graph 结构

### AgentState

```python
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]    # 对话历史
    tools: list[BaseTool]                       # 可用工具列表
    context_window: dict                        # {used_tokens, max_tokens, threshold}
    memory_context: str                         # 记忆检索结果
    user_profile: str                           # USER.md 快照
    agent_notes: str                            # MEMORY.md 快照
    pending_human_input: bool                   # human-in-the-loop 标志
    iteration_count: int                        # 循环计数
    max_iterations: int                         # 最大循环轮次
    nudge_counter: int                          # 自我进化触发计数
    provider_name: str                          # 当前提供商
```

### Graph 拓扑

```
START
  │
  ▼
[memory_retrieve]    检索 MEMORY.md + FTS5 + FAISS
  │
  ▼
[agent_node]         LLM 推理 + 工具调用决策
  │
  ├──(tool_calls)──▶ [tool_executor]   执行工具
  │                       │
  │                       ▼
  │                  [observe_node]    格式化工具结果
  │                       │
  │                       ▼
  │                  [human_gate]      危险操作检测
  │                    │         │
  │                (danger)   (safe)
  │                    │         │
  │                    ▼         │
  │            [__interrupt__]    │
  │                    │         │
  │                    ▼         │
  │            [human_input]     │
  │                    │         │
  │                    └─────────┘
  │                       │
  │                       ▼
  │                  [nudge_check]     自我进化触发
  │                       │
  │                       ▼
  │                  [memory_save]     持久化
  │                       │
  │                       └──────▶ [agent_node] (loop)
  │
  └──(no tool_call)──▶ END
```

### 条件边逻辑

| 来源 | 条件 | 目标 |
|------|------|------|
| agent_node | LLM 返回 tool_calls | tool_executor |
| agent_node | LLM 返回纯文本 | END |
| agent_node | iteration_count >= max | END |
| human_gate | tool.risk == "dangerous" | __interrupt__ |
| human_gate | tool.risk != "dangerous" | nudge_check |

---

## 2. 工具系统

### 工具列表（8-12 个）

**文件操作 (risk: write)**
- `read_file` — 读取文件内容
- `write_file` — 写入文件
- `list_dir` — 列举目录

**Web 工具 (risk: dangerous)**
- `web_search` — 网页搜索
- `web_fetch` — 获取网页内容

**系统工具 (risk: dangerous)**
- `run_shell` — 执行 shell 命令
- `get_time` — 获取当前时间

**Agent 工具 (risk: write)**
- `todo_write` — 任务列表管理
- `memory_search` — 检索历史记忆

### 风险等级

| 等级 | 含义 | human_gate 行为 |
|------|------|----------------|
| safe | 纯读取 | 直接通过 |
| write | 写入/修改 | 记录日志，不中断 |
| dangerous | 删除/网络/shell | 触发 interrupt，等待确认 |

### ToolExecutor

```python
class ToolExecutor:
    tools_by_name: dict[str, BaseTool]

    def execute(self, tool_calls: list[ToolCall]) -> list[ToolMessage]:
        # 1. 参数验证
        # 2. 执行工具
        # 3. 异常捕获 → Error ToolMessage
        # 4. 结果截断（>4000 chars 自动 trim）
```

---

## 3. 记忆系统

### 四层架构

| Layer | 存储 | 生命周期 | 用途 |
|-------|------|----------|------|
| L1 | AgentState.messages | 单次会话 | 上下文窗口 |
| L2 | MEMORY.md / USER.md | 永久 | agent 知识 + 用户画像 |
| L3 | SQLite FTS5 | 90 天 | 全文检索历史对话 |
| L4 | FAISS 向量索引 | 500 条 | 语义相似搜索 |

### 存储路径

```
~/.easy_agent/
├── memory/
│   ├── MEMORY.md          (~2000 chars cap)
│   ├── USER.md            (~1500 chars cap)
│   ├── history.db         (SQLite + FTS5)
│   └── vectors/           (FAISS index)
├── skills/                (技能笔记)
│   └── YYYY-MM-DD-<topic>.md
└── config.yaml
```

### memory_retrieve 节点

```
用户输入
  → FTS5 关键词检索 → top 3 历史对话
  → FAISS 语义检索  → top 3 相关记忆
  → 读取 MEMORY.md + USER.md
  → 合并注入 AgentState.memory_context
```

### memory_save 节点

```
每轮对话结束
  → 写入 SQLite history + FTS5
  → embedding + FAISS upsert
  → 更新 MEMORY.md / USER.md（仅 nudge 触发时）
```

### 容量控制

- MEMORY.md: ~2000 chars，超出时 LLM 压缩优先保留高频信息
- USER.md: ~1500 chars，同上
- SQLite: 保留 90 天内记录
- FAISS: 最多 500 条，超出时先进先出

---

## 4. Human-in-the-Loop

### 触发条件

1. **危险操作确认**: tool.risk == "dangerous"
2. **显式请求**: LLM 输出包含 `[HUMAN_INPUT: ...]`
3. **质量干预**: 连续 3 轮推理质量低

### 实现

使用 LangGraph 原生 `interrupt()` + `Command(resume=...)`:

```python
def human_gate(state: AgentState):
    last_tool = get_last_tool_call(state)
    if last_tool.risk == "dangerous":
        interrupt({
            "type": "tool_approval",
            "tool": last_tool.name,
            "args": last_tool.args,
        })
```

### 终端交互格式

```
⚠️  危险操作确认:
  工具: run_shell
  命令: rm -rf /tmp/build/
  批准? [y/n/skip] y
  ✅ 已执行
```

### 超时

- 中断等待超时 300s → 自动拒绝危险操作

---

## 5. 多 Provider 配置

### 配置文件 (config.yaml)

```yaml
providers:
  openai:
    type: openai
    api_key: "${OPENAI_API_KEY}"
    base_url: "https://api.openai.com/v1"
    default_model: "gpt-4.1"

  zhipu:
    type: openai_compatible
    api_key: "${ZHIPU_API_KEY}"
    base_url: "https://open.bigmodel.cn/api/paas/v4"
    default_model: "glm-4-plus"

  anthropic:
    type: anthropic
    api_key: "${ANTHROPIC_API_KEY}"
    default_model: "claude-sonnet-4-6"

active_provider: "zhipu"
fallback_provider: "openai"

agent:
  max_iterations: 15
  context_compression_threshold: 0.7
  memory_nudge_interval: 10
```

### Provider 适配层

```
providers/
├── base.py              # BaseProvider (ABC)
├── openai_provider.py   # ChatOpenAI
├── anthropic_provider.py # ChatAnthropic
└── factory.py           # get_provider(name) -> BaseProvider
```

### 切换方式

```bash
python main.py --provider zhipu
python main.py --provider openai --model gpt-4.1
python main.py --provider anthropic --model claude-sonnet-4-6
```

### 降级策略

```
推理失败 → 自动重试 (max 2) → 降级备选 provider → 友好报错
```

---

## 6. 简化版自我进化

### 触发

每 N 轮对话后（默认 10），nudge_check 节点触发异步 review。

### Review 流程

```python
async def nudge_review(state: AgentState):
    recent = extract_recent_turns(state.messages, n=10)
    prompt = """回顾对话，判断是否需要更新:
    1. 新的用户偏好 → 更新 USER.md
    2. 新的项目知识 → 更新 MEMORY.md
    3. 值得记录的解决方案 → 创建 skill 笔记
    如无需更新，回复 NOTHING_TO_SAVE。"""
    result = await llm.ainvoke(prompt)
    if "USER" in result: update_user_md()
    if "MEMORY" in result: update_memory_md()
    if "SKILL" in result: save_skill_note()
```

### 与 Hermes 完整版的差异

| 特性 | Hermes | easy_agent |
|------|--------|------------|
| 执行方式 | 后台 fork 独立进程 | 同进程 asyncio.create_task |
| Review 类型 | 3 种（memory/skill/combined） | 统一 review |
| 安全扫描 | 完整 prompt injection 检测 | 关键词过滤 |
| 复杂度 | ~2000 行 | ~200 行 |

### 技能笔记格式

```markdown
---
name: fix-pnpm-peer-deps
created: 2026-04-29
context: 解决 pnpm peer dependency 冲突
---

## 解决步骤
1. ...
## 注意事项
- ...
```

---

## 7. 项目结构

```
easy_agent/
├── main.py                    # 入口，CLI 参数解析
├── config.yaml                # 默认配置
├── requirements.txt           # 依赖
├── src/
│   ├── __init__.py
│   ├── graph.py               # StateGraph 构建（核心）
│   ├── state.py               # AgentState 定义
│   ├── nodes/
│   │   ├── __init__.py
│   │   ├── agent_node.py      # LLM 推理节点
│   │   ├── tool_executor.py   # 工具执行节点
│   │   ├── memory_retrieve.py # 记忆检索节点
│   │   ├── memory_save.py     # 记忆持久化节点
│   │   ├── human_gate.py      # Human-in-the-loop 节点
│   │   ├── observe_node.py    # 工具结果格式化
│   │   ├── human_input.py     # human-in-the-loop 恢复
│   │   └── nudge_check.py     # 自我进化节点
│   ├── tools/
│   │   ├── __init__.py
│   │   ├── file_tools.py
│   │   ├── web_tools.py
│   │   ├── system_tools.py
│   │   └── agent_tools.py
│   ├── memory/
│   │   ├── __init__.py
│   │   ├── fts5_store.py      # SQLite FTS5
│   │   ├── vector_store.py    # FAISS
│   │   └── file_memory.py     # MEMORY.md / USER.md
│   └── providers/
│       ├── __init__.py
│       ├── base.py
│       ├── openai_provider.py
│       ├── anthropic_provider.py
│       └── factory.py
├── docs/
│   └── superpowers/specs/
│       └── 2026-04-29-hermes-style-agent-design.md
└── tests/
    ├── test_graph.py
    ├── test_tools.py
    └── test_memory.py
```

## 8. 依赖

```
langgraph>=1.1.10
langchain>=0.3.13
langchain-core>=0.3.28
langchain-openai>=0.2.14
langchain-anthropic>=0.3.0
faiss-cpu>=1.8.0
sentence-transformers>=3.0.0
pyyaml>=6.0
```

## 9. 风险 & 注意事项

- FTS5 和 FAISS 首次初始化需要少量计算（embedding 模型加载）
- Shell 工具在终端 agent 中存在执行风险，human_gate 必须严格拦截危险命令
- Anthropic provider 需要 `ANTHROPIC_API_KEY`，与 OpenAI 体系不兼容
- 简化版 self-evolution 不做复杂安全扫描，仅关键词过滤
