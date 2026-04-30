# Easy Agent 使用说明书

> 版本：v0.1.0
> 更新日期：2026-04-30

---

## 目录

1. [快速开始](#1-快速开始)
2. [安装配置](#2-安装配置)
3. [启动方式](#3-启动方式)
4. [交互命令](#4-交互命令)
5. [Provider 配置](#5-provider-配置)
6. [工具系统](#6-工具系统)
7. [记忆系统](#7-记忆系统)
8. [常见问题](#8-常见问题)

---

## 1. 快速开始

```bash
# 1. 进入项目目录
cd /Users/yanxs/code/ai_coding/learn/easy_agent

# 2. 安装依赖
pip install -r requirements.txt

# 3. 设置 API 密钥
export ZHIPU_API_KEY="你的智谱AI密钥"

# 4. 启动
python main.py
```

---

## 2. 安装配置

### 2.1 环境要求

- Python 3.11+
- pip 包管理器

### 2.2 安装依赖

```bash
pip install -r requirements.txt
```

依赖包列表：
- `langgraph>=1.1.10` - Agent 图编排框架
- `langchain>=0.3.13` - LLM 工具链
- `langchain-openai>=0.2.14` - OpenAI 适配
- `langchain-anthropic>=0.3.0` - Anthropic 适配
- `faiss-cpu>=1.8.0` - 向量检索
- `sentence-transformers>=3.0.0` - 文本嵌入
- `pyyaml>=6.0` - 配置解析

### 2.3 配置 API 密钥

**方式一：环境变量（推荐）**

```bash
# 智谱 AI
export ZHIPU_API_KEY="your-zhipu-api-key"

# OpenAI
export OPENAI_API_KEY="your-openai-api-key"

# Anthropic
export ANTHROPIC_API_KEY="your-anthropic-api-key"
```

**方式二：修改 config.yaml**

```yaml
providers:
  zhipu:
    api_key: "直接填写密钥"  # 不推荐，有安全风险
```

---

## 3. 启动方式

### 3.1 基本启动

```bash
# 使用默认 Provider（config.yaml 中的 active_provider）
python main.py

# 指定 Provider
python main.py --provider zhipu
python main.py --provider openai
python main.py --provider anthropic

# 指定模型
python main.py --provider openai --model gpt-4o
python main.py --provider zhipu --model glm-4-plus

# 恢复会话
python main.py --session abc12345
```

### 3.2 命令行参数

| 参数 | 简写 | 说明 | 示例 |
|------|------|------|------|
| `--provider` | `-p` | 指定 LLM 提供商 | `-p openai` |
| `--model` | `-m` | 指定模型名称 | `-m gpt-4o` |
| `--config` | `-c` | 指定配置文件路径 | `-c ./my-config.yaml` |
| `--session` | `-s` | 恢复指定会话 | `-s abc12345` |

### 3.3 启动界面

```
╔══════════════════════════════════════════╗
║         🤖 Easy Agent v0.1.0             ║
║    Hermes-style LangGraph Agent          ║
║    Type /help for commands, /quit to exit║
╚══════════════════════════════════════════╝

Provider: zhipu | Session: a1b2c3d4

👤 You:
```

---

## 4. 交互命令

### 4.1 斜杠命令

在对话中输入以下命令：

| 命令 | 说明 |
|------|------|
| `/help` 或 `/h` | 显示帮助信息 |
| `/quit` 或 `/q` | 退出 Agent |
| `/memory` | 查看当前 MEMORY.md 内容 |
| `/profile` | 查看当前 USER.md 内容 |
| `/clear` | 开始新会话 |

### 4.2 对话示例

```
👤 You: 你好

🤖 Agent: 你好！我是 Easy Agent，有什么可以帮助你的吗？

👤 You: 帮我列出当前目录的文件

🤖 Agent: [调用 list_dir 工具]
当前目录包含：
  src/
  tests/
  main.py
  config.yaml
  README.md
  requirements.txt

👤 You: 读取 config.yaml 的内容

🤖 Agent: [调用 read_file 工具]
config.yaml 内容如下：
providers:
  zhipu:
    type: openai_compatible
    ...

👤 You: 执行命令 echo hello

⚠️  Dangerous operation:
  Tool: run_shell
  Args: {'command': 'echo hello'}
  Approve? [y/n]: y

🤖 Agent: hello

👤 You: /quit
Goodbye!
```

### 4.3 危险操作确认

当 Agent 尝试执行危险操作时，会提示确认：

```
⚠️  Dangerous operation:
  Tool: run_shell
  Args: {'command': 'rm -rf /tmp/test'}
  Approve? [y/n]:
```

- 输入 `y` 确认执行
- 输入 `n` 或其他字符拒绝执行

---

## 5. Provider 配置

### 5.1 配置文件结构

`config.yaml` 文件结构：

```yaml
providers:
  <provider_name>:
    type: <provider_type>
    api_key: "${ENV_VAR}"        # 环境变量
    base_url: "<api_url>"        # API 地址
    default_model: "<model>"     # 默认模型
    timeout: 60                  # 超时时间（秒）

active_provider: "<provider_name>"      # 当前使用的 Provider
fallback_provider: "<provider_name>"    # 失败时回退的 Provider

agent:
  max_iterations: 15                    # 最大循环次数
  context_compression_threshold: 0.7    # 上下文压缩阈值
  memory_nudge_interval: 10             # 自我进化触发间隔

memory:
  data_dir: "~/.easy_agent"             # 数据存储目录
  memory_md_max_chars: 2000             # MEMORY.md 最大字符数
  user_md_max_chars: 1500               # USER.md 最大字符数
  fts5_retention_days: 90               # 历史记录保留天数
  vector_max_entries: 500               # 向量存储最大条目数
```

### 5.2 Provider 类型

| 类型 | 说明 | 适用场景 |
|------|------|----------|
| `openai` | OpenAI 官方 API | 使用 OpenAI 服务 |
| `openai_compatible` | 兼容 OpenAI API 的服务 | 智谱 AI、自定义代理等 |
| `anthropic` | Anthropic Claude API | 使用 Claude 模型 |

### 5.3 添加自定义代理

在 `config.yaml` 中添加：

```yaml
providers:
  # 自定义代理示例
  my_proxy:
    type: openai_compatible
    api_key: "${MY_PROXY_API_KEY}"
    base_url: "https://your-proxy-url.com/v1"
    default_model: "gpt-4o"
    timeout: 120

active_provider: "my_proxy"
```

然后设置环境变量：

```bash
export MY_PROXY_API_KEY="your-api-key"
python main.py --provider my_proxy
```

### 5.4 常用 Provider 配置

**智谱 AI**
```yaml
zhipu:
  type: openai_compatible
  api_key: "${ZHIPU_API_KEY}"
  base_url: "https://open.bigmodel.cn/api/paas/v4"
  default_model: "glm-4-plus"
```

**OpenAI**
```yaml
openai:
  type: openai
  api_key: "${OPENAI_API_KEY}"
  base_url: "https://api.openai.com/v1"
  default_model: "gpt-4o"
```

**Anthropic Claude**
```yaml
anthropic:
  type: anthropic
  api_key: "${ANTHROPIC_API_KEY}"
  default_model: "claude-sonnet-4-6"
```

---

## 6. 工具系统

### 6.1 内置工具列表

| 工具 | 风险等级 | 说明 |
|------|----------|------|
| `read_file` | safe | 读取文件内容 |
| `list_dir` | safe | 列出目录内容 |
| `get_time` | safe | 获取当前时间 |
| `memory_search` | safe | 搜索历史记忆 |
| `write_file` | write | 写入文件 |
| `todo_write` | write | 任务列表管理 |
| `web_search` | dangerous | 网页搜索（需确认） |
| `web_fetch` | dangerous | 获取网页内容（需确认） |
| `run_shell` | dangerous | 执行 Shell 命令（需确认） |

### 6.2 风险等级说明

| 等级 | 行为 |
|------|------|
| `safe` | 直接执行，无需确认 |
| `write` | 执行并记录日志 |
| `dangerous` | 需要用户确认后才执行 |

### 6.3 工具使用示例

**读取文件**
```
👤 You: 读取 /path/to/file.txt 的内容
🤖 Agent: [调用 read_file 工具]
```

**写入文件**
```
👤 You: 创建一个 hello.txt 文件，内容是 "Hello World"
🤖 Agent: [调用 write_file 工具]
Successfully wrote 11 chars to hello.txt
```

**执行命令**
```
👤 You: 执行 ls -la 命令
⚠️  Dangerous operation:
  Tool: run_shell
  Args: {'command': 'ls -la'}
  Approve? [y/n]: y
🤖 Agent: [命令输出结果]
```

---

## 7. 记忆系统

### 7.1 四层记忆架构

| 层级 | 存储 | 生命周期 | 用途 |
|------|------|----------|------|
| L1 | AgentState.messages | 单次会话 | 对话上下文 |
| L2 | MEMORY.md / USER.md | 永久 | Agent 知识 + 用户画像 |
| L3 | SQLite FTS5 | 90 天 | 全文检索历史对话 |
| L4 | FAISS 向量索引 | 500 条 | 语义相似搜索 |

### 7.2 数据存储位置

```
~/.easy_agent/
├── MEMORY.md          # Agent 学习的知识笔记
├── USER.md            # 用户画像和偏好
├── history.db         # 对话历史（SQLite + FTS5）
├── vectors/           # 向量索引（FAISS）
│   ├── faiss.index
│   └── meta.pkl
└── skills/            # 技能笔记
    └── YYYY-MM-DD-*.md
```

### 7.3 查看记忆内容

```bash
# 在 Agent 中查看
👤 You: /memory
MEMORY.md:
[Agent 学习的知识内容]

👤 You: /profile
USER.md:
[用户画像信息]
```

### 7.4 自我进化机制

每 N 轮对话后（默认 10 轮），Agent 会自动回顾对话：

- 发现用户偏好 → 更新 USER.md
- 学习新知识 → 更新 MEMORY.md
- 解决复杂问题 → 创建技能笔记

---

## 8. 常见问题

### Q1: 启动报错 "Provider 'xxx' not found in config"

**原因**：配置文件中没有该 Provider
**解决**：检查 `config.yaml` 中是否配置了该 Provider，或使用已配置的 Provider 名称

### Q2: 启动报错 "Config file not found"

**原因**：找不到配置文件
**解决**：确保在项目根目录运行，或使用 `-c` 指定配置文件路径

### Q3: API 调用失败

**原因**：API 密钥未设置或无效
**解决**：
```bash
# 检查环境变量
echo $ZHIPU_API_KEY

# 重新设置
export ZHIPU_API_KEY="your-api-key"
```

### Q4: 向量模型加载慢

**原因**：首次加载需要下载嵌入模型
**解决**：等待下载完成，后续会使用缓存

### Q5: 如何清除所有记忆数据

```bash
rm -rf ~/.easy_agent/
```

### Q6: 如何切换语言

Agent 会根据 Provider 返回的内容自动适配语言，无需手动配置。

---

## 附录：项目结构

```
easy_agent/
├── main.py              # CLI 入口
├── config.yaml          # 默认配置
├── requirements.txt     # 依赖列表
├── README.md            # 项目说明
├── USAGE.md             # 本使用说明书
├── src/
│   ├── __init__.py
│   ├── state.py         # AgentState 定义
│   ├── config.py        # 配置加载
│   ├── graph.py         # StateGraph 组装
│   ├── nodes/           # 图节点
│   │   ├── __init__.py
│   │   ├── agent_node.py
│   │   ├── tool_executor.py
│   │   ├── observe_node.py
│   │   ├── human_gate.py
│   │   ├── human_input.py
│   │   ├── memory_retrieve.py
│   │   ├── memory_save.py
│   │   └── nudge_check.py
│   ├── tools/           # 工具实现
│   │   ├── __init__.py
│   │   ├── file_tools.py
│   │   ├── web_tools.py
│   │   ├── system_tools.py
│   │   └── agent_tools.py
│   ├── memory/          # 记忆存储
│   │   ├── __init__.py
│   │   ├── file_memory.py
│   │   ├── fts5_store.py
│   │   └── vector_store.py
│   └── providers/       # Provider 适配
│       ├── __init__.py
│       ├── base.py
│       ├── openai_provider.py
│       ├── anthropic_provider.py
│       └── factory.py
└── tests/               # 测试套件
    ├── __init__.py
    ├── test_state.py
    ├── test_config.py
    ├── test_providers.py
    ├── test_tools.py
    ├── test_memory.py
    ├── test_nodes.py
    └── test_graph.py
```

---

**文档版本**：v1.0
**最后更新**：2026-04-30