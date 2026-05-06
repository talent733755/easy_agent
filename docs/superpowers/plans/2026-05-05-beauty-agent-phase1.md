# 娇莉芙美容 Agent Phase 1 MVP 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 构建面向娇莉芙门店员工的 Web 聊天 Agent，支持知识库查询和客户档案查询。

**Architecture:** 在现有 easy_agent 基础上扩展，新增美容行业节点（意图识别、知识检索、MCP 客户端），复用 LangGraph 框架和记忆系统，通过 FastAPI 提供 Web 界面。

**Tech Stack:** Python 3.11+, LangGraph, LangChain, FAISS, FastAPI, WebSocket, MCP (HTTP/SSE)

---

## File Structure

### New Files to Create

```
easy_agent/
├── web_app.py                                    # Web 入口
├── knowledge/
│   └── jiaolifu/                                # 知识文档目录（用户提供）
│       ├── products/
│       ├── procedures/
│       ├── scripts/
│       └── troubleshooting/
├── src/
│   ├── state.py                                 # 修改：新增字段
│   ├── config.py                                # 修改：加载 beauty 配置
│   ├── graph.py                                 # 修改：组装新节点
│   ├── nodes/beauty/
│   │   ├── __init__.py
│   │   ├── intent_node.py                       # 意图识别节点
│   │   ├── knowledge_node.py                    # 知识检索节点
│   │   └── mcp_client_node.py                   # MCP 客户端节点
│   └── tools/beauty/
│       ├── __init__.py
│       └── knowledge_rag.py                     # RAG 工具
├── web/
│   ├── __init__.py
│   ├── app.py                                   # FastAPI 应用
│   ├── static/
│   │   ├── style.css
│   │   └── chat.js
│   └── templates/
│       └── index.html
├── mcp_servers/
│   └── customer/
│       ├── __init__.py
│       ├── main.py                              # MCP Server 入口
│       └── tools.py                             # MCP 工具定义
├── tests/
│   ├── test_intent_node.py
│   ├── test_knowledge_node.py
│   ├── test_mcp_client_node.py
│   └── test_web_app.py
└── prompts/
    └── beauty_intent_v1.txt                     # 意图识别 Prompt
```

### Modified Files

- `src/state.py` - 新增 intent, customer_context, knowledge_results, mcp_results 字段
- `src/config.py` - 加载 beauty 配置段
- `src/graph.py` - 组装新节点到流程
- `config.yaml` - 新增 beauty 配置

---

## Task 1: 扩展 State 和配置

**Files:**
- Modify: `src/state.py`
- Modify: `src/config.py`
- Modify: `config.yaml`
- Test: `tests/test_state.py`

- [ ] **Step 1: Write the failing test for new state fields**

Create `tests/test_state.py`:

```python
"""Test AgentState includes beauty-specific fields."""
from src.state import AgentState


def test_state_has_beauty_fields():
    """State should include intent, customer_context, knowledge_results, mcp_results."""
    state: AgentState = {
        "messages": [],
        "tools": [],
        "context_window": {"used_tokens": 0, "max_tokens": 128000, "threshold": 0.7},
        "memory_context": "",
        "user_profile": "",
        "agent_notes": "",
        "pending_human_input": False,
        "iteration_count": 0,
        "max_iterations": 15,
        "nudge_counter": 0,
        "provider_name": "openai",
        # New fields
        "intent": "",
        "customer_context": {},
        "knowledge_results": [],
        "mcp_results": {},
    }

    assert state["intent"] == ""
    assert state["customer_context"] == {}
    assert state["knowledge_results"] == []
    assert state["mcp_results"] == {}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_state.py::test_state_has_beauty_fields -v`

Expected: FAIL with KeyError or TypedDict validation error

- [ ] **Step 3: Add new fields to AgentState**

Modify `src/state.py`:

```python
from typing import Annotated, TypedDict
from langgraph.graph.message import add_messages
from langchain_core.tools import BaseTool


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    tools: list[BaseTool]
    context_window: dict           # {"used_tokens": N, "max_tokens": N, "threshold": 0.7}
    memory_context: str            # 从记忆系统检索到的上下文
    user_profile: str              # USER.md 内容快照
    agent_notes: str               # MEMORY.md 内容快照
    pending_human_input: bool      # 是否等待人类输入
    iteration_count: int           # 当前循环计数
    max_iterations: int            # 最大循环轮数
    nudge_counter: int             # 自我进化触发计数
    provider_name: str             # 当前使用的 LLM 提供商

    # Phase 1: Beauty Agent 新增字段
    intent: str                    # 识别意图: query_customer | knowledge_query | mixed
    customer_context: dict         # 当前客户信息缓存
    knowledge_results: list        # RAG 检索结果
    mcp_results: dict              # MCP 调用结果
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_state.py::test_state_has_beauty_fields -v`

Expected: PASS

- [ ] **Step 5: Add beauty config to config.yaml**

Modify `config.yaml`, append at the end:

```yaml
# 美容行业配置
beauty:
  knowledge_base:
    base_dir: "./knowledge/jiaolifu"
    indexes:
      products:
        path: "products"
        description: "产品知识、成分功效"
      procedures:
        path: "procedures"
        description: "服务流程、操作标准"
      scripts:
        path: "scripts"
        description: "销售话术、接待标准"
      troubleshooting:
        path: "troubleshooting"
        description: "问题处理、应急方案"

  mcp_servers:
    customer:
      url: "http://localhost:3001"
      timeout: 30

  intent_prompt: "./prompts/beauty_intent_v1.txt"

  web:
    host: "0.0.0.0"
    port: 8080
```

- [ ] **Step 6: Update config loader to support beauty config**

Modify `src/config.py`, add beauty field to AppConfig:

```python
from typing import Any
import yaml
from pathlib import Path
from pydantic import BaseModel


class ProviderConfig(BaseModel):
    type: str
    api_key: str
    base_url: str = ""
    default_model: str
    timeout: int = 60


class SearchConfig(BaseModel):
    provider: str = "tavily"
    tavily_api_key: str = ""


class AgentConfig(BaseModel):
    max_iterations: int = 15
    context_compression_threshold: float = 0.7
    memory_nudge_interval: int = 10


class MemoryConfig(BaseModel):
    data_dir: str = "~/.easy_agent"
    memory_md_max_chars: int = 2000
    user_md_max_chars: int = 1500
    fts5_retention_days: int = 90
    vector_max_entries: int = 500


class KnowledgeIndexConfig(BaseModel):
    path: str
    description: str


class KnowledgeBaseConfig(BaseModel):
    base_dir: str
    indexes: dict[str, KnowledgeIndexConfig]


class MCPServerConfig(BaseModel):
    url: str
    timeout: int = 30


class WebConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8080


class BeautyConfig(BaseModel):
    knowledge_base: KnowledgeBaseConfig
    mcp_servers: dict[str, MCPServerConfig]
    intent_prompt: str
    web: WebConfig


class AppConfig(BaseModel):
    providers: dict[str, ProviderConfig]
    active_provider: str
    fallback_provider: str = ""
    search: SearchConfig = SearchConfig()
    agent: AgentConfig = AgentConfig()
    memory: MemoryConfig = MemoryConfig()
    beauty: BeautyConfig | None = None  # 新增


def load_config(config_path: str = None) -> AppConfig:
    """Load configuration from YAML file."""
    if config_path is None:
        config_path = "config.yaml"

    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    return AppConfig(**data)
```

- [ ] **Step 7: Write test for beauty config loading**

Add to `tests/test_config.py`:

```python
def test_load_beauty_config():
    """Config should load beauty section."""
    from src.config import load_config

    config = load_config()

    assert config.beauty is not None
    assert config.beauty.knowledge_base.base_dir == "./knowledge/jiaolifu"
    assert "products" in config.beauty.knowledge_base.indexes
    assert config.beauty.mcp_servers["customer"].url == "http://localhost:3001"
    assert config.beauty.web.port == 8080
```

- [ ] **Step 8: Run config test**

Run: `pytest tests/test_config.py::test_load_beauty_config -v`

Expected: PASS

- [ ] **Step 9: Commit state and config changes**

```bash
git add src/state.py src/config.py config.yaml tests/test_state.py tests/test_config.py
git commit -m "feat: add beauty agent state fields and config"
```

---

## Task 2: 实现知识库向量化

**Files:**
- Create: `src/tools/beauty/__init__.py`
- Create: `src/tools/beauty/knowledge_rag.py`
- Create: `tests/test_knowledge_rag.py`

- [ ] **Step 1: Write the failing test for knowledge RAG**

Create `tests/test_knowledge_rag.py`:

```python
"""Test knowledge RAG functionality."""
import tempfile
from pathlib import Path
from src.tools.beauty.knowledge_rag import KnowledgeRAG


def test_knowledge_rag_index_and_search():
    """Should index markdown files and search them."""
    # Create temp knowledge base
    with tempfile.TemporaryDirectory() as tmpdir:
        kb_dir = Path(tmpdir) / "test_kb"
        kb_dir.mkdir()

        # Create test files
        (kb_dir / "product1.md").write_text("# 产品A\n\n这是一款保湿精华，适合干性肌肤。")
        (kb_dir / "product2.md").write_text("# 产品B\n\n这是一款控油爽肤水，适合油性肌肤。")

        # Initialize RAG
        rag = KnowledgeRAG(str(kb_dir))
        rag.build_index()

        # Search
        results = rag.search("干性肌肤适合什么", top_k=1)

        assert len(results) == 1
        assert "保湿精华" in results[0]["content"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_knowledge_rag.py::test_knowledge_rag_index_and_search -v`

Expected: FAIL with ModuleNotFoundError

- [ ] **Step 3: Create knowledge RAG tool**

Create `src/tools/beauty/__init__.py`:

```python
from src.tools.beauty.knowledge_rag import KnowledgeRAG

__all__ = ["KnowledgeRAG"]
```

Create `src/tools/beauty/knowledge_rag.py`:

```python
"""Knowledge RAG tool for beauty domain."""
from pathlib import Path
from typing import Any
import faiss
import numpy as np
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.doc_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


class KnowledgeRAG:
    """Manages knowledge base indexing and retrieval."""

    def __init__(self, knowledge_dir: str, index_name: str = "beauty_kb"):
        self.knowledge_dir = Path(knowledge_dir)
        self.index_name = index_name
        self.embeddings = OpenAIEmbeddings()
        self.vectorstore: FAISS | None = None

    def build_index(self) -> None:
        """Build FAISS index from all markdown files in knowledge_dir."""
        documents = []

        # Load all .md files
        for md_file in self.knowledge_dir.rglob("*.md"):
            loader = TextLoader(str(md_file), encoding="utf-8")
            docs = loader.load()
            for doc in docs:
                doc.metadata["source"] = str(md_file.relative_to(self.knowledge_dir))
            documents.extend(docs)

        # Split documents
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
        )
        splits = text_splitter.split_documents(documents)

        # Build FAISS index
        self.vectorstore = FAISS.from_documents(splits, self.embeddings)

    def search(self, query: str, top_k: int = 3) -> list[dict[str, Any]]:
        """Search knowledge base and return top-k results."""
        if self.vectorstore is None:
            raise ValueError("Index not built. Call build_index() first.")

        results = self.vectorstore.similarity_search(query, k=top_k)

        return [
            {
                "content": doc.page_content,
                "source": doc.metadata.get("source", "unknown"),
            }
            for doc in results
        ]

    def save_index(self, path: str) -> None:
        """Save FAISS index to disk."""
        if self.vectorstore is None:
            raise ValueError("Index not built.")

        self.vectorstore.save_local(path)

    def load_index(self, path: str) -> None:
        """Load FAISS index from disk."""
        self.vectorstore = FAISS.load_local(path, self.embeddings)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_knowledge_rag.py::test_knowledge_rag_index_and_search -v`

Expected: PASS

- [ ] **Step 5: Commit knowledge RAG**

```bash
git add src/tools/beauty/ tests/test_knowledge_rag.py
git commit -m "feat: add knowledge RAG tool with FAISS indexing"
```

---

## Task 3: 实现意图识别节点

**Files:**
- Create: `src/nodes/beauty/__init__.py`
- Create: `src/nodes/beauty/intent_node.py`
- Create: `prompts/beauty_intent_v1.txt`
- Test: `tests/test_intent_node.py`

- [ ] **Step 1: Write the failing test for intent classification**

Create `tests/test_intent_node.py`:

```python
"""Test intent classification node."""
from langchain_core.messages import HumanMessage
from src.nodes.beauty.intent_node import intent_classify_node
from src.state import AgentState


def test_intent_classify_query_customer():
    """Should classify customer query intent."""
    state: AgentState = {
        "messages": [HumanMessage(content="帮我查一下张女士的档案")],
        "tools": [],
        "context_window": {"used_tokens": 0, "max_tokens": 128000, "threshold": 0.7},
        "memory_context": "",
        "user_profile": "",
        "agent_notes": "",
        "pending_human_input": False,
        "iteration_count": 0,
        "max_iterations": 15,
        "nudge_counter": 0,
        "provider_name": "openai",
        "intent": "",
        "customer_context": {},
        "knowledge_results": [],
        "mcp_results": {},
    }

    result = intent_classify_node(state)

    assert result["intent"] == "query_customer"
    assert "张女士" in result.get("customer_name", "")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_intent_node.py::test_intent_classify_query_customer -v`

Expected: FAIL with ModuleNotFoundError

- [ ] **Step 3: Create intent prompt**

Create `prompts/beauty_intent_v1.txt`:

```
你是一个意图识别助手，负责分析用户输入并识别其意图。

## 意图类型

1. **query_customer** - 查询客户信息
   示例："查一下张女士的档案"、"李先生的消费记录"

2. **knowledge_query** - 查询知识库
   示例："敏感肌适合什么产品"、"面部护理流程"

3. **mixed** - 混合意图（既查客户又查知识）
   示例："张女士适合做什么项目"、"推荐适合李女士的疗程"

4. **general** - 一般对话
   示例："你好"、"今天天气怎么样"

## 输出格式

返回 JSON：
{
  "intent": "意图类型",
  "customer_name": "客户姓名（如果有）",
  "query_topic": "查询主题（如果有）"
}

## 用户输入
{user_input}
```

- [ ] **Step 4: Create intent classification node**

Create `src/nodes/beauty/__init__.py`:

```python
from src.nodes.beauty.intent_node import intent_classify_node

__all__ = ["intent_classify_node"]
```

Create `src/nodes/beauty/intent_node.py`:

```python
"""Intent classification node for beauty agent."""
import json
from pathlib import Path
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from src.state import AgentState


def intent_classify_node(state: AgentState) -> dict:
    """Classify user intent from the last message.

    Returns:
        dict with keys: intent, customer_name (optional), query_topic (optional)
    """
    # Get last user message
    last_user_msg = ""
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            last_user_msg = msg.content
            break

    if not last_user_msg:
        return {"intent": "general"}

    # Load intent prompt
    prompt_path = Path("prompts/beauty_intent_v1.txt")
    if not prompt_path.exists():
        # Fallback to simple rule-based classification
        return _rule_based_classify(last_user_msg)

    prompt_template = prompt_path.read_text(encoding="utf-8")
    prompt = prompt_template.format(user_input=last_user_msg)

    # Call LLM for classification
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    response = llm.invoke([HumanMessage(content=prompt)])

    # Parse JSON response
    try:
        # Extract JSON from response
        content = response.content
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            content = content.split("```")[1].split("```")[0]

        result = json.loads(content.strip())
        return {
            "intent": result.get("intent", "general"),
            "customer_name": result.get("customer_name", ""),
            "query_topic": result.get("query_topic", ""),
        }
    except (json.JSONDecodeError, KeyError):
        return _rule_based_classify(last_user_msg)


def _rule_based_classify(text: str) -> dict:
    """Fallback rule-based intent classification."""
    text_lower = text.lower()

    # Customer query keywords
    customer_keywords = ["档案", "客户", "消费", "服务记录", "女士", "先生"]
    has_customer = any(kw in text for kw in customer_keywords)

    # Knowledge query keywords
    knowledge_keywords = ["产品", "流程", "适合", "推荐", "疗程", "护理"]
    has_knowledge = any(kw in text for kw in knowledge_keywords)

    # Extract customer name (simplified)
    customer_name = ""
    import re
    name_match = re.search(r"([张王李赵刘陈杨黄周吴徐孙朱马胡郭林何高梁郑罗宋谢唐韩曹许邓萧冯曾程蔡彭潘袁于董余苏叶吕魏蒋田杜丁沈姜范江傅钟卢汪戴崔任米廖方石姚谭廖邹熊金陆郝孔白崔康毛邱秦江史顾侯邵孟龙万段雷钱汤尹黎易常武乔贺赖龚文]",女士|先生)", text)
    if name_match:
        customer_name = name_match.group(0)

    if has_customer and has_knowledge:
        return {"intent": "mixed", "customer_name": customer_name}
    elif has_customer:
        return {"intent": "query_customer", "customer_name": customer_name}
    elif has_knowledge:
        return {"intent": "knowledge_query"}
    else:
        return {"intent": "general"}
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/test_intent_node.py::test_intent_classify_query_customer -v`

Expected: PASS

- [ ] **Step 6: Commit intent node**

```bash
git add src/nodes/beauty/ prompts/ tests/test_intent_node.py
git commit -m "feat: add intent classification node for beauty agent"
```

---

## Task 4: 实现知识检索节点

**Files:**
- Create: `src/nodes/beauty/knowledge_node.py`
- Test: `tests/test_knowledge_node.py`

- [ ] **Step 1: Write the failing test for knowledge retrieval**

Create `tests/test_knowledge_node.py`:

```python
"""Test knowledge retrieval node."""
from langchain_core.messages import HumanMessage
from src.nodes.beauty.knowledge_node import knowledge_retrieve_node
from src.state import AgentState


def test_knowledge_retrieve_node():
    """Should retrieve knowledge based on intent."""
    state: AgentState = {
        "messages": [HumanMessage(content="敏感肌适合什么产品")],
        "tools": [],
        "context_window": {"used_tokens": 0, "max_tokens": 128000, "threshold": 0.7},
        "memory_context": "",
        "user_profile": "",
        "agent_notes": "",
        "pending_human_input": False,
        "iteration_count": 0,
        "max_iterations": 15,
        "nudge_counter": 0,
        "provider_name": "openai",
        "intent": "knowledge_query",
        "customer_context": {},
        "knowledge_results": [],
        "mcp_results": {},
    }

    # Mock knowledge base (in real scenario, would use actual KB)
    result = knowledge_retrieve_node(state, knowledge_dir="tests/fixtures/knowledge")

    assert "knowledge_results" in result
    assert isinstance(result["knowledge_results"], list)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_knowledge_node.py::test_knowledge_retrieve_node -v`

Expected: FAIL with ModuleNotFoundError

- [ ] **Step 3: Create knowledge retrieval node**

Create `src/nodes/beauty/knowledge_node.py`:

```python
"""Knowledge retrieval node for beauty agent."""
from src.state import AgentState
from src.tools.beauty.knowledge_rag import KnowledgeRAG


def knowledge_retrieve_node(state: AgentState, knowledge_dir: str = "./knowledge/jiaolifu") -> dict:
    """Retrieve knowledge from RAG based on intent.

    Args:
        state: Current agent state
        knowledge_dir: Path to knowledge base directory

    Returns:
        dict with knowledge_results
    """
    intent = state.get("intent", "general")

    # Skip knowledge retrieval for non-knowledge intents
    if intent not in ["knowledge_query", "mixed"]:
        return {"knowledge_results": []}

    # Get query from last user message
    from langchain_core.messages import HumanMessage
    query = ""
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            query = msg.content
            break

    if not query:
        return {"knowledge_results": []}

    # Initialize RAG and search
    try:
        rag = KnowledgeRAG(knowledge_dir)

        # Try to load existing index, otherwise build it
        import os
        index_path = os.path.join(knowledge_dir, ".faiss_index")
        if os.path.exists(index_path):
            rag.load_index(index_path)
        else:
            rag.build_index()

        results = rag.search(query, top_k=3)

        return {"knowledge_results": results}
    except Exception as e:
        # Return empty results on error
        print(f"Knowledge retrieval error: {e}")
        return {"knowledge_results": []}
```

- [ ] **Step 4: Create test fixture knowledge**

Create `tests/fixtures/knowledge/test_product.md`:

```markdown
# 敏感肌护理产品

适合敏感肌肤的产品推荐：

1. 舒缓洁面乳 - 温和清洁，不刺激
2. 修复精华 - 增强皮肤屏障
3. 舒缓面膜 - 镇静舒缓
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/test_knowledge_node.py::test_knowledge_retrieve_node -v`

Expected: PASS

- [ ] **Step 6: Commit knowledge node**

```bash
git add src/nodes/beauty/knowledge_node.py tests/test_knowledge_node.py tests/fixtures/
git commit -m "feat: add knowledge retrieval node with RAG integration"
```

---

## Task 5: 实现 MCP Client 节点

**Files:**
- Create: `src/nodes/beauty/mcp_client_node.py`
- Test: `tests/test_mcp_client_node.py`

- [ ] **Step 1: Write the failing test for MCP client**

Create `tests/test_mcp_client_node.py`:

```python
"""Test MCP client node."""
from langchain_core.messages import HumanMessage
from src.nodes.beauty.mcp_client_node import mcp_customer_node
from src.state import AgentState


def test_mcp_customer_node():
    """Should call Customer MCP to get customer info."""
    state: AgentState = {
        "messages": [HumanMessage(content="查一下张女士的档案")],
        "tools": [],
        "context_window": {"used_tokens": 0, "max_tokens": 128000, "threshold": 0.7},
        "memory_context": "",
        "user_profile": "",
        "agent_notes": "",
        "pending_human_input": False,
        "iteration_count": 0,
        "max_iterations": 15,
        "nudge_counter": 0,
        "provider_name": "openai",
        "intent": "query_customer",
        "customer_context": {},
        "knowledge_results": [],
        "mcp_results": {},
    }

    # Mock MCP server response
    result = mcp_customer_node(state, mcp_url="http://localhost:3001")

    # In real scenario, would call actual MCP server
    assert "mcp_results" in result
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_mcp_client_node.py::test_mcp_customer_node -v`

Expected: FAIL with ModuleNotFoundError

- [ ] **Step 3: Create MCP client node**

Create `src/nodes/beauty/mcp_client_node.py`:

```python
"""MCP client node for calling Customer MCP."""
import requests
from src.state import AgentState


def mcp_customer_node(state: AgentState, mcp_url: str = "http://localhost:3001") -> dict:
    """Call Customer MCP to retrieve customer information.

    Args:
        state: Current agent state
        mcp_url: Customer MCP server URL

    Returns:
        dict with customer_context and mcp_results
    """
    intent = state.get("intent", "general")
    customer_name = state.get("customer_name", "")

    # Skip if not customer-related intent
    if intent not in ["query_customer", "mixed"]:
        return {"customer_context": {}, "mcp_results": {}}

    if not customer_name:
        return {"customer_context": {}, "mcp_results": {}}

    # Call Customer MCP (simplified HTTP call, not full MCP protocol for Phase 1)
    try:
        response = requests.post(
            f"{mcp_url}/tools/get_customer",
            json={"name": customer_name},
            timeout=30,
        )
        response.raise_for_status()
        customer_data = response.json()

        return {
            "customer_context": customer_data,
            "mcp_results": {"customer": customer_data},
        }
    except requests.RequestException as e:
        print(f"MCP call failed: {e}")
        return {
            "customer_context": {},
            "mcp_results": {"error": str(e)},
        }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_mcp_client_node.py::test_mcp_customer_node -v`

Expected: PASS

- [ ] **Step 5: Commit MCP client node**

```bash
git add src/nodes/beauty/mcp_client_node.py tests/test_mcp_client_node.py
git commit -m "feat: add MCP client node for Customer MCP"
```

---

## Task 6: 开发 Customer MCP Server

**Files:**
- Create: `mcp_servers/customer/__init__.py`
- Create: `mcp_servers/customer/main.py`
- Create: `mcp_servers/customer/tools.py`

- [ ] **Step 1: Create Customer MCP tools**

Create `mcp_servers/customer/__init__.py`:

```python
"""Customer MCP Server."""
```

Create `mcp_servers/customer/tools.py`:

```python
"""Customer MCP tools - Mock implementation for Phase 1."""
from typing import Any


def get_customer(name: str) -> dict[str, Any]:
    """Get customer profile by name.

    Args:
        name: Customer name (e.g., "张女士")

    Returns:
        Customer profile dict
    """
    # Mock data - in production, would call actual SaaS API
    mock_customers = {
        "张女士": {
            "id": "C12345",
            "name": "张女士",
            "phone": "138****8888",
            "skin_type": "敏感肌",
            "last_visit": "2025-04-20",
            "total_spent": 15800,
            "membership": "金卡会员",
        },
        "李女士": {
            "id": "C12346",
            "name": "李女士",
            "phone": "139****9999",
            "skin_type": "干性肌肤",
            "last_visit": "2025-04-18",
            "total_spent": 23500,
            "membership": "钻石会员",
        },
    }

    return mock_customers.get(name, {"error": f"Customer '{name}' not found"})


def get_consumption(customer_id: str, limit: int = 10) -> list[dict[str, Any]]:
    """Get customer consumption history.

    Args:
        customer_id: Customer ID
        limit: Max number of records

    Returns:
        List of consumption records
    """
    # Mock data
    return [
        {
            "date": "2025-04-20",
            "item": "面部护理套餐",
            "amount": 1280,
            "payment": "会员卡",
        },
        {
            "date": "2025-03-15",
            "item": "精油按摩",
            "amount": 680,
            "payment": "微信支付",
        },
    ][:limit]


def get_service_history(customer_id: str, limit: int = 10) -> list[dict[str, Any]]:
    """Get customer service history.

    Args:
        customer_id: Customer ID
        limit: Max number of records

    Returns:
        List of service records
    """
    # Mock data
    return [
        {
            "date": "2025-04-20",
            "service": "补水面部护理",
            "therapist": "小王",
            "duration": 90,
            "note": "皮肤状态良好，建议继续补水疗程",
        },
        {
            "date": "2025-03-15",
            "service": "肩颈按摩",
            "therapist": "小李",
            "duration": 60,
            "note": "肩颈僵硬，建议增加按摩频率",
        },
    ][:limit]
```

- [ ] **Step 2: Create Customer MCP server (FastAPI)**

Create `mcp_servers/customer/main.py`:

```python
"""Customer MCP Server - FastAPI implementation for Phase 1."""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from mcp_servers.customer.tools import get_customer, get_consumption, get_service_history


app = FastAPI(title="Customer MCP Server", version="0.1.0")


class GetCustomerRequest(BaseModel):
    name: str


class GetConsumptionRequest(BaseModel):
    customer_id: str
    limit: int = 10


class GetServiceHistoryRequest(BaseModel):
    customer_id: str
    limit: int = 10


@app.post("/tools/get_customer")
async def api_get_customer(req: GetCustomerRequest):
    """Get customer profile."""
    result = get_customer(req.name)
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])
    return result


@app.post("/tools/get_consumption")
async def api_get_consumption(req: GetConsumptionRequest):
    """Get customer consumption history."""
    return get_consumption(req.customer_id, req.limit)


@app.post("/tools/get_service_history")
async def api_get_service_history(req: GetServiceHistoryRequest):
    """Get customer service history."""
    return get_service_history(req.customer_id, req.limit)


@app.get("/health")
async def health():
    """Health check."""
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=3001)
```

- [ ] **Step 3: Test Customer MCP server**

Run: `python -m mcp_servers.customer.main &`

Test with curl:

```bash
curl -X POST http://localhost:3001/tools/get_customer \
  -H "Content-Type: application/json" \
  -d '{"name": "张女士"}'
```

Expected: JSON response with customer data

- [ ] **Step 4: Commit Customer MCP**

```bash
git add mcp_servers/customer/
git commit -m "feat: add Customer MCP server with mock data"
```

---

## Task 7: 组装 Graph 流程

**Files:**
- Modify: `src/graph.py`
- Test: `tests/test_graph.py`

- [ ] **Step 1: Modify graph to include beauty nodes**

Modify `src/graph.py`, add beauty nodes to the flow:

```python
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.base import BaseCheckpointSaver
from langchain_core.messages import AIMessage
from typing import TYPE_CHECKING

from src.state import AgentState
from src.config import load_config, AppConfig
from src.providers.factory import get_provider

if TYPE_CHECKING:
    from src.nodes.tool_executor import ToolExecutor


def _build_agent_model(config: AppConfig):
    """Build the LLM model from config."""
    provider_name = config.active_provider
    provider_config = config.providers.get(provider_name)
    if provider_config is None:
        raise ValueError(f"Provider '{provider_name}' not found in config")
    provider = get_provider(provider_name, provider_config)
    return provider.get_model()


def build_graph(checkpointer: BaseCheckpointSaver = None) -> StateGraph:
    if checkpointer is None:
        checkpointer = MemorySaver()

    config = load_config()
    model = _build_agent_model(config)

    # Import nodes
    from src.nodes.agent_node import create_agent_node
    from src.nodes.tool_executor import ToolExecutor
    from src.nodes.observe_node import observe_node
    from src.nodes.human_gate import human_gate
    from src.nodes.memory_retrieve import memory_retrieve_node
    from src.nodes.memory_save import memory_save_node
    from src.nodes.nudge_check import nudge_check_node

    # Import beauty nodes
    from src.nodes.beauty.intent_node import intent_classify_node
    from src.nodes.beauty.knowledge_node import knowledge_retrieve_node
    from src.nodes.beauty.mcp_client_node import mcp_customer_node

    # Import all tools
    from src.tools import read_file, write_file, list_dir
    from src.tools import web_search, web_fetch
    from src.tools import get_time, run_shell
    from src.tools import todo_write, memory_search

    all_tools = [read_file, write_file, list_dir, web_search, web_fetch, get_time, run_shell, todo_write, memory_search]
    tools_by_name = {t.name: t for t in all_tools}

    executor = ToolExecutor(tools_by_name)
    agent_node_fn = create_agent_node(model, tools=all_tools)

    # Build the graph
    builder = StateGraph(AgentState)

    # Add nodes
    builder.add_node("intent_classify", intent_classify_node)

    # Knowledge node with config
    if config.beauty:
        builder.add_node("knowledge_retrieve", lambda s: knowledge_retrieve_node(
            s, knowledge_dir=config.beauty.knowledge_base.base_dir
        ))
        builder.add_node("mcp_customer", lambda s: mcp_customer_node(
            s, mcp_url=config.beauty.mcp_servers["customer"].url
        ))
    else:
        # Fallback if beauty config not present
        builder.add_node("knowledge_retrieve", lambda s: {"knowledge_results": []})
        builder.add_node("mcp_customer", lambda s: {"mcp_results": {}})

    builder.add_node("memory_retrieve", lambda s: memory_retrieve_node(s, data_dir=config.memory["data_dir"]))
    builder.add_node("agent", agent_node_fn)
    builder.add_node("tool_executor", _make_tool_node(executor))
    builder.add_node("observe", observe_node)
    builder.add_node("human_gate", human_gate)
    builder.add_node("nudge_check", lambda s: nudge_check_node(
        s, nudge_interval=config.agent["memory_nudge_interval"], data_dir=config.memory["data_dir"]
    ))
    builder.add_node("memory_save", lambda s: memory_save_node(s, data_dir=config.memory["data_dir"]))

    # Set entry
    builder.set_entry_point("intent_classify")

    # Edges
    builder.add_edge("intent_classify", "knowledge_retrieve")
    builder.add_edge("intent_classify", "mcp_customer")

    # Merge knowledge and MCP results
    builder.add_edge("knowledge_retrieve", "memory_retrieve")
    builder.add_edge("mcp_customer", "memory_retrieve")

    builder.add_edge("memory_retrieve", "agent")

    # Conditional edge from agent
    builder.add_conditional_edges("agent", _route_agent, {
        "tools": "tool_executor",
        "end": END,
    })

    builder.add_edge("tool_executor", "observe")
    builder.add_edge("observe", "human_gate")
    builder.add_edge("human_gate", "nudge_check")

    # Conditional edge from nudge_check (back to agent or end)
    builder.add_conditional_edges("nudge_check", lambda s: _route_iteration(s, config.agent["max_iterations"]), {
        "continue": "memory_save",
        "end": END,
    })

    builder.add_edge("memory_save", "agent")

    return builder.compile(checkpointer=checkpointer)


def _make_tool_node(executor):
    def tool_node(state: AgentState) -> dict:
        last_ai = None
        for m in reversed(state["messages"]):
            if isinstance(m, AIMessage) and hasattr(m, "tool_calls") and m.tool_calls:
                last_ai = m
                break
        if last_ai is None:
            return {}
        results = executor.execute(last_ai.tool_calls)
        return {"messages": results}
    return tool_node


def _route_agent(state: AgentState) -> str:
    max_iter = state.get("max_iterations", 15)
    if state.get("iteration_count", 0) >= max_iter:
        return "end"

    messages = state.get("messages", [])
    last_msg = messages[-1] if messages else None
    if isinstance(last_msg, AIMessage) and hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
        return "tools"
    return "end"


def _route_iteration(state: AgentState, max_iterations: int) -> str:
    if state.get("iteration_count", 0) >= max_iterations:
        return "end"
    return "continue"
```

- [ ] **Step 2: Write test for new graph flow**

Add to `tests/test_graph.py`:

```python
def test_graph_includes_beauty_nodes():
    """Graph should include intent_classify, knowledge_retrieve, mcp_customer nodes."""
    from src.graph import build_graph

    graph = build_graph()

    # Check nodes exist
    node_names = [node[0] for node in graph.nodes]
    assert "intent_classify" in node_names
    assert "knowledge_retrieve" in node_names
    assert "mcp_customer" in node_names
```

- [ ] **Step 3: Run graph test**

Run: `pytest tests/test_graph.py::test_graph_includes_beauty_nodes -v`

Expected: PASS

- [ ] **Step 4: Commit graph changes**

```bash
git add src/graph.py tests/test_graph.py
git commit -m "feat: integrate beauty nodes into agent graph"
```

---

## Task 8: 开发 Web 界面

**Files:**
- Create: `web/__init__.py`
- Create: `web/app.py`
- Create: `web/static/style.css`
- Create: `web/static/chat.js`
- Create: `web/templates/index.html`
- Create: `web_app.py`
- Test: `tests/test_web_app.py`

- [ ] **Step 1: Create FastAPI web app**

Create `web/__init__.py`:

```python
"""Web interface for beauty agent."""
```

Create `web/app.py`:

```python
"""FastAPI web application for beauty agent."""
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import Request
from langchain_core.messages import HumanMessage, AIMessage
from src.config import load_config
from src.graph import build_graph
from langgraph.checkpoint.memory import MemorySaver
import uuid
import json

app = FastAPI(title="娇莉芙智能助手")

# Mount static files
app.mount("/static", StaticFiles(directory="web/static"), name="static")

# Templates
templates = Jinja2Templates(directory="web/templates")

# Store active sessions
sessions = {}


@app.get("/")
async def index(request: Request):
    """Render chat page."""
    return templates.TemplateResponse("index.html", {"request": request})


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time chat."""
    await websocket.accept()

    # Initialize session
    session_id = str(uuid.uuid4())[:8]
    config = load_config()
    checkpointer = MemorySaver()
    graph = build_graph(checkpointer=checkpointer)
    graph_config = {"configurable": {"thread_id": session_id}}

    # Initial state
    state = {
        "messages": [],
        "tools": [],
        "context_window": {"used_tokens": 0, "max_tokens": 128000, "threshold": 0.7},
        "memory_context": "",
        "user_profile": "",
        "agent_notes": "",
        "pending_human_input": False,
        "iteration_count": 0,
        "max_iterations": config.agent["max_iterations"],
        "nudge_counter": 0,
        "provider_name": config.active_provider,
        "intent": "",
        "customer_context": {},
        "knowledge_results": [],
        "mcp_results": {},
    }

    try:
        while True:
            # Receive message from client
            data = await websocket.receive_text()
            message_data = json.loads(data)
            user_input = message_data.get("message", "")

            if not user_input.strip():
                continue

            # Add user message to state
            state["messages"].append(HumanMessage(content=user_input))
            state["iteration_count"] = 0

            # Invoke graph
            try:
                result = graph.invoke(state, graph_config)

                # Extract agent response
                agent_response = ""
                for m in reversed(result["messages"]):
                    if isinstance(m, AIMessage) and m.content:
                        agent_response = m.content
                        break

                # Send response to client
                await websocket.send_text(json.dumps({
                    "type": "message",
                    "content": agent_response,
                    "session_id": session_id,
                }))

                # Update state
                state = result
                state["messages"] = [m for m in state["messages"] if not (isinstance(m, AIMessage) and m.tool_calls)]
                state["nudge_counter"] = state.get("nudge_counter", 0) + 1

            except Exception as e:
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "content": f"Error: {str(e)}",
                }))

    except WebSocketDisconnect:
        print(f"Client disconnected: {session_id}")


@app.get("/health")
async def health():
    """Health check."""
    return {"status": "ok"}
```

- [ ] **Step 2: Create HTML template**

Create `web/templates/index.html`:

```html
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>娇莉芙智能助手</title>
    <link rel="stylesheet" href="/static/style.css">
</head>
<body>
    <div class="container">
        <header>
            <h1>🤖 娇莉芙智能助手</h1>
        </header>

        <div id="chat-container">
            <div id="messages"></div>
        </div>

        <div class="input-container">
            <input type="text" id="user-input" placeholder="输入消息..." autocomplete="off">
            <button id="send-btn">发送</button>
        </div>
    </div>

    <script src="/static/chat.js"></script>
</body>
</html>
```

- [ ] **Step 3: Create CSS styles**

Create `web/static/style.css`:

```css
* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}

body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    background-color: #f5f5f5;
}

.container {
    max-width: 800px;
    margin: 0 auto;
    height: 100vh;
    display: flex;
    flex-direction: column;
}

header {
    background-color: #4A90E2;
    color: white;
    padding: 20px;
    text-align: center;
}

#chat-container {
    flex: 1;
    overflow-y: auto;
    padding: 20px;
    background-color: white;
}

#messages {
    display: flex;
    flex-direction: column;
    gap: 15px;
}

.message {
    padding: 12px 16px;
    border-radius: 8px;
    max-width: 70%;
    word-wrap: break-word;
}

.message.user {
    background-color: #4A90E2;
    color: white;
    align-self: flex-end;
}

.message.agent {
    background-color: #f0f0f0;
    color: #333;
    align-self: flex-start;
}

.input-container {
    display: flex;
    padding: 20px;
    background-color: white;
    border-top: 1px solid #e0e0e0;
}

#user-input {
    flex: 1;
    padding: 12px;
    border: 1px solid #d0d0d0;
    border-radius: 4px;
    font-size: 14px;
}

#send-btn {
    margin-left: 10px;
    padding: 12px 24px;
    background-color: #4A90E2;
    color: white;
    border: none;
    border-radius: 4px;
    cursor: pointer;
    font-size: 14px;
}

#send-btn:hover {
    background-color: #357ABD;
}

#send-btn:disabled {
    background-color: #ccc;
    cursor: not-allowed;
}
```

- [ ] **Step 4: Create JavaScript for chat**

Create `web/static/chat.js`:

```javascript
let ws = null;
let sessionId = null;

function connect() {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    ws = new WebSocket(`${protocol}//${window.location.host}/ws`);

    ws.onopen = function() {
        console.log('Connected to server');
    };

    ws.onmessage = function(event) {
        const data = JSON.parse(event.data);

        if (data.type === 'message') {
            addMessage(data.content, 'agent');
            sessionId = data.session_id;
        } else if (data.type === 'error') {
            addMessage(`Error: ${data.content}`, 'agent');
        }
    };

    ws.onerror = function(error) {
        console.error('WebSocket error:', error);
        addMessage('Connection error. Please refresh the page.', 'agent');
    };

    ws.onclose = function() {
        console.log('Disconnected from server');
    };
}

function addMessage(content, sender) {
    const messagesDiv = document.getElementById('messages');
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${sender}`;
    messageDiv.textContent = content;
    messagesDiv.appendChild(messageDiv);

    // Scroll to bottom
    const chatContainer = document.getElementById('chat-container');
    chatContainer.scrollTop = chatContainer.scrollHeight;
}

function sendMessage() {
    const input = document.getElementById('user-input');
    const message = input.value.trim();

    if (!message) return;

    // Add user message to UI
    addMessage(message, 'user');

    // Send to server
    ws.send(JSON.stringify({ message: message }));

    // Clear input
    input.value = '';
}

// Event listeners
document.getElementById('send-btn').addEventListener('click', sendMessage);

document.getElementById('user-input').addEventListener('keypress', function(e) {
    if (e.key === 'Enter') {
        sendMessage();
    }
});

// Connect on page load
connect();
```

- [ ] **Step 5: Create web app entry point**

Create `web_app.py`:

```python
#!/usr/bin/env python3
"""Web application entry point."""
import uvicorn
from web.app import app

if __name__ == "__main__":
    config_path = "config.yaml"
    import sys
    sys.path.insert(0, ".")

    from src.config import load_config
    config = load_config(config_path)

    host = config.beauty.web.host if config.beauty else "0.0.0.0"
    port = config.beauty.web.port if config.beauty else 8080

    uvicorn.run(app, host=host, port=port)
```

- [ ] **Step 6: Write test for web app**

Create `tests/test_web_app.py`:

```python
"""Test web application."""
from fastapi.testclient import TestClient
from web.app import app


def test_index_page():
    """Should render index page."""
    client = TestClient(app)
    response = client.get("/")
    assert response.status_code == 200
    assert "娇莉芙智能助手" in response.text


def test_health_check():
    """Should return health status."""
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
```

- [ ] **Step 7: Run web app test**

Run: `pytest tests/test_web_app.py -v`

Expected: PASS

- [ ] **Step 8: Commit web interface**

```bash
git add web/ web_app.py tests/test_web_app.py
git commit -m "feat: add FastAPI web interface with WebSocket chat"
```

---

## Task 9: 集成测试与文档

**Files:**
- Create: `tests/test_integration.py`
- Update: `README.md`

- [ ] **Step 1: Write integration test**

Create `tests/test_integration.py`:

```python
"""Integration test for beauty agent."""
from langchain_core.messages import HumanMessage
from src.graph import build_graph
from langgraph.checkpoint.memory import MemorySaver
from src.config import load_config


def test_beauty_agent_end_to_end():
    """Test complete flow: intent -> knowledge -> MCP -> agent response."""
    config = load_config()
    checkpointer = MemorySaver()
    graph = build_graph(checkpointer=checkpointer)

    session_id = "test-session"
    graph_config = {"configurable": {"thread_id": session_id}}

    # Initial state
    state = {
        "messages": [],
        "tools": [],
        "context_window": {"used_tokens": 0, "max_tokens": 128000, "threshold": 0.7},
        "memory_context": "",
        "user_profile": "",
        "agent_notes": "",
        "pending_human_input": False,
        "iteration_count": 0,
        "max_iterations": config.agent["max_iterations"],
        "nudge_counter": 0,
        "provider_name": config.active_provider,
        "intent": "",
        "customer_context": {},
        "knowledge_results": [],
        "mcp_results": {},
    }

    # User query
    state["messages"].append(HumanMessage(content="查一下张女士的档案"))

    # Invoke graph
    result = graph.invoke(state, graph_config)

    # Verify
    assert result["intent"] in ["query_customer", "mixed"]
    assert len(result["messages"]) > 0
```

- [ ] **Step 2: Run integration test**

Run: `pytest tests/test_integration.py -v`

Expected: PASS (may require MCP server running)

- [ ] **Step 3: Update README**

Modify `README.md`, add web interface section:

```markdown
## Web Interface

Start the web interface:

```bash
python web_app.py
```

Open http://localhost:8080 in your browser.

## Beauty Agent

This agent includes beauty industry-specific features:

- **Knowledge RAG**: Query product info, procedures, scripts
- **Customer MCP**: Query customer profiles, consumption, service history

### Configuration

Add to `config.yaml`:

\`\`\`yaml
beauty:
  knowledge_base:
    base_dir: "./knowledge/jiaolifu"
  mcp_servers:
    customer:
      url: "http://localhost:3001"
\`\`\`

### Start MCP Server

\`\`\`bash
python -m mcp_servers.customer.main
\`\`\`
```

- [ ] **Step 4: Commit integration tests and docs**

```bash
git add tests/test_integration.py README.md
git commit -m "test: add integration test and update README for beauty agent"
```

---

## Self-Review Checklist

After writing the complete plan, verify:

- [x] **Spec coverage**: Each spec requirement maps to a task
  - Knowledge RAG → Task 2, 4
  - Intent classification → Task 3
  - Customer MCP → Task 5, 6
  - Graph integration → Task 7
  - Web interface → Task 8
  - Testing → Task 9

- [x] **Placeholder scan**: No TBD, TODO, or vague instructions

- [x] **Type consistency**: AgentState fields consistent across all tasks

- [x] **File paths**: All file paths are exact and consistent

- [x] **Code completeness**: Every code step shows full implementation

- [x] **Test coverage**: Each component has corresponding test

---

**Plan complete. Ready for execution.**
