# 娇莉芙美容行业垂直 Agent - Phase 1 MVP 设计文档

**日期**: 2026-05-05
**版本**: Phase 1 MVP
**目标**: 验证端到端流程，实现知识库 RAG + 客户查询 MCP

---

## 1. 概述

### 1.1 目标
构建面向娇莉芙生美门店员工的 Web 聊天 Agent，替代现有店务系统的查询操作。Phase 1 聚焦验证核心架构：分层知识库检索 + MCP 业务查询。

### 1.2 成功标准
- [ ] 员工可通过自然语言查询产品/流程知识
- [ ] 员工可查询客户档案信息
- [ ] 多轮对话保持上下文
- [ ] Web 界面响应时间 < 3s

---

## 2. 架构设计

### 2.1 系统架构

```
┌─────────────────────────────────────────┐
│           Web Chat Interface            │
│         (FastAPI + WebSocket)           │
└─────────────────────────────────────────┘
                    │
┌─────────────────────────────────────────┐
│           LangGraph Agent               │
│  ┌─────────┐    ┌─────────┐    ┌─────┐ │
│  │ Intent  │───→│Knowledge│    │MCP  │ │
│  │ Classify│    │  RAG    │    │Client│ │
│  └─────────┘    └─────────┘    └──┬──┘ │
│       │                           │    │
│       └──────────→ Agent Node ←───┘    │
│                    │                   │
│              Tool Executor             │
└─────────────────────────────────────────┘
                    │
        ┌───────────┴───────────┐
        ▼                       ▼
┌───────────────┐       ┌───────────────┐
│  FAISS Vector │       │ Customer MCP  │
│    Store      │       │  (SaaS API)   │
└───────────────┘       └───────────────┘
```

### 2.2 数据流

```
用户: "帮我查一下张女士的档案，她适合做什么项目？"
  │
  ▼
[Intent Classify] → 识别: query_customer + recommend_service
  │
  ├─→ [Knowledge RAG] → 检索: 适合敏感肌的疗程方案
  │
  └─→ [MCP Client] → 调用: customer-mcp.get_customer("张女士")
  │
  ▼
[Agent Node] → 整合信息 → 生成回复
  │
  ▼
用户: "张女士，35岁，敏感肌... 推荐XX舒缓疗程..."
```

---

## 3. 组件设计

### 3.1 新增 State 字段

```python
class AgentState(TypedDict):
    # ... 现有字段 ...

    # Phase 1 新增
    intent: str                    # 识别意图: query_customer | knowledge_query | mixed
    customer_context: dict         # 当前客户信息缓存
    knowledge_results: list        # RAG 检索结果
    mcp_results: dict              # MCP 调用结果
```

### 3.2 新增 Graph 节点

| 节点 | 职责 | 输入 | 输出 |
|------|------|------|------|
| `intent_classify` | 识别用户意图 | messages | intent, customer_name(可选) |
| `knowledge_retrieve` | 分层知识检索 | intent, query | knowledge_results |
| `mcp_customer` | 调用 Customer MCP | customer_name | customer_context, mcp_results |

### 3.3 Graph 流程 (Phase 1)

```
START → intent_classify ─┬─> knowledge_retrieve ─┐
                         │                       │
                         └─> mcp_customer ───────┤
                                                 ▼
                         memory_retrieve → agent_node → tool_executor
                                                         │
                              ┌──────────────────────────┘
                              ▼
                         observe → human_gate → memory_save → END
```

---

## 4. 知识库设计

### 4.1 目录结构

```
knowledge/jiaolifu/
├── products/              # 产品知识
│   ├── 品牌A/
│   │   ├── 洁面乳.md
│   │   └── 精华液.md
│   └── 品牌B/
├── procedures/            # 服务流程
│   ├── 面部护理/
│   ├── 身体护理/
│   └── 仪器操作/
├── scripts/               # 销售话术
│   ├── 新客接待.md
│   ├── 疗程推荐.md
│   └── 异议处理.md
└── troubleshooting/       # 问题处理
    ├── 过敏反应.md
    ├── 客诉处理.md
    └── 应急方案.md
```

### 4.2 检索策略

1. **意图路由**: 根据 `intent` 确定搜索哪些索引
2. **向量检索**: 在各索引内做相似度搜索
3. **重排序**: 按相关性 + 时效性排序
4. **上下文组装**: Top-3 结果注入 Prompt

---

## 5. MCP 设计

### 5.1 Customer MCP (Phase 1 唯一 MCP)

**服务**: `customer-mcp`
**协议**: MCP over HTTP/SSE
**端口**: 3001

#### Tools

| Tool | 描述 | 参数 | 返回值 |
|------|------|------|--------|
| `get_customer` | 查询客户档案 | name 或 phone | 客户基本信息 |
| `get_consumption` | 查询消费记录 | customer_id, limit | 消费历史列表 |
| `get_service_history` | 查询服务历史 | customer_id, limit | 服务记录列表 |

#### 示例调用

```json
// Request
{
  "tool": "get_customer",
  "args": {"name": "张女士"}
}

// Response
{
  "id": "C12345",
  "name": "张女士",
  "phone": "138****8888",
  "skin_type": "敏感肌",
  "last_visit": "2025-04-20",
  "total_spent": 15800
}
```

---

## 6. Web 界面设计

### 6.1 技术栈

- **Backend**: FastAPI + WebSocket
- **Frontend**: HTML + Vanilla JS (Phase 1 简化)
- **样式**: 简洁聊天界面，类似 Claude Web

### 6.2 界面布局

```
┌─────────────────────────────────────┐
│  🤖 娇莉芙智能助手                    │
├─────────────────────────────────────┤
│                                     │
│  ┌────────┐                         │
│  │ 用户   │ 帮我查一下张女士...      │
│  └────────┘                         │
│                                     │
│         ┌──────────────────┐        │
│         │ 🤖 找到张女士... │        │
│         │ 推荐XX疗程...     │        │
│         └──────────────────┘        │
│                                     │
├─────────────────────────────────────┤
│ [输入消息...]              [发送]   │
└─────────────────────────────────────┘
```

### 6.3 API 端点

| 端点 | 方法 | 描述 |
|------|------|------|
| `/` | GET | 聊天页面 |
| `/ws` | WebSocket | 实时对话 |
| `/health` | GET | 健康检查 |

---

## 7. 配置设计

### 7.1 config.yaml 新增

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
      url: "http://localhost:3001/sse"
      timeout: 30

  intent_prompt: "./prompts/beauty_intent_v1.txt"

  web:
    host: "0.0.0.0"
    port: 8080
```

---

## 8. 实现计划

### 8.1 任务分解

| # | 任务 | 预估工时 | 依赖 |
|---|------|---------|------|
| 1 | 知识库文档预处理 → 向量化 | 4h | - |
| 2 | 实现 Knowledge RAG 节点 | 6h | 1 |
| 3 | 实现 Intent Classify 节点 | 4h | - |
| 4 | 开发 Customer MCP Server | 6h | - |
| 5 | 实现 MCP Client 节点 | 4h | 4 |
| 6 | 组装 Graph 流程 | 4h | 2,3,5 |
| 7 | 开发 Web 界面 (FastAPI) | 6h | 6 |
| 8 | 集成测试 & 调优 | 4h | 7 |

**总计**: ~38 小时

### 8.2 目录结构

```
easy_agent/
├── main.py                    # CLI 入口（保留）
├── web_app.py                 # Web 入口（新增）
├── config.yaml
├── knowledge/
│   └── jiaolifu/             # 知识文档（用户数据）
├── src/
│   ├── ...                   # 现有代码
│   ├── nodes/
│   │   ├── ...               # 现有节点
│   │   └── beauty/           # 新增美容节点
│   │       ├── __init__.py
│   │       ├── intent_node.py
│   │       ├── knowledge_node.py
│   │       └── mcp_client.py
│   └── tools/beauty/         # 美容工具
│       ├── __init__.py
│       └── knowledge_rag.py
├── web/                      # Web 界面
│   ├── __init__.py
│   ├── app.py
│   ├── static/
│   │   ├── style.css
│   │   └── chat.js
│   └── templates/
│       └── index.html
└── mcp_servers/              # MCP 服务
    └── customer/
        ├── main.py
        └── tools.py
```

---
## 9. 风险与应对

| 风险 | 影响 | 应对 |
|------|------|------|
| MCP 协议学习成本 | 中 | 先用 HTTP 简化，后续迁移 |
| 知识库检索不准 | 高 | 意图路由 + 重排序优化 |
| SaaS API 延迟高 | 中 | 添加缓存层，异步加载 |

---

## 10. Phase 2 展望

Phase 1 验证后，Phase 2 可扩展：
- 接入更多 MCP（appointment, sales, service）
- 支持写操作（创建预约、记录成交）
- 多门店权限隔离
- 对话历史持久化

---

**设计确认**: 待用户审批
**下一步**: 审批后编写详细实现计划 (writing-plans)
