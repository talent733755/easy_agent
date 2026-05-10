# 设计文档：Session 稳定性 + AI 分析深度优化

**日期：** 2026-05-10
**项目：** easy_agent（娇莉芙美容院 AI 助手）
**范围：** Session 心跳保活、重连上下文恢复、AI 结构化分析框架

---

## 背景

当前系统存在三个问题：
1. WebSocket 长时间无操作后断连，用户需手动刷新
2. 重连后 session 丢失，历史上下文无法恢复
3. AI 回答对客户数据的分析流于表面，缺乏结构化深度洞察

本文档覆盖三个优化方向的设计。

---

## 一、Session 心跳保活

### 目标
长时间无操作（如员工暂时离开）后，连接不断开；断开时自动重连，用户无感知。

### 前端设计

**心跳发送：**
- 连接建立后，每 30 秒发送 `{"type": "ping"}`
- 使用 `setInterval` 管理，重连后重新启动，断开时清除

**自动重连：**
- `ws.onclose` 触发时，等待后自动重连
- 重试策略：指数退避，最多 5 次
  - 第 1 次：2 秒后重试
  - 第 2 次：4 秒后重试
  - 第 3 次：8 秒后重试
  - 第 4 次：16 秒后重试
  - 第 5 次：32 秒后重试，失败则停止并提示"连接失败，请刷新页面"
- 重连期间状态栏显示"重新连接中..."

### 后端设计

**ping/pong 处理：**
- WebSocket handler 的消息循环中增加 `ping` 类型处理
- 收到 `{"type": "ping"}` 后立即回复 `{"type": "pong"}`

**uvicorn 配置：**
- 启动时加 `--ws-ping-interval 30 --ws-ping-timeout 300`，防止网络层超时

---

## 二、重连后上下文恢复

### 目标
重连后自动恢复到上次会话，对话记录和 LangGraph 状态均保留。

### 设计

**前端：**
- 首次连接成功、收到 `session` 消息后，将 `session_id` 存入 `localStorage`
- 重连时，握手后发送 `{"type": "resume", "session_id": "<旧ID>"}`
- 若服务端确认恢复（返回 `{"type": "session", "resumed": true}`），不清空聊天记录
- 若服务端返回新 session，显示系统提示"会话已过期，已开始新对话"并清空记录

**后端：**
- 新增 `session_last_active: dict[str, float]`，每次收到消息时更新时间戳
- 收到 `resume` 消息时：
  - 检查 `session_id` 是否存在于 `sessions` 中
  - 检查最后活跃时间是否在 2 小时内
  - 满足条件：复用旧 `thread_id`（即旧 `graph_config`），恢复 state，回复 `{"type": "session", "resumed": true, "session_id": "<旧ID>"}`
  - 不满足：创建新 session，回复 `{"type": "session", "resumed": false, "session_id": "<新ID>"}`

**LangGraph 状态：**
- `MemorySaver` 已按 `thread_id` 持久化对话历史，复用旧 `thread_id` 即可恢复 checkpointer 状态
- `sessions` dict 中的 state 也一并复用，无需重新初始化

---

## 三、AI 结构化分析框架

### 目标
针对四类业务场景，输出有固定结构、有实质洞察的分析，而不是泛泛复述数据。

### 四个场景框架

#### ① 接待前准备
触发条件：`intent == query_customer`，有 MCP 数据，用户问"客户情况/准备接待"

输出结构：
```
【客户快照】姓名、年龄、会员等级、累计消费
【上次到店】时间、项目、满意度
【本次预判】根据间隔和偏好推测可能需求
【接待建议】1-3 条具体提示（如：注意过敏史、可推荐XX项目）
```

#### ② 项目/产品咨询
触发条件：`intent == knowledge_query`，知识库有命中结果

输出结构：
```
【核心功效】2-3 句简洁描述
【适合人群】具体描述（肤质、年龄、问题类型）
【注意事项】禁忌或特殊要求
【推荐理由】如有客户数据，结合客户肤质/需求说明为何推荐
```

#### ③ 消费潜力分析
触发条件：`intent == query_customer`，有消费/档案数据，用户问"消费/潜力/推荐"

输出结构：
```
【消费行为摘要】频次、客单价、偏好品类
【规律洞察】消费周期、季节偏好、升单趋势
【潜在需求】基于数据推断的未满足需求
【行动建议】推荐套餐或下一步跟进动作
```

#### ④ 回店/流失预警
触发条件：`intent == query_customer`，有行为数据，用户问"回店/流失/跟进"

输出结构：
```
【最后到店】时间和项目
【到店间隔趋势】是否在拉长
【流失风险】高 / 中 / 低（附判断依据）
【建议话术】1-2 句具体的跟进话术示例
```

### 实现方式

- **不新增节点**，在 `agent_node.py` 的 `SYSTEM_PROMPT` 增加"分析框架"章节
- `agent_node` 根据 `state["intent"]` + `mcp_results` 是否非空，动态拼接对应框架指令
- 框架指令以"当前场景为XX，请按以下结构输出"的方式注入
- 明确禁止语：禁止输出"可能会回店"、"建议关注"等无实质内容的泛泛表述

---

## 文件改动清单

| 文件 | 改动 |
|------|------|
| `web/static/chat.js` | 心跳 setInterval、自动重连指数退避、resume 握手、localStorage |
| `web/app.py` | ping/pong 处理、resume 逻辑、session_last_active 维护 |
| `start.sh` | uvicorn 增加 ws-ping-interval/timeout 参数 |
| `src/nodes/agent_node.py` | SYSTEM_PROMPT 增加分析框架、动态拼接逻辑 |

---

## 不在本次范围内

- 跨服务重启的 session 持久化（需要 Redis/DB，后续扩展）
- 多用户并发隔离（当前单机部署，暂不需要）
- 分析节点独立化（当前 prompt 方案已够用，后续可升级）
