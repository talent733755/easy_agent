# 自动记忆学习系统设计

**日期:** 2026-05-08
**状态:** 已批准

---

## 背景

当前记忆系统（MEMORY.md / USER.md / FTS5 / FAISS）仅支持手动维护，每轮对话会读取但不会自动学习。用户提出需要系统能够**自动从纠错和显式指令中学习**，避免重复犯错。

---

## 设计目标

1. **纠错学习** — 用户指出错误时，自动提取场景、错误做法、正确做法，存入记忆
2. **显式指令学习** — "记住xxx"、"记下来"等指令，自动写入记忆
3. **去重合并** — 防止记忆膨胀，相似经验自动合并
4. **分层使用** — 核心经验始终注入，案例经验按需检索

---

## 架构概览

```
用户消息 → [memory_learn_node] → MEMORY.md / agent_notes
                    │
                    ├── 步骤1: 检测触发（关键词+LLM兜底）
                    ├── 步骤2: 提取结构化内容
                    ├── 步骤3: 去重合并
                    └── 步骤4: 分层存储
```

---

## 触发检测

### 关键词快速匹配（优先）

扫描用户消息，包含以下模式则触发学习流程：
- 纠错类：`不对`、`错了`、`不是这样`、`应该`、`正确的是`、`实际上`
- 指令类：`记住`、`记下来`、`以后要`、`以后别`、`别再`、`以后要记得`

### LLM 兜底（模糊情况）

关键词未触发时，用 LLM 判断是否包含可学习的纠错或指令。

**LLM 判断 Prompt：**
```
判断以下用户消息是否包含值得记忆的内容：
1. 纠错：指出 Agent 之前的错误做法
2. 显式指令：明确要求记住某事

用户消息：{last_message}
最近 Agent 回复：{last_agent_reply}

输出：
- SKIP：无学习价值
- CORRECTION：包含纠错，提取 [场景] [错误做法] [正确做法]
- INSTRUCTION：包含显式指令，提取 [指令内容]
```

---

## 结构化存储格式

### MEMORY.md 格式

```markdown
---
last_updated: 2026-05-08
---

## 错误经验

- [场景] 产品推荐时混淆了嫩小白和焕颜精华的功效
  [错误] 将美白淡斑功效归于嫩小白
  [正确] 嫩小白→补水保湿，焕颜精华→美白淡斑
  [来源] 2026-05-08 用户纠错
  [重要度] high
  [纠错次数] 1

---

## 用户指令

- [指令] 回答要简洁，不要太长的解释
  [来源] 2026-05-08 用户指令
  [重要度] medium
```

### 分层存储规则

| 条件 | 存储位置 | 使用方式 |
|------|---------|---------|
| 被纠错 ≥ 2 次 | agent_notes | 每次都注入 system prompt |
| 显式指令 | agent_notes | 每次都注入 system prompt |
| 普通纠错（1次） | memory_context | 仅在检索命中的场景出现 |

---

## 去重合并策略

新经验到来时，与现有 MEMORY.md 逐条对比：

1. **LLM 判断语义相似度**（阈值 0.8）
2. **相似** → 合并：将新细节追加到现有条目，更新 `纠错次数`、`last_updated`
3. **不相似** → 追加新条目
4. **容量兜底**：超过 20 条时，优先淘汰 `重要度=low` 且 `纠错次数=1` 的条目

---

## 组件设计

### memory_learn_node

**位置：** 在 `observe_node` 之后、`nudge_check_node` 之前执行

**输入：** `AgentState`（包含 `messages`、`agent_notes`）

**输出：** 写入/更新 MEMORY.md，返回空 dict（不修改 state）

**依赖：**
- `FileMemory` — 读写 MEMORY.md
- `config` 中的 LLM provider — 用于 LLM 兜底判断和去重合并
- `memory` 配置：`data_dir`、`memory_md_max_chars`

### 安全设计

沿用 `FileMemory` 的注入检测（`INJECTION_PATTERNS`），拒绝包含恶意模式的内容写入。

---

## 流程集成

在 `src/graph.py` 中改造或替换 `memory_save_node`：

```python
# 当前：
builder.add_edge("human_gate", "nudge_check")

# 改为：
builder.add_node("memory_learn", memory_learn_node)
builder.add_edge("human_gate", "memory_learn")
builder.add_edge("memory_learn", "nudge_check")
```

---

## 验收标准

1. 用户说"不对，嫩小白是补水的不是美白的" → 自动写入 MEMORY.md，标注场景/错误/正确
2. 用户说"记住，以后回答要简洁" → 自动写入 MEMORY.md
3. 相同场景第二次被纠错 → 合并到已有条目，`纠错次数` +1，重要度升级
4. 再次推荐产品时，Agent 不会重复混淆嫩小白和焕颜精华
5. 再次回答时，Agent 知道要简洁

---

## 实现顺序（建议）

1. **memory_learn_node** — 核心逻辑，关键词检测 + LLM 提取 + 去重合并 + 分层存储
2. **图集成** — 插入到 observe → nudge_check 之间
3. **测试** — 模拟各种触发场景，验证写入正确
4. **联调** — 真实对话中触发，观察 Agent 是否应用记忆