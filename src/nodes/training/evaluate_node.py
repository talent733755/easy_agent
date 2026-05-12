"""Training evaluation node: scores the trainee's performance and saves results."""

import json
import time
import sqlite3
import uuid
import os
from pathlib import Path

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.language_models import BaseChatModel

from src.state import AgentState


def _load_prompt(name: str) -> str:
    prompt_path = Path(__file__).parent.parent.parent / "prompts" / "training" / f"{name}.txt"
    if prompt_path.exists():
        return prompt_path.read_text(encoding="utf-8")
    return ""


def _format_conversation_history(messages: list) -> str:
    """格式化对话记录用于评分。"""
    parts = []
    for m in messages:
        if isinstance(m, HumanMessage):
            parts.append(f"员工：{m.content}")
        elif isinstance(m, AIMessage) and m.content:
            content = m.content
            if isinstance(content, list):
                text_parts = [
                    block["text"] for block in content
                    if isinstance(block, dict) and block.get("type") == "text"
                ]
                content = "\n".join(text_parts) if text_parts else str(content)
            parts.append(f"客户：{content}")
    return "\n".join(parts)


def _parse_score(report: str) -> dict:
    """从评价报告中解析总分和各维度分数。"""
    import re
    result = {"total": 0, "details": {}}

    # 解析总分
    total_match = re.search(r'总分[：:]\s*(\d+)', report)
    if total_match:
        result["total"] = int(total_match.group(1))

    # 解析各维度分数
    dim_pattern = r'\|\s*(沟通技巧|需求挖掘|产品/项目知识|异议处理|促成成交)\s*\|\s*(\d+)'
    for match in re.finditer(dim_pattern, report):
        dim_name = match.group(1)
        dim_score = int(match.group(2))
        result["details"][dim_name] = dim_score

    return result


def _save_training_session(
    db_path: str,
    scenario: str,
    customer_name: str,
    score: dict,
    conversation: str,
    feedback: str,
):
    """保存对练记录到数据库。"""
    session_id = str(uuid.uuid4())[:8]
    now = time.time()

    with sqlite3.connect(db_path) as conn:
        # 确保表存在
        conn.execute("""
            CREATE TABLE IF NOT EXISTS training_sessions (
                id TEXT PRIMARY KEY,
                scenario TEXT,
                customer_name TEXT,
                total_score INTEGER,
                score_detail TEXT,
                conversation TEXT,
                feedback TEXT,
                created_at REAL NOT NULL
            )
        """)
        conn.execute(
            """INSERT INTO training_sessions
               (id, scenario, customer_name, total_score, score_detail, conversation, feedback, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                session_id,
                scenario,
                customer_name,
                score.get("total", 0),
                json.dumps(score.get("details", {}), ensure_ascii=False),
                conversation,
                feedback,
                now,
            ),
        )
        conn.commit()

    return session_id


def evaluate_training_node(state: AgentState, model: BaseChatModel) -> dict:
    """评分节点：调用 LLM 生成评分报告，保存对练记录。"""
    training_context = state.get("training_context", {})
    scenario = training_context.get("scenario", {})
    customer_name = training_context.get("customer_name", "模拟客户")
    customer_info = training_context.get("customer_info", "")

    # 格式化对话记录（排除 welcome/setup 阶段的消息）
    messages = state.get("messages", [])
    conversation = _format_conversation_history(messages)

    # 构建评分 prompt
    eval_template = _load_prompt("evaluate")
    eval_prompt = eval_template.format(
        scenario_description=scenario.get("description", "自由对话"),
        customer_info=customer_info[:500],
        conversation_history=conversation,
    )

    # 调用 LLM 评分
    system_msg = SystemMessage(content=eval_prompt)
    response = model.invoke([system_msg])

    report = response.content
    if isinstance(report, list):
        text_parts = [
            block["text"] for block in report
            if isinstance(block, dict) and block.get("type") == "text"
        ]
        report = "\n".join(text_parts) if text_parts else str(report)

    # 解析分数
    score = _parse_score(report)

    # 保存到数据库
    data_dir = os.path.expanduser("~/.easy_agent")
    db_path = os.path.join(data_dir, "conversations.db")
    try:
        session_id = _save_training_session(
            db_path=db_path,
            scenario=scenario.get("name", "未知场景"),
            customer_name=customer_name,
            score=score,
            conversation=conversation,
            feedback=report,
        )
        report += f"\n\n_对练记录已保存（ID: {session_id}）_"
    except Exception as e:
        report += f"\n\n_保存记录失败: {e}_"

    # 重置训练状态，准备下一轮
    reply = (
        f"{report}\n\n"
        "---\n"
        "输入 **\u201c开始陪练\u201d** 或 **\u201c再来一轮\u201d** 开始新的对练。"
    )

    return {
        "messages": [AIMessage(content=reply)],
        "training_phase": "evaluate",
        "training_context": {},
        "training_scenario": "",
        "training_score": score,
    }
