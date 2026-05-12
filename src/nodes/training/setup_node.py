"""Training setup node: scenario selection + customer profile fetch."""

import json
import re
import httpx
from pathlib import Path

from src.state import AgentState
from src.config import load_config


# 预定义场景
SCENARIOS = {
    "1": {
        "id": "new_consult",
        "name": "新客首访咨询",
        "description": "客户第一次到店，需要了解客户需求、建立信任、推荐适合的基础护理项目。重点在于倾听和需求挖掘，不要急于推销。",
        "personality_hints": ["谨慎", "好奇", "货比三家", "关注价格"],
    },
    "2": {
        "id": "upgrade",
        "name": "老客续卡/升单",
        "description": "老客户到店续卡或升级服务，需要根据历史消费推荐更合适的项目或套餐。重点在于价值塑造和个性化推荐。",
        "personality_hints": ["信任门店", "关注性价比", "有明确偏好", "时间有限"],
    },
    "3": {
        "id": "after_sales",
        "name": "售后/客诉处理",
        "description": "客户对上次服务不满意或出现副作用，需要安抚情绪、了解问题、给出解决方案。重点在于共情和专业处理。",
        "personality_hints": ["不满", "担心", "需要被重视", "要求明确答复"],
    },
    "4": {
        "id": "reactivate",
        "name": "消耗提升（激活沉睡客）",
        "description": "联系长期未到店的老客户，唤醒消费意愿、推荐新项目或活动。重点在于不生硬、自然激活。",
        "personality_hints": ["忙碌", "遗忘", "有顾虑", "需要理由"],
    },
}


def _extract_customer_info(mcp_results: dict) -> tuple[str, str]:
    """从 MCP 结果中提取客户信息，返回 (customer_name, formatted_info)。"""
    for svc_name, data in mcp_results.items():
        if isinstance(data, dict) and "error" not in data:
            if "result" in data and isinstance(data["result"], str):
                text = data["result"]
                name_match = re.search(r'姓名[：:]\s*(\S+)', text)
                name = name_match.group(1) if name_match else "模拟客户"
                return name, text
            else:
                text = json.dumps(data, ensure_ascii=False, indent=2)
                name = data.get("name", data.get("customer_name", "模拟客户"))
                return str(name), text
    return "模拟客户", "(无客户档案数据)"


def _extract_key_info_llm(customer_info: str, scenario: dict, model) -> list[str]:
    """用 LLM 从客户档案中提取场景最相关的 8 条关键信息。"""
    from langchain_core.messages import SystemMessage, HumanMessage

    scenario_name = scenario.get("name", "自由对话")
    scenario_desc = scenario.get("description", "")
    print(f"[DEBUG] _extract_key_info_llm 开始, scenario: {scenario_name}")

    system_prompt = """你是一位美容行业培训系统的信息提取专家。从客户档案中提取与指定场景最相关的 8 条关键信息。
输出要求：
1. 严格输出 8 条，每条一行
2. 格式：标签：值（如「年龄：35岁」「偏好项目：面部护理」）
3. 只提取对该场景最有用的信息，去掉无关内容
4. 如果档案信息不足 8 条，用合理的推断补充（标注为推断）
5. 不要输出任何解释文字，只输出 8 行信息"""

    user_prompt = f"""## 场景：{scenario_name}
{scenario_desc}

## 客户档案
{customer_info[:1500]}

请提取 8 条关键信息："""

    try:
        response = model.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ])
        content = response.content
        if isinstance(content, list):
            text_parts = [
                block["text"] for block in content
                if isinstance(block, dict) and block.get("type") == "text"
            ]
            content = "\n".join(text_parts) if text_parts else str(content)

        lines = [line.strip().lstrip("- ").lstrip("* ") for line in content.strip().split("\n") if line.strip()]
        print(f"[DEBUG] LLM 提取成功, {len(lines)} 条: {lines[:3]}")
        return lines[:8]
    except Exception as e:
        print(f"[DEBUG] LLM 提取失败: {e}, 使用 fallback")
        return _extract_key_info_fallback(customer_info, scenario.get("id", ""), 8)


def _extract_key_info_fallback(info_text: str, scenario_id: str, max_items: int = 8) -> list[str]:
    """降级方案：关键词提取。"""
    scenario_keywords = {
        "new_consult": ["姓名", "年龄", "肤质", "过敏", "敏感", "职业", "关注", "预算", "到店", "渠道"],
        "upgrade": ["姓名", "会员", "消费", "项目", "偏好", "卡项", "余额", "频次", "最近", "客单"],
        "after_sales": ["姓名", "消费", "项目", "过敏", "投诉", "记录", "等级", "联系", "最近", "注意"],
        "reactivate": ["姓名", "最近", "到店", "间隔", "天数", "偏好", "余额", "沉睡", "消费", "等级"],
    }
    keywords = scenario_keywords.get(scenario_id, [])
    lines = info_text.split("\n")
    scored_lines = []
    max_line_len = 80  # 超过此长度的行认为是格式错误，跳过

    for line in lines:
        line = line.strip()
        if not line or len(line) < 4 or len(line) > max_line_len:
            continue
        score = sum(1 for kw in keywords if kw in line)
        if score > 0:
            scored_lines.append((score, line))

    scored_lines.sort(key=lambda x: x[0], reverse=True)
    key_info = [line for _, line in scored_lines[:max_items]]

    if len(key_info) < max_items:
        for line in lines:
            line = line.strip()
            if line and 4 <= len(line) <= max_line_len and line not in key_info:
                key_info.append(line)
            if len(key_info) >= max_items:
                break

    return key_info[:max_items]


def _fetch_knowledge(scenario: dict, knowledge_dir: str) -> str:
    """从知识库检索与场景相关的内容。"""
    try:
        from src.tools.beauty.knowledge_rag import KnowledgeRAG

        kb_path = Path(knowledge_dir)
        if not kb_path.exists():
            return ""

        rag = KnowledgeRAG(str(kb_path))
        index_path = kb_path / ".faiss_index"
        if index_path.exists():
            try:
                rag.load_index(str(index_path))
            except Exception:
                return ""
        else:
            return ""

        query = f"{scenario['name']} {scenario['description'][:50]}"
        results = rag.search(query, top_k=3)

        parts = []
        for r in results:
            source = r.get("source", "unknown")
            content = r.get("content", "")
            parts.append(f"【来源：{source}】\n{content}")
        return "\n\n".join(parts)
    except Exception:
        return ""


def create_setup_node(model):
    """创建训练设置节点工厂。"""
    def setup_training_node(state: AgentState) -> dict:
        """训练设置节点：解析场景选择，获取客户档案，提取关键信息。"""
        from langchain_core.messages import HumanMessage, AIMessage

        messages = state.get("messages", [])
        last_user_msg = ""
        for m in reversed(messages):
            if isinstance(m, HumanMessage):
                last_user_msg = m.content.strip()
                break

        training_context = state.get("training_context", {})
        scenario = training_context.get("scenario")

        if not scenario:
            return _handle_scenario_selection(last_user_msg, training_context)
        else:
            return _handle_customer_setup(last_user_msg, training_context, scenario, model)

    return setup_training_node


def _handle_scenario_selection(last_user_msg: str, training_context: dict) -> dict:
    """第一步：选择场景。"""
    from langchain_core.messages import AIMessage

    start_triggers = ["开始陪练", "开始对练", "开始训练", "陪练",
                      "再来一轮", "再练一次", "重新开始", "继续对练"]
    if last_user_msg in start_triggers:
        menu = "请选择对练场景（输入编号 1-4）：\n\n"
        for key, s in SCENARIOS.items():
            menu += f"**{key}. {s['name']}** — {s['description'][:30]}...\n"
        menu += "\n或直接输入你想练习的场景描述（自定义场景）。"
        return {
            "training_context": training_context,
            "training_phase": "setup",
            "messages": [AIMessage(content=menu)],
        }

    scenario = None
    if last_user_msg in SCENARIOS:
        scenario = SCENARIOS[last_user_msg]
    else:
        for key, s in SCENARIOS.items():
            if s["name"] in last_user_msg or s["id"] in last_user_msg:
                scenario = s
                break
        if not scenario:
            scenario = {
                "id": "custom",
                "name": "自定义场景",
                "description": last_user_msg,
                "personality_hints": ["普通客户"],
            }

    training_context["scenario"] = scenario
    reply = (
        f"场景已选择：**{scenario['name']}**\n\n"
        f"{scenario['description']}\n\n"
        f"请输入客户信息来模拟：提供 **user_id**（如 468388）或 **手机号**，"
        f"我会从系统中调取该客户档案来模拟。"
        f"如果想用虚拟客户，直接输入「虚拟客户」。"
    )
    return {
        "training_context": training_context,
        "training_phase": "setup",
        "training_scenario": scenario["name"],
        "messages": [AIMessage(content=reply)],
    }


def _handle_customer_setup(last_user_msg: str, training_context: dict, scenario: dict, model) -> dict:
    """第二步：获取客户档案并提取关键信息。"""
    from langchain_core.messages import AIMessage

    if last_user_msg in ("虚拟客户", "虚拟", "随机"):
        customer_name = "王女士"
        customer_info = (
            "姓名：王女士\n年龄：35岁\n会员等级：银卡\n"
            "职业：企业白领\n肤质：混合偏干\n"
            "最近到店：2026-04-15\n累计消费：¥12,800\n"
            "偏好项目：面部护理、补水项目\n"
            "过敏史：无\n备注：注重性价比，时间观念强"
        )
        key_info = _extract_key_info_llm(customer_info, scenario, model)
        training_context.update({
            "customer_name": customer_name,
            "customer_info": customer_info,
            "key_info": key_info,
            "knowledge_refs": "",
            "rounds": 0,
        })
        reply = (
            f"已加载虚拟客户：**{customer_name}**\n\n"
            f"**场景关键信息：**\n"
            + "\n".join(f"- {info}" for info in key_info)
            + f"\n\n对练开始！请以顾问身份开始和客户沟通。"
            f"（输入「结束对练」或「评价」可随时结束）"
        )
        return {
            "training_context": training_context,
            "training_phase": "roleplay",
            "messages": [AIMessage(content=reply)],
        }

    # 尝试通过 MCP 获取真实客户档案
    config = load_config()
    beauty_config = config.beauty
    gateway_url = beauty_config.mcp_gateway.url if beauty_config else "http://localhost:3001"

    # 提取标识符
    identifiers = {}
    phone_match = re.search(r'1[3-9]\d{9}', last_user_msg)
    if phone_match:
        identifiers["phone"] = phone_match.group()
    msg_no_phone = re.sub(r'1[3-9]\d{9}', '', last_user_msg)
    id_match = re.search(r'(?<!\d)(\d{4,10})(?!\d)', msg_no_phone)
    if id_match:
        identifiers["user_id"] = id_match.group(1)

    if not identifiers:
        reply = (
            "未能识别客户信息。请输入 **user_id**（如 468388）或 **手机号**（如 18102481137），"
            "或输入「虚拟客户」使用虚拟客户。"
        )
        return {
            "training_context": training_context,
            "training_phase": "setup",
            "messages": [AIMessage(content=reply)],
        }

    # 调用 MCP 获取客户档案
    mcp_results = {}
    for svc_name in ("user_profile", "customer"):
        svc_url = f"{gateway_url}/{svc_name}"
        try:
            params = {}
            if "user_id" in identifiers:
                params["user_id"] = identifiers["user_id"]
            if "phone" in identifiers:
                params["mobile"] = identifiers["phone"]
                params["phone"] = identifiers["phone"]

            resp = httpx.post(
                f"{svc_url}/tools/get_customer" if svc_name == "customer" else f"{svc_url}/tools/get_user_info",
                json=params,
                timeout=15,
            )
            if resp.status_code < 400:
                data = resp.json()
                if "error" not in data:
                    mcp_results[svc_name] = data
        except Exception:
            continue

    if not mcp_results:
        reply = (
            f"未找到该客户信息（{identifiers}）。请确认 user_id 或手机号是否正确，"
            f"或输入「虚拟客户」使用虚拟客户。"
        )
        return {
            "training_context": training_context,
            "training_phase": "setup",
            "messages": [AIMessage(content=reply)],
        }

    # 提取客户信息
    customer_name, customer_info = _extract_customer_info(mcp_results)
    print(f"[DEBUG] customer_info 长度: {len(customer_info)}, 前200字: {customer_info[:200]}")
    key_info = _extract_key_info_llm(customer_info, scenario, model)
    print(f"[DEBUG] key_info 结果: {key_info}")

    # 获取相关知识库内容
    kb_dir = beauty_config.knowledge_base.base_dir if beauty_config else ""
    knowledge_refs = _fetch_knowledge(scenario, kb_dir)

    training_context.update({
        "customer_name": customer_name,
        "customer_info": customer_info,
        "key_info": key_info,
        "knowledge_refs": knowledge_refs,
        "mcp_results": mcp_results,
        "rounds": 0,
    })

    reply = (
        f"已加载客户档案：**{customer_name}**\n\n"
        f"**场景关键信息：**\n"
        + "\n".join(f"- {info}" for info in key_info)
        + f"\n\n对练开始！请以顾问身份开始和客户沟通。"
        f"（输入「结束对练」或「评价」可随时结束）"
    )
    return {
        "training_context": training_context,
        "training_phase": "roleplay",
        "messages": [AIMessage(content=reply)],
    }
