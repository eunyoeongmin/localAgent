import sys
import subprocess
import importlib.util

# [추가] Hugging Face 환경에서 ddgs 모듈을 찾지 못하는 문제 해결을 위한 런타임 설치
try:
    importlib.util.find_spec("ddgs")
except (ImportError, AttributeError):
    print("[INFO] ddgs 패키지가 없어 설치를 시도합니다...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-U", "duckduckgo-search"])

import subprocess as sp 
import chainlit as cl
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from config import chat_llm, tool_llm, SYSTEM_PROMPT, create_llm, CHAT_TEMPERATURE, TOOL_TEMPERATURE
from tools import all_tools

# 툴 바인딩
tool_llm_with_tools = tool_llm.bind_tools(all_tools)

async def safe_llm_call(messages, is_tool=False):
    """LLM 호출을 시도하고 실패 시 사용자에게 알림을 보냅니다."""
    llm_instance = chat_llm if not is_tool else tool_llm_with_tools
    
    # 도구 호출인 경우 로컬 모델에 도구 바인딩
    if is_tool:
        llm_instance = create_llm(temperature=TOOL_TEMPERATURE, provider="local").bind_tools(all_tools)

    try:
        return await llm_instance.ainvoke(messages)
    except Exception as e:
        await cl.Message(content=f"❌ **[로컬 에러]** 모델 호출 중 오류가 발생했습니다: {str(e)}").send()
        raise e

# --- 기존 로직 유지 ---
CHANGE_ACTION_KEYWORDS = ["修正", "直して", "リファクタリング", "パッチ", "追加", "削除", "変更", "改善", "リ네임"]
CODE_CONTEXT_KEYWORDS = ["코드", "関数", "クラス", "モジュール", "バグ", "エラー", "テスト", "lint", ".py", ".js", ".ts", ".tsx", ".jsx", ".md", ".json", ".yaml", ".yml"]
APPROVAL_WORDS = {"承認", "進行", "go", "yes", "y", "ok", "確認"}

def is_code_change_request(text: str) -> bool:
    lowered = text.lower()
    return any(k in lowered for k in CHANGE_ACTION_KEYWORDS) and any(k in lowered for k in CODE_CONTEXT_KEYWORDS)

def is_approval(text: str) -> bool:
    return text.strip().lower() in APPROVAL_WORDS

async def generate_change_plan(user_request: str) -> str:
    planner_messages = [SystemMessage(content="あなたは計画作成器です。手順を번호付きで作成してください。"), HumanMessage(content=user_request)]
    plan_response = await safe_llm_call(planner_messages, is_tool=False)
    return str(plan_response.content).strip()

def extract_content(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join([item['text'] if isinstance(item, dict) and 'text' in item else str(item) for item in content])
    return str(content)

@cl.on_chat_start
async def start_chat():
    """채팅 시작 시 초기화"""
    DYNAMIC_PROMPT = SYSTEM_PROMPT + "\n[重要] 検索結果が不十分な場合はキーワードを変更して再検索してください。"
    cl.user_session.set("messages", [SystemMessage(content=DYNAMIC_PROMPT)])
    cl.user_session.set("pending_change_request", None)

@cl.on_message
async def main(message: cl.Message):
    query = message.content
    messages = cl.user_session.get("messages")
    pending_change_request = cl.user_session.get("pending_change_request")

    if pending_change_request is not None:
        if is_approval(query):
            messages.append(HumanMessage(content=f"사용자가 계획을 승인했습니다.\n元のリクエスト: {pending_change_request}"))
            cl.user_session.set("pending_change_request", None)
        else:
            await cl.Message(content="[시스템] 계획이 취소되었습니다.").send()
            cl.user_session.set("pending_change_request", None)
            return
    elif is_code_change_request(query):
        plan_text = await generate_change_plan(query)
        await cl.Message(content=f"📋 **[변경 계획]**\n{plan_text}\n\n위 계획대로 진행하시겠습니까? '승인'을 입력해 주세요.").send()
        messages.append(HumanMessage(content=query))
        messages.append(AIMessage(content=f"[변경 계획]\n{plan_text}"))
        cl.user_session.set("pending_change_request", query)
        return
    else:
        messages.append(HumanMessage(content=query))

    msg = cl.Message(content="")
    await msg.send()

    max_retries = 3
    current_attempt = 0
    while current_attempt < max_retries:
        try:
            response = await safe_llm_call(messages, is_tool=True)
            if not response.tool_calls:
                final_response = await safe_llm_call(messages, is_tool=False)
                messages.append(final_response)
                msg.content = extract_content(final_response.content)
                await msg.update()
                break
            
            messages.append(response)
            for tool_call in response.tool_calls:
                matched = next((t for t in all_tools if t.name == tool_call['name']), None)
                result = await matched.ainvoke(tool_call['args']) if matched else "Error: Tool not found."
                messages.append(ToolMessage(content=str(result), tool_call_id=tool_call['id']))
            current_attempt += 1
        except Exception:
            return

    cl.user_session.set("messages", messages)

if __name__ == "__main__":
    if "chainlit" not in sys.argv[0]:
        sp.run([sys.executable, "-m", "chainlit", "run", __file__, "--port", "7860", "--host", "0.0.0.0", "--headless"])
