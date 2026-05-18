import sys
import subprocess
import importlib.util
import os

# [追加] Hugging Face環境でddgsモジュールが見つからない問題解決のためのランタイムインストール
try:
    importlib.util.find_spec("ddgs")
except (ImportError, AttributeError):
    print("[INFO] ddgsパッケージが見つからないため、インストールを試行します...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-U", "duckduckgo-search"])

import subprocess as sp 
import chainlit as cl
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from config import chat_llm, tool_llm, SYSTEM_PROMPT, create_llm, CHAT_TEMPERATURE, TOOL_TEMPERATURE, GOOGLE_API_KEY
from tools import all_tools

# 툴 바인딩
tool_llm_with_tools = tool_llm.bind_tools(all_tools)

async def show_provider_selector():
    """사용자가 LLM을 선택할 수 있는 버튼을 띄웁니다."""
    actions = [
        cl.Action(name="select_provider", value="gemini", label="✨ Gemini 3.1", description="Google Gemini 모델 사용"),
        cl.Action(name="select_provider", value="local", label="🏠 Local LLM", description="로컬 모델(LM Studio 등) 사용")
    ]
    await cl.Message(content="🤖 **사용하실 AI 모델을 선택해 주세요:**", actions=actions).send()

async def safe_llm_call(messages, is_tool=False):
    """LLM 호출을 시도하고 실패 시 사용자에게 알림을 보냅니다."""
    # 세션에서 현재 프로바이더 가져오기 (없으면 기본값 설정)
    provider = cl.user_session.get("current_provider")
    if not provider:
        provider = "gemini" if GOOGLE_API_KEY else "local"
        cl.user_session.set("current_provider", provider)
    
    if provider == "gemini":
        llm_instance = chat_llm if not is_tool else tool_llm_with_tools
    else:
        llm_instance = create_llm(temperature=CHAT_TEMPERATURE if not is_tool else TOOL_TEMPERATURE, provider="local")
        if is_tool:
            llm_instance = llm_instance.bind_tools(all_tools)

    try:
        return await llm_instance.ainvoke(messages)
    except Exception as e:
        error_msg = str(e).lower()
        if provider == "gemini" and any(keyword in error_msg for keyword in ["google", "gemini", "401", "403", "429", "500", "503", "connection"]):
            await cl.Message(content="🚫 **[Gemini 제한 알림]** 현재 서비스 이용이 원활하지 않습니다. 잠시 후 다시 시도하시거나 아래 버튼으로 모델을 변경해 주세요.").send()
            await show_provider_selector()
            raise e
        raise e

@cl.action_callback("select_provider")
async def on_action(action):
    """모델 선택 버튼 클릭 시 처리"""
    provider = action.value
    cl.user_session.set("current_provider", provider)
    
    if provider == "gemini":
        if not GOOGLE_API_KEY:
            await cl.Message(content="❌ API 키가 설정되어 있지 않아 Gemini를 사용할 수 없습니다. 로컬 모드로 유지합니다.").send()
            cl.user_session.set("current_provider", "local")
            return
        await cl.Message(content="✅ **Gemini 3.1** 모델로 전환되었습니다. 이제부터 Gemini가 답변합니다.").send()
    else:
        await cl.Message(content="✅ **로컬 LLM** 모드로 전환되었습니다. 이제부터 내 컴퓨터의 모델이 답변합니다.").send()

# --- 기존 로직 유지 ---
CHANGE_ACTION_KEYWORDS = ["修正", "直して", "リファクタリング", "パッチ", "追加", "削除", "変更", "改善", "リネーム"]
CODE_CONTEXT_KEYWORDS = ["코드", "関数", "クラス", "モジュール", "バグ", "エラー", "テスト", "lint", ".py", ".js", ".ts", ".tsx", ".jsx", ".md", ".json", ".yaml", ".yml"]
APPROVAL_WORDS = {"承認", "進行", "go", "yes", "y", "ok", "確認"}

def is_code_change_request(text: str) -> bool:
    lowered = text.lower()
    return any(k in lowered for k in CHANGE_ACTION_KEYWORDS) and any(k in lowered for k in CODE_CONTEXT_KEYWORDS)

def is_approval(text: str) -> bool:
    return text.strip().lower() in APPROVAL_WORDS

async def generate_change_plan(user_request: str) -> str:
    planner_messages = [SystemMessage(content="あなたは計画作成器です。手順を番号付きで作成してください。"), HumanMessage(content=user_request)]
    plan_response = await safe_llm_call(planner_messages, is_tool=False)
    return str(plan_response.content).strip()

def extract_content(content) -> str:
    if isinstance(content, str): return content
    if isinstance(content, list):
        return "".join([item['text'] if isinstance(item, dict) and 'text' in item else str(item) for item in content])
    return str(content)

@cl.on_chat_start
async def start_chat():
    """채팅 시작 시 초기화"""
    DYNAMIC_PROMPT = SYSTEM_PROMPT + "\n[重要] 検索結果が不十分な場合はキーワードを変更して再検索してください。"
    cl.user_session.set("messages", [SystemMessage(content=DYNAMIC_PROMPT)])
    cl.user_session.set("pending_change_request", None)
    
    provider = "gemini" if GOOGLE_API_KEY else "local"
    cl.user_session.set("current_provider", provider)
    
    await cl.Message(content=f"🚀 **시스템이 준비되었습니다.** (기본 모드: {'Gemini' if provider == 'gemini' else 'Local'})").send()
    await show_provider_selector()

@cl.on_message
async def main(message: cl.Message):
    query = message.content
    messages = cl.user_session.get("messages")
    pending_change_request = cl.user_session.get("pending_change_request")

    if pending_change_request is not None:
        if is_approval(query):
            messages.append(HumanMessage(content=f"ユーザーが計画を承認しました。\n元のリクエスト: {pending_change_request}"))
            cl.user_session.set("pending_change_request", None)
        else:
            await cl.Message(content="[시스템] 計画がキャンセルされました。").send()
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
            # safe_llm_call에서 이미 에러 메시지를 보냈음
            return

    cl.user_session.set("messages", messages)

if __name__ == "__main__":
    if "chainlit" not in sys.argv[0]:
        sp.run([sys.executable, "-m", "chainlit", "run", __file__, "--port", "7860", "--host", "0.0.0.0", "--headless"])
