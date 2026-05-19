import sys
import subprocess
import importlib.util

# [追加] Hugging Face環境でddgsモジュールが見つからない問題を解決するためのランタイムインストール
try:
    importlib.util.find_spec("ddgs")
except (ImportError, AttributeError):
    print("[INFO] ddgsパッケージがないため、インストールを試行します...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-U", "duckduckgo-search"])

import subprocess as sp
import chainlit as cl
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from config import chat_llm, tool_llm, SYSTEM_PROMPT, create_llm, TOOL_TEMPERATURE
from tools import all_tools

# ツールバインディング
tool_llm_with_tools = tool_llm.bind_tools(all_tools)

async def safe_llm_call(messages, is_tool=False):
    """LLM呼び出しを試行し、失敗した場合はユーザーに通知します。"""
    llm_instance = chat_llm if not is_tool else tool_llm_with_tools

    # ツール呼び出しの場合はツールをバインド
    if is_tool:
        llm_instance = create_llm(temperature=TOOL_TEMPERATURE).bind_tools(all_tools)

    try:
        return await llm_instance.ainvoke(messages)
    except Exception as e:
        await cl.Message(content=f"❌ **[システムエラー]** モデル呼び出し中にエラーが発生しました: {str(e)}").send()
        raise e

# --- 既存のロジック維持 (UI日本語) ---
CHANGE_ACTION_KEYWORDS = ["修正", "直して", "リファクタリング", "パッチ", "追加", "削除", "変更", "改善", "リネーム"]
CODE_CONTEXT_KEYWORDS = ["コード", "関数", "クラス", "モジュール", "バグ", "エラー", "テスト", "lint", ".py", ".js", ".ts", ".tsx", ".jsx", ".md", ".json", ".yaml", ".yml"]
APPROVAL_WORDS = {"承認", "進行", "go", "yes", "y", "ok", "確認"}

def is_code_change_request(text: str) -> bool:
    lowered = text.lower() if text else ""
    return any(k in lowered for k in CHANGE_ACTION_KEYWORDS) and any(k in lowered for k in CODE_CONTEXT_KEYWORDS)

def is_approval(text: str) -> bool:
    return text.strip().lower() in APPROVAL_WORDS if text else False

async def generate_change_plan(user_request: str) -> str:
    planner_messages = [SystemMessage(content="あなたは計画作成器です。手順を番号付きで作成してください。"), HumanMessage(content=user_request)]
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
    """チャット開始時の初期化"""
    cl.user_session.set("messages", [SystemMessage(content=SYSTEM_PROMPT)])
    cl.user_session.set("pending_change_request", None)

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
            await cl.Message(content="[システム] 計画がキャンセルされました。").send()
            cl.user_session.set("pending_change_request", None)
            return
    elif is_code_change_request(query):
        plan_text = await generate_change_plan(query)
        await cl.Message(content=f"📋 **[変更計画]**\n{plan_text}\n\n上記計画通りに進めますか？ '承認'と入力してください。").send()   
        messages.append(HumanMessage(content=query))
        messages.append(AIMessage(content=f"[変更計画]\n{plan_text}"))
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
