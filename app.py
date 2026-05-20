import sys
import subprocess
import importlib.util
import traceback

# [追加] Hugging Face環境でddgsモジュールが見つからない問題を解決するためのランタイムインストール
try:
    importlib.util.find_spec("ddgs")
except (ImportError, AttributeError):
    print("[INFO] ddgsパッケージがないため、インストールを試行します...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-U", "duckduckgo-search"])

import subprocess as sp
import chainlit as cl
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from config import chat_llm, tool_llm, SYSTEM_PROMPT
from tools import all_tools

# ツールバインディング
tool_llm_with_tools = tool_llm.bind_tools(all_tools)

async def safe_llm_call(messages, is_tool=False):
    """LLM呼び出しを試行します。"""
    llm_instance = chat_llm if not is_tool else tool_llm_with_tools
    try:
        print(f"[DEBUG] Initiating LLM Call (is_tool={is_tool})")
        return await llm_instance.ainvoke(messages)
    except Exception as e:
        print("--- [LLM CALL ERROR] ---")
        traceback.print_exc()
        print("--- [ERROR END] ---")
        await cl.Message(content=f"❌ **[システムエラー]** {str(e)}").send()
        raise e

async def safe_llm_stream_process(messages, msg: cl.Message, is_tool=False):
    """astreamを使用してLLM応答をストリーミング処理します。"""
    full_content = ""
    tool_calls = []
    llm_instance = chat_llm if not is_tool else tool_llm_with_tools

    try:
        print(f"[DEBUG] Initiating Stream Call (is_tool={is_tool})")
        async for chunk in llm_instance.astream(messages):
            if hasattr(chunk, "tool_calls") and chunk.tool_calls:
                for tc in chunk.tool_calls:
                    if not any(existing['id'] == tc['id'] for existing in tool_calls):
                        tool_calls.append(tc)
            
            if chunk.content:
                full_content += chunk.content
                await msg.stream_token(chunk.content)
                
        return AIMessage(content=full_content, tool_calls=tool_calls)
    except Exception as e:
        print(f"--- [STREAM ERROR] ---")
        traceback.print_exc()
        print(f"--- [ERROR END] ---")
        if full_content or tool_calls:
            print(f"[WARN] Connection dropped, but content preserved: {str(e)}")
            return AIMessage(content=full_content, tool_calls=tool_calls)
        await cl.Message(content=f"❌ **[システムエラー]** {str(e)}").send()
        raise e

# --- 既存のロジック維持 (UI日本語) ---
CHANGE_ACTION_KEYWORDS = ["修正", "直して", "リファクタリング", "パッチ", "追加", "削除", "変更", "改善", "リ네임"]
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

    texts = [query] if query else []
    images = []

    if message.elements:
        for element in message.elements:
            if "image" in element.mime:
                import base64
                with open(element.path, "rb") as f:
                    base64_image = base64.b64encode(f.read()).decode("utf-8")
                images.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:{element.mime};base64,{base64_image}"}
                })
            elif any(ext in element.mime for ext in ["text", "json", "javascript", "python"]):
                with open(element.path, "r", encoding="utf-8", errors="ignore") as f:
                    file_content = f.read()
                    max_chars = 2000
                    if len(file_content) > max_chars:
                        file_content = file_content[:max_chars] + f"\n\n...[内容が長すぎるため、{max_chars}文字でカットされました]..."
                    texts.append(f"\n[添付ファイル: {element.name}]\n{file_content}")

    combined_text = "\n".join(texts)
    content = []
    if combined_text:
        content.append({"type": "text", "text": combined_text})
    content.extend(images)

    if not content:
        return

    human_msg = HumanMessage(content=content if images else combined_text)

    # 過去のメッセージから画像を削除（メモリ節約）
    for i in range(len(messages)):
        if isinstance(messages[i], HumanMessage) and isinstance(messages[i].content, list):
            text_only = "".join([item["text"] for item in messages[i].content if item.get("type") == "text"])
            messages[i] = HumanMessage(content=text_only + "\n[過去の画像はメモリ最適化のため削除されました]")

    if pending_change_request is not None:
        if is_approval(query):
            messages.append(HumanMessage(content=f"ユーザー가 계획을 승인했습니다.\n원래 요청: {pending_change_request}"))
            cl.user_session.set("pending_change_request", None)
        else:
            await cl.Message(content="[시스템] 계획이 취소되었습니다.").send()
            cl.user_session.set("pending_change_request", None)
            return
    elif is_code_change_request(query):
        plan_text = await generate_change_plan(query)
        await cl.Message(content=f"📋 **[변경 계획]**\n{plan_text}\n\n위 계획대로 진행할까요? '승인'이라고 입력해주세요.").send()   
        messages.append(human_msg)
        messages.append(AIMessage(content=f"[변경 계획]\n{plan_text}"))
        cl.user_session.set("pending_change_request", query)
        return
    else:
        messages.append(human_msg)

    msg = cl.Message(content="")
    await msg.send()

    max_retries = 3
    current_attempt = 0
    while current_attempt < max_retries:
        try:
            # 1. 툴 호출 확인
            response = await safe_llm_call(messages, is_tool=True)
            
            if not response.tool_calls:
                # 2. 최종 응답 (스트리밍)
                final_response = await safe_llm_stream_process(messages, msg, is_tool=False)
                messages.append(final_response)
                msg.content = extract_content(final_response.content)
                await msg.update()
                break

            # 3. 툴 실행
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
