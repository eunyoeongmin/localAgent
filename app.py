import sys
import subprocess
import importlib.util
import base64

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

# --- 既存のロジック維持 ---
CHANGE_ACTION_KEYWORDS = ["修正", "直して", "リファクタリング", "パッチ", "追加", "削除", "変更", "改善", "リネーム"]
CODE_CONTEXT_KEYWORDS = ["コード", "関数", "クラス", "モジュール", "バグ", "エラー", "テスト", "lint", ".py", ".js", ".ts", ".tsx", ".jsx", ".md", ".json", ".yaml", ".yml"]
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
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join([item['text'] if isinstance(item, dict) and 'text' in item else str(item) for item in content])
    return str(content)

@cl.on_chat_start
async def start_chat():
    """チャット開始時の初期化"""
    DYNAMIC_PROMPT = SYSTEM_PROMPT + "\n[重要] 検索結果が不十分な場合はキーワードを変更して再検索してください。"
    cl.user_session.set("messages", [SystemMessage(content=DYNAMIC_PROMPT)])
    cl.user_session.set("pending_change_request", None)

@cl.on_message
async def main(message: cl.Message):
    query = message.content
    messages = cl.user_session.get("messages")
    pending_change_request = cl.user_session.get("pending_change_request")

    # [追加] ファイルアップロード処理
    content_list = []
    if query:
        content_list.append({"type": "text", "text": query})

    if message.elements:
        for element in message.elements:
            # 画像処理 (Vision)
            if "image" in element.mime:
                with open(element.path, "rb") as f:
                    base64_image = base64.b64encode(f.read()).decode("utf-8")
                content_list.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:{element.mime};base64,{base64_image}"}
                })
            # テキストファイル処理
            elif "text" in element.mime or "json" in element.mime or "application/javascript" in element.mime:
                with open(element.path, "r", encoding="utf-8", errors="ignore") as f:
                    file_content = f.read()
                content_list.append({
                    "type": "text", 
                    "text": f"\n[添付ファイル: {element.name}]\n{file_content}"
                })

    # HumanMessage生成 (マルチモーダル対応)
    if not content_list:
        return
    
    # テキストのみの場合とマルチモーダル（画像含む）の場合を区別
    human_msg = HumanMessage(content=content_list if any(c['type'] == 'image_url' for c in content_list) else query)

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
        messages.append(human_msg)
        messages.append(AIMessage(content=f"[変更計画]\n{plan_text}"))
        cl.user_session.set("pending_change_request", query)
        return
    else:
        messages.append(human_msg)

    msg = cl.Message(content="")
    await msg.send()

    max_retries = 3
    current_attempt = 0
    full_response_content = ""
    
    while current_attempt < max_retries:
        try:
            # ツール呼び出しの有無を確認 (ツール呼び出しはストリーミングしない)
            response = await safe_llm_call(messages, is_tool=True)
            
            if not response.tool_calls:
                # 最終回答の場合はストリーミングで出力
                async for chunk in chat_llm.astream(messages):
                    if chunk.content:
                        full_response_content += chunk.content
                        await msg.stream_token(chunk.content)
                
                messages.append(AIMessage(content=full_response_content))
                await msg.send()
                break
            
            # ツール呼び出しの場合の処理
            messages.append(response)
            for tool_call in response.tool_calls:
                matched = next((t for t in all_tools if t.name == tool_call['name']), None)
                result = await matched.ainvoke(tool_call['args']) if matched else "Error: Tool not found."
                messages.append(ToolMessage(content=str(result), tool_call_id=tool_call['id']))
            current_attempt += 1
        except Exception as e:
            await cl.Message(content=f"❌ **[システムエラー]** ストリーミング中にエラーが発生しました: {str(e)}").send()
            return

    cl.user_session.set("messages", messages)

if __name__ == "__main__":
    if "chainlit" not in sys.argv[0]:
        sp.run([sys.executable, "-m", "chainlit", "run", __file__, "--port", "7860", "--host", "0.0.0.0", "--headless"])
