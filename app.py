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
from config import chat_llm, tool_llm, SYSTEM_PROMPT
from tools import all_tools

# ツールバインディング
tool_llm_with_tools = tool_llm.bind_tools(all_tools)

async def safe_llm_call(messages, is_tool=False):
    """LLM呼び出しを試行し、失敗した場合はユーザーに通知します。"""
    # config.pyで既に生成されたインスタンスを再利用
    llm_instance = chat_llm if not is_tool else tool_llm_with_tools

    try:
        return await llm_instance.ainvoke(messages)
    except Exception as e:
        print(f"[DEBUG] Connection Retry Failed: {str(e)}")
        await cl.Message(content=f"❌ **[システムエラー]** モデル呼び出し中にエラーが発生しました: {str(e)}").send()
        raise e

async def safe_llm_stream_process(messages, msg: cl.Message, is_tool=False):
    """GPU/ローカル環境の不安定な切断を考慮したストリーミング関数"""
    full_content = ""
    tool_calls = []
    llm_instance = tool_llm_with_tools if is_tool else chat_llm

    try:
        async for chunk in llm_instance.astream(messages):
            if hasattr(chunk, "tool_calls") and chunk.tool_calls:
                tool_calls.extend(chunk.tool_calls)
            
            if chunk.content:
                full_content += chunk.content
                await msg.stream_token(chunk.content)
                
        return AIMessage(content=full_content, tool_calls=tool_calls)

    except Exception as e:
        # 💡 [重要] すでに回答の一部を受け取っている状態で切断された場合、エラーとして扱わず生成された内容を返します。
        if full_content or tool_calls:
            print(f"[WARN] 接続が切れましたが、生成された内容は保持されます: {str(e)}")
            return AIMessage(content=full_content, tool_calls=tool_calls)
        
        # 最初から通信に失敗した場合のみ、エラーメッセージを表示します。
        print(f"[ERROR] Streaming Failed: {str(e)}")
        await cl.Message(content=f"❌ **[システムエラー]** ストリーミング中にエラーが発生しました: {str(e)}").send()
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

    for i in range(len(messages)):
        if isinstance(messages[i], HumanMessage) and isinstance(messages[i].content, list):
            text_only = "".join([item["text"] for item in messages[i].content if item.get("type") == "text"])
            messages[i] = HumanMessage(content=text_only + "\n[過去の画像はメモリ最適化のため削除されました]")

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
        await cl.Message(content=f"📋 **[変更計画]**\n{plan_text}\n\n上記計画通りに進めますか? '承認'と入力してください。").send()   
        messages.append(human_msg)
        messages.append(AIMessage(content=f"[変更計画]\n{plan_text}"))
        cl.user_session.set("pending_change_request", query)
        return
    else:
        messages.append(human_msg)

    msg = cl.Message(content="")
    await msg.send()

    try:
        response = await safe_llm_stream_process(messages, msg, is_tool=True)
        
        if response.tool_calls:
            messages.append(response)
            for tool_call in response.tool_calls:
                matched = next((t for t in all_tools if t.name == tool_call['name']), None)
                result = await matched.ainvoke(tool_call['args']) if matched else "Error: Tool not found."
                messages.append(ToolMessage(content=str(result), tool_call_id=tool_call['id']))
            
            msg_final = cl.Message(content="")
            await msg_final.send()
            final_response = await safe_llm_stream_process(messages, msg_final, is_tool=False)
            messages.append(final_response)
        else:
            messages.append(response)
            
    except Exception as e:
        print(f"[ERROR] LLM Processing Error: {str(e)}")
        # safe_llm_call 内で既にメッセージ送信済みのため、ここではログ出力のみ

    cl.user_session.set("messages", messages)

if __name__ == "__main__":
    if "chainlit" not in sys.argv[0]:
        sp.run([sys.executable, "-m", "chainlit", "run", __file__, "--port", "7860", "--host", "0.0.0.0", "--headless"])
