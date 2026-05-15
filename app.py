import sys
import os
import subprocess
import importlib.util

# [追加] Hugging Face環境でddgsモジュールが見つからない問題解決のためのランタイムインストール
try:
    importlib.util.find_spec("ddgs")
except (ImportError, AttributeError):
    print("[INFO] ddgsパッケージが見つからないため、インストールを試行します...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-U", "duckduckgo-search"])

import asyncio
import subprocess as sp # spとしてエイリアス設定 (既存のsubprocessとの衝突防止)
import chainlit as cl
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from config import chat_llm, tool_llm, SYSTEM_PROMPT
from tools import all_tools

tool_llm_with_tools = tool_llm.bind_tools(all_tools)

CHANGE_ACTION_KEYWORDS = [
    "修正", "直して", "リファクタリング", "パッチ", "追加", "削除", "変更", "改善", "リネーム"
]

CODE_CONTEXT_KEYWORDS = [
    "コード", "関数", "クラス", "モジュール", "バグ", "エラー", "テスト", "lint",
    ".py", ".js", ".ts", ".tsx", ".jsx", ".md", ".json", ".yaml", ".yml"
]


APPROVAL_WORDS = {"承認", "進行", "go", "yes", "y", "ok", "確認"}


def is_code_change_request(text: str) -> bool:
    lowered = text.lower()
    has_change_action = any(keyword in lowered for keyword in CHANGE_ACTION_KEYWORDS)
    has_code_context = any(keyword in lowered for keyword in CODE_CONTEXT_KEYWORDS)
    return has_change_action and has_code_context


def is_approval(text: str) -> bool:
    lowered = text.strip().lower()
    return lowered in APPROVAL_WORDS


async def generate_change_plan(user_request: str) -> str:
    planner_system = (
        "あなたはコード変更計画作成器です。 "
        "ユーザーのリクエストを見て、ツール実行前に必要な作業を3〜7段階で日本語で作成してください。 "
        "出力形式は必ず「計画」というタイトルと番号リストのみを使用し、実際の変更や実行は行わないでください。"
    )
    planner_messages = [
        SystemMessage(content=planner_system),
        HumanMessage(content=user_request),
    ]
    plan_response = await tool_llm.ainvoke(planner_messages)
    return str(plan_response.content).strip()


async def generate_final_response(messages: list) -> AIMessage:
    return await chat_llm.ainvoke(messages)


# [変更] コメントアウトを解除し、CLI用のメインループ関数を有効化
async def run_agent():
    DYNAMIC_PROMPT = SYSTEM_PROMPT + "\n[重要] 検索結果が質問を解決するには不十分であるか、無効なデータである場合は、キーワードを具体的に変更して web_search ツールを再度呼び出してください。"
    messages = [SystemMessage(content=DYNAMIC_PROMPT)]
    pending_change_request = None

    while True:
        query = input("👤 User: ")
        if query.lower() in ['exit', 'quit', 'q']:
            break
        if not query.strip():
            continue

        if pending_change_request is not None:
            if is_approval(query):
                approved_request = pending_change_request
                pending_change_request = None
                approval_note = (
                    "ユーザーが変更計画を承認しました。 "
                    "承認された範囲内でのみツールを使用して変更を実行してください。"
                )
                messages.append(HumanMessage(content=approval_note + f"\n元のリクエスト: {approved_request}"))
            else:
                print("\n[システム] 計画がキャンセルされました。他のリクエストを入力してください。\n")
                pending_change_request = None
                continue
        elif is_code_change_request(query):
            plan_text = await generate_change_plan(query)
            print("\n[変更計画]\n" + plan_text)
            print("\n[システム] この計画通りに進める場合は「承認」と入力してください。キャンセルする場合は他の入力をしてください。\n")
            messages.append(HumanMessage(content=query))
            messages.append(AIMessage(content=f"[変更計画]\n{plan_text}"))
            pending_change_request = query
            continue
        else:
            messages.append(HumanMessage(content=query))

        max_retries = 3
        current_attempt = 0
        tool_phase_completed = False

        while current_attempt < max_retries:
            response = await tool_llm_with_tools.ainvoke(messages)

            if not response.tool_calls:
                final_response = await generate_final_response(messages)
                messages.append(final_response)
                print(f"\n{final_response.content}\n")
                tool_phase_completed = True
                break

            messages.append(response)

            for tool_call in response.tool_calls:
                tool_name = tool_call['name']
                tool_args = tool_call['args']
                matched = next((t for t in all_tools if t.name == tool_name), None)
                if matched:
                    result = await matched.ainvoke(tool_args)
                else:
                    result = f"エラー：不明なツール（{tool_name}）です。"
                messages.append(ToolMessage(content=str(result), tool_call_id=tool_call['id']))

            current_attempt += 1

        if not tool_phase_completed and current_attempt == max_retries:
            print("\n[システム] 検索を数回試みましたが、完璧な情報を見つけることができませんでした。これまでの情報を基に回答を要約します。")
            final_fallback = await generate_final_response(messages)
            messages.append(final_fallback)
            print(f"{final_fallback.content}\n")


@cl.on_chat_start
async def start_chat():
    """ウェブブラウザが更新されたり、初めて接続したときに実行（初期化）"""
    DYNAMIC_PROMPT = SYSTEM_PROMPT + "\n[重要] 検索結果が質問を解決するには不十分であるか、無効なデータである場合は、キーワードを具体的に変更して web_search ツールを再度呼び出してください。"
    cl.user_session.set("messages", [SystemMessage(content=DYNAMIC_PROMPT)])
    cl.user_session.set("pending_change_request", None)


@cl.on_message
async def main(message: cl.Message):
    """ユーザーがチャット欄にメッセージを入力するたびに実行"""
    query = message.content
    
    messages = cl.user_session.get("messages")
    pending_change_request = cl.user_session.get("pending_change_request")

    if pending_change_request is not None:
        if is_approval(query):
            approved_request = pending_change_request
            cl.user_session.set("pending_change_request", None)
            approval_note = (
                "ユーザーが変更計画を承認しました。 "
                "承認された範囲内でのみツールを使用して変更を実行してください。"
            )
            messages.append(HumanMessage(content=approval_note + f"\n元のリクエスト: {approved_request}"))
        else:
            await cl.Message(content="\n[システム] 計画がキャンセルされました。他のリクエストを入力してください。\n").send()
            cl.user_session.set("pending_change_request", None)
            return
            
    elif is_code_change_request(query):
        plan_text = await generate_change_plan(query)
        res_text = f"\n[変更計画]\n{plan_text}\n\n[システム] この計画通りに進める場合は「承認」と入力してください。キャンセルする場合は他の入力をしてください。"
        await cl.Message(content=res_text).send()
        
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
    tool_phase_completed = False

    while current_attempt < max_retries:
        response = await tool_llm_with_tools.ainvoke(messages)

        if not response.tool_calls:
            final_response = await generate_final_response(messages)
            messages.append(final_response)
            
            msg.content = final_response.content
            await msg.update()
            
            tool_phase_completed = True
            break

        messages.append(response)

        for tool_call in response.tool_calls:
            tool_name = tool_call['name']
            tool_args = tool_call['args']
            matched = next((t for t in all_tools if t.name == tool_name), None)
            
            if matched:
                result = await matched.ainvoke(tool_args)
            else:
                result = f"エラー：不明なツール（{tool_name}）です。"
                
            messages.append(ToolMessage(content=str(result), tool_call_id=tool_call['id']))

        current_attempt += 1

    if not tool_phase_completed and current_attempt == max_retries:
        fallback_msg = "\n[システム] 検索を数回試みましたが、完璧な情報を見つけることができませんでした。これまでの情報を基に回答を要約します。\n\n"
        final_fallback = await generate_final_response(messages)
        messages.append(final_fallback)
        
        msg.content = fallback_msg + final_fallback.content
        await msg.update()

    cl.user_session.set("messages", messages)


# [変更] 実行時に 1(CLI) か 2(GUI) を選択して起動するロジック
if __name__ == "__main__":
    if "chainlit" not in sys.argv[0]:
        # 環境変数 RUN_MODE が 'GUI' なら即座に起動 (Docker用)

        run_mode = os.getenv("RUN_MODE", "").upper()

        if run_mode == "GUI":
            choice = "2"
        elif run_mode == "CLI":
            choice = "1"
        else:
            print("実行モードを選択してください (EnterでGUI):")
            print("1: ターミナル (CLI)")
            print("2: Web GUI (Chainlit)")
            choice = input("入力 (1 または 2): ").strip() or "2"


        if choice == "2":
            print("\n[INFO] GUIモードを起動します... (Chainlit サーバー開始)")
            # 7860ポートはHugging Face Spacesのデフォルトポートです。
            # --host 0.0.0.0 을 추가하여 외부 접속을 허용하고, --headless 로 브라우저 실행을 방지합니다.
            sp.run([sys.executable, "-m", "chainlit", "run", __file__, "--port", "7860", "--host", "0.0.0.0", "--headless"])
        else:

            print("\n[INFO] ターミナル(CLI)モードで実行します。")
            asyncio.run(run_agent())