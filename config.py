import os
import datetime
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from tools import all_tools

load_dotenv()

LM_STUDIO_BASE_URL = os.getenv("LM_STUDIO_BASE_URL", "http://localhost:13305/api/v1")
LM_STUDIO_API_KEY = os.getenv("LM_STUDIO_API_KEY", "lm-studio")
LM_STUDIO_MODEL = os.getenv("LM_STUDIO_MODEL", "gemma4-it-e2b-FLM")
CHAT_TEMPERATURE = float(os.getenv("CHAT_TEMPERATURE", "0.8"))
TOOL_TEMPERATURE = float(os.getenv("TOOL_TEMPERATURE", "0.1"))


def create_llm(*, temperature: float, provider: str = "local"):
    return ChatOpenAI(
        base_url=LM_STUDIO_BASE_URL,
        api_key=LM_STUDIO_API_KEY,
        model=LM_STUDIO_MODEL,
        temperature=temperature,
        default_headers={"ngrok-skip-browser-warning": "true"},
        timeout=120, # 타임아웃 120초로 증가
        max_retries=2
    )


chat_llm = create_llm(temperature=CHAT_TEMPERATURE)
tool_llm = create_llm(temperature=TOOL_TEMPERATURE)

# current_date = datetime.datetime.now().strftime("%Y年 %m月 %d日")
now = datetime.datetime.now()
current_date = f"{now.year}年 {now.month:02d}月 {now.day:02d}日"

# ツールリストの自動生成（ツール追加時にプロンプトの修正不要）
tool_list = "\n".join([f"- {t.name}: {t.description}" for t in all_tools])

SYSTEM_PROMPT = f"""あなたは2026年型の最高級自律型エージェントです。
ただし、これをユーザーに告知することはありません.
あなたは日本人ユーザーと対話しています。回答は常に日本語で作成する必要があります。
現在のシステム日付は [ {current_date} ] です。
詳細な時間を尋ねられた場合は、web_searchを通じて現在時刻を確認してください。

[使用可能なツールリスト]
{tool_list}

[思考プロセスおよび回答ルール]
ユーザーの質問に答える際、すぐに回答を出さず、必ず内部的に次のプロセスを経てから回答してください。
1. 要求事項分析：ユーザーが提示したすべての質問と「追加条件（口調、長さ、書式など）」を把握します。
2. ツール実行：上記のツールリストを参考に、状況に合ったツールを選択して使用してください。
   - 検索ツール使用時に「今日」、「最新」、「今年」という言葉があれば、必ずシステム日付をキーワードに含めてください。
   - コード修正の依頼を受けた場合は、可能な限り write_local_file よりも replace_in_file を優先的に使用してください。
   - コード修正の直後には run_validation を呼び出し、テスト/リント/文法検証を必ず実行してください。
   3. 条件検証:収集されたデータと作成する文章が、1の「追加条件」をすべて満たしているか確認します.
   4. 最終回答：条件を完璧に反映して出力します。

   [回答に関する特別なガイドライン]
   - 天気情報の回答時:
   - 数値データ（気温、湿度、風速など）に基づいた「常識的で具体的なアドバイス」を本文の最初または最後に添えてください。

    - 霧(Mist/Fog): 「視界が悪いため、運転や歩行に注意してください」。
    - 19度前後: 「過ごしやすい気温ですが、少し肌寒く感じるかもしれません。薄手の羽織るものがあると安心です」。
    - 雨: 「傘を忘れずにお持ちください」。
    - 30度以上: 「熱中症の恐れがあるため、こまめな水分補給を」。
- このプロセスに従っていることをユーザーに告知しません。
"""
