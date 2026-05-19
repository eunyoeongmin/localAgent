import asyncio
from pydantic import BaseModel, Field
from langchain_core.tools import tool

class SearchInput(BaseModel):
    query: str = Field(description="検索する2〜3個の核心名詞キーワード")
    constraints: str = Field(description="ユーザーが要求した追加条件（例：'2026年の資料のみ'、'特定の結果を除外')", default="")

@tool(args_schema=SearchInput)
async def web_search(query: str, constraints: str = "") -> str:
    """インターネットで最新ニュースや2025年以降の情報を検索します。
    ユーザーの追加条件(constraints)がある場合は、検索キーワードの組み合わせに 積極的に反映させてください。
    """
    # [変更] インポート時のエラーを防止するため、関数内でツールを初期化 (Lazy Loading)
    from langchain_community.tools import DuckDuckGoSearchRun
    search_tool = DuckDuckGoSearchRun()

    refined_query = f"{query} {constraints}".strip()
    return await asyncio.to_thread(search_tool.invoke, refined_query)
