import asyncio
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun


class SearchInput(BaseModel):
    query: str = Field(description="検索する2〜3個の核心名詞キーワード")
    constraints: str = Field(description="ユーザーが要求した追加条件（例：'2026年の資料のみ'、'特定の結果を除外')", default="")


_search_tool = DuckDuckGoSearchRun()


@tool(args_schema=SearchInput)
async def web_search(query: str, constraints: str = "") -> str:
    """インターネットで最新ニュースや2025年以降の情報を検索します。
    ユーザーの追加条件(constraints)がある場合は、検索キーワードの組み合わせに積極的に反映させてください。
    """
    refined_query = f"{query} {constraints}".strip()
    return await asyncio.to_thread(_search_tool.invoke, refined_query)
