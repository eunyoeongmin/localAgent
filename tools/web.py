import httpx
from pydantic import BaseModel, Field
from langchain_core.tools import tool


class WebAnalysisInput(BaseModel):
    url: str = Field(description="分析するウェブサイトのURL")
    query: str = Field(description="ウェブサイトから探したい内容や質問")


@tool(args_schema=WebAnalysisInput)
async def analyze_website(url: str, query: str) -> str:
    """
    ユーザーが特定のウェブサイトリンク(URL)を提示しながら、要約や質問をしたときに使用するツールです。
    Jina AI Readerを通じてJSレンダリングサイト、ニュース再配布サイトなどの本文も抽出します。
    """
    try:
        jina_url = f"https://r.jina.ai/{url}"
        print(f"\n[DEBUG] Jina AIリクエスト: {jina_url}")

        async with httpx.AsyncClient() as client:
            response = await client.get(
                jina_url,
                headers={"Accept": "text/plain"},
                timeout=30.0
            )
            response.raise_for_status()
            text = response.text

        print(f"[DEBUG] Jina AI抽出完了 - 文字数: {len(text)}")
        print(f"[DEBUG] プレビュー:\n{text[:300]}\n")

        if not text.strip():
            return "ページから本文テキストを抽出できませんでした。"

        return f"[ウェブサイト抽出内容]\n{text[:8000]}"

    except Exception as e:
        print(f"[DEBUG] エラー発生: {str(e)}")
        return f"ウェブサイト分析中にエラーが発生しました: {str(e)}"
