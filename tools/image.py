import os
import time
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from huggingface_hub import AsyncInferenceClient


class ImageGenerationInput(BaseModel):
    prompt: str = Field(description="画像で描画する内容を描写する具体的な「英語」のプロンプト（名詞中心、画風を含む）")


@tool(args_schema=ImageGenerationInput)
async def generate_image(prompt: str) -> str:
    """
    ユーザーが絵、写真、画像の作成を依頼したときに使用するツールです。
    入力値(prompt)は必ず具体的な英語に翻訳および補強して渡す必要があります。
    """
    print(f"\n🎨 [システム] Qwenモデル(fal-ai)で画像を生成中... (プロンプト: {prompt})")

    client = AsyncInferenceClient(
        provider="fal-ai",
        api_key=os.environ["HF_TOKEN"],
    )

    try:
        image = await client.text_to_image(
            prompt,
            model="Qwen/Qwen-Image"
        )

        filename = f"output_{int(time.time())}.png"
        import asyncio
        await asyncio.to_thread(image.save, filename)

        return f"成功：画像が正常に生成され、'{filename}' ファイルとして保存されました。ユーザーに絵が完成したことを案内してください。"

    except Exception as e:
        return f"失敗：画像生成中にエラーが発生しました。 ({str(e)})"
