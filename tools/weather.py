import httpx
from pydantic import BaseModel, Field
from langchain_core.tools import tool

class WeatherResponse(BaseModel):
    location: str = Field(description="照会地域")
    temp: float = Field(description="現在の気温(Celsius)")
    feels_like: float = Field(description="体感温度(Celsius)")
    condition: str = Field(description="天候状態")
    humidity: int = Field(description="湿度(%)")
    wind_speed: float = Field(description="風速(km/h)")
    uv_index: int = Field(description="UV指数")


@tool
async def get_weather(location: str) -> dict:
    """
    特定の地域のリアルタイムの気象情報を取得します。
    気温、体感温度、湿度、風速、UV指数などの詳細な気象データを提供します。
    [🚨非常に重要🚨] location 引数は必ず「英語(English)」でのみ入力してください！
    例：刈谷市 -> "Kariya"、東京 -> "Tokyo"、ソウル -> "Seoul"
    """
    url = f"https://wttr.in/{location}?format=j1"

    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(url, timeout=10.0)
            response.raise_for_status()
            data = response.json()

        current = data['current_condition'][0]
        return WeatherResponse(
            location=location,
            temp=float(current['temp_C']),
            feels_like=float(current['FeelsLikeC']),
            condition=current['weatherDesc'][0]['value'],
            humidity=int(current['humidity']),
            wind_speed=float(current['windspeedKmph']),
            uv_index=int(current['uvIndex'])
        ).model_dump()

    except Exception as e:
        return {
            "error": "APIリクエストに失敗しました。location 引数が英語(English)に正しく翻訳されているか確認して、再試行してください。",
            "details": str(e)
        }
