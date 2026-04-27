import json
import urllib.request
import urllib.error
from fastapi import APIRouter, HTTPException, Header
from pydantic import BaseModel
from typing import Optional, List
import database as db

router = APIRouter(prefix="/api/agent", tags=["agent"])

_NVIDIA_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
_MODEL      = "mistralai/mistral-large-3-675b-instruct-2512"


class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[Message]


def _sensor_context() -> str:
    rows = db.query_latest()
    m = {}
    for row in rows:
        for k, v in row.items():
            if v is not None and k not in ("id", "raw_json", "topic", "node_id"):
                m[k] = v
    return json.dumps(m, ensure_ascii=False, default=str)


def _system_prompt() -> str:
    ctx = _sensor_context()
    return f"""你是 SmartAgri 智慧温室的 AI 助手，实时接入温室传感器与设备数据。

【当前传感器快照】
{ctx}

【设备映射】
- light  → 补光灯
- pump   → 灌溉水泵
- curtain→ 灌溉水管阀门
- fan    → 通风风扇

【你的职责】
1. 用中文回答关于温室环境、土壤状态、设备状态的问题
2. 根据传感器数据给出专业农业建议
3. 解释各传感器读数的含义和参考范围
   - CO₂ 正常范围：400–1500 ppm，>2000 需通风
   - 土壤湿度：30–70% 为宜，<20% 需浇水
   - 土壤温度：15–28°C 为宜
   - 氮(N)/磷(P)/钾(K)：根据读数评估肥力
4. 当用户询问是否需要操作设备时，给出建议并说明原因

回复风格：简洁、专业、直接，使用中文。如有需要可适当使用数据表格。"""


@router.post("/chat")
async def chat(req: ChatRequest, authorization: Optional[str] = Header(None)):
    from config import API_TOKEN, NVIDIA_API_KEY
    if authorization != f"Bearer {API_TOKEN}":
        raise HTTPException(status_code=401, detail="Unauthorized")

    messages = [{"role": "system", "content": _system_prompt()}]
    messages += [{"role": m.role, "content": m.content} for m in req.messages]

    payload = json.dumps({
        "model": _MODEL,
        "messages": messages,
        "max_tokens": 1024,
        "temperature": 0.65,
        "stream": False,
    }).encode()

    request = urllib.request.Request(
        _NVIDIA_URL,
        data=payload,
        headers={
            "Authorization": f"Bearer {NVIDIA_API_KEY}",
            "Content-Type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as resp:
            result = json.loads(resp.read())
            content = result["choices"][0]["message"]["content"]
            return {"content": content}
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="ignore")
        raise HTTPException(status_code=502, detail=f"LLM error {e.code}: {body[:200]}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
