"""LLM/VLM caller — OpenAI-compatible API format (Ollama, NVIDIA, OpenAI, etc.)"""

import base64
import json
import logging
import re
import httpx

log = logging.getLogger(__name__)


def _extract_json_array(text: str) -> list:
    """Extract JSON array from LLM response."""
    m = re.search(r'```(?:json)?\s*(\[.*?\])\s*```', text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            pass
    m = re.search(r'\[.*\]', text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            pass
    m = re.search(r'\{.*\}', text, re.DOTALL)
    if m:
        try:
            return [json.loads(m.group(0))]
        except Exception:
            pass
    return []


def _extract_json_object(text: str) -> dict | None:
    """Extract single JSON object from LLM response."""
    m = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            pass
    m = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', text, re.DOTALL)
    if not m:
        m = re.search(r'\{.*\}', text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            pass
    return None


class VLM:
    """Calls any OpenAI-compatible vision LLM API."""

    def __init__(self, config):
        self.cfg = config
        self.client = httpx.AsyncClient(timeout=120)

    async def _call(self, messages: list) -> str:
        """Call LLM API in OpenAI chat/completions format."""
        headers = {"Content-Type": "application/json"}
        if self.cfg.llm_api_key:
            headers["Authorization"] = f"Bearer {self.cfg.llm_api_key}"

        body = {
            "model": self.cfg.llm_model,
            "messages": messages,
            "max_tokens": 2048,
            "temperature": 0.1,
        }

        try:
            resp = await self.client.post(self.cfg.llm_url, json=body, headers=headers)
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"].strip()
        except Exception as e:
            log.error("LLM call failed: %s(%s)", type(e).__name__, e)
            return ""

    def _make_image_message(self, prompt: str, image_b64: str) -> dict:
        """Build user message with image in OpenAI vision format."""
        return {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
                {"type": "text", "text": prompt},
            ]
        }

    async def grounding_detect(self, image_bytes: bytes,
                                img_w: int = 1920, img_h: int = 1080) -> list[dict]:
        """
        Detect plants with bounding boxes.
        Returns: [{"bbox": [x1,y1,x2,y2], "type": "rose", "health": "healthy", "description": "..."}]
        """
        b64 = base64.b64encode(image_bytes).decode()
        species_list = ", ".join(self.cfg.classes)

        prompt = (
            f"Detect and locate every plant in this image. "
            f"The possible plant species are: {species_list}. "
            f"For each plant found, output a JSON array. Each element must have "
            f'"bbox_2d" (normalized 0-1000 coordinates [x1,y1,x2,y2]) and '
            f'"label" (the species name from the list above), and '
            f'"health" (healthy, mild_stress, stressed, severe_stress). '
            f"Output ONLY the JSON array, no other text."
        )

        msg = self._make_image_message(prompt, b64)
        result = await self._call([msg])
        log.info("VLM grounding (%d chars): %.500s", len(result), result)

        if not result:
            return []

        return self._parse_grounding(result, img_w, img_h)

    def _parse_grounding(self, text: str, img_w: int, img_h: int) -> list[dict]:
        """Parse grounding output into standardized results.
        Model outputs normalized 0-1000 coords, convert to pixel coords.
        """
        items = _extract_json_array(text)
        plants = []
        for item in items:
            if not isinstance(item, dict):
                continue

            bbox = (item.get("bbox_2d") or item.get("bbox") or
                    item.get("bounding_box") or item.get("box"))
            if not bbox:
                continue
            if isinstance(bbox, str):
                try:
                    bbox = json.loads(bbox)
                except (json.JSONDecodeError, ValueError):
                    continue
            if not isinstance(bbox, list) or len(bbox) != 4:
                continue

            try:
                raw = [float(v) for v in bbox]
            except (ValueError, TypeError):
                continue

            # Detect coordinate system: if all values <= 1000, treat as normalized 0-1000
            # and convert to pixel coordinates
            if all(v <= 1000 for v in raw):
                x1 = int(raw[0] / 1000 * img_w)
                y1 = int(raw[1] / 1000 * img_h)
                x2 = int(raw[2] / 1000 * img_w)
                y2 = int(raw[3] / 1000 * img_h)
            else:
                # Already pixel coordinates
                x1, y1, x2, y2 = [int(v) for v in raw]

            # clamp
            x1 = max(0, min(img_w, x1))
            y1 = max(0, min(img_h, y1))
            x2 = max(0, min(img_w, x2))
            y2 = max(0, min(img_h, y2))
            if x2 <= x1 or y2 <= y1:
                continue

            species = str(item.get("species") or item.get("type") or
                          item.get("label") or item.get("name") or "unknown")
            species = self.cfg.normalize_class(species) or species.lower().strip()

            health = str(item.get("health") or item.get("health_status") or "healthy")

            plants.append({
                "bbox": [x1, y1, x2, y2],
                "type": species,
                "health": health,
                "description": str(item.get("description", "")),
                "x_pct": round((x1 + x2) / 2 / img_w * 100),
                "y_pct": round((y1 + y2) / 2 / img_h * 100),
            })

        return plants

    async def diagnose(self, image_bytes: bytes, sensor_data: dict = None) -> dict:
        """
        Close-up diagnosis: identify species, health, and give advice.
        """
        b64 = base64.b64encode(image_bytes).decode()

        sensor_context = ""
        if sensor_data:
            parts = []
            for k, v in sensor_data.items():
                parts.append(f"{k}={v}")
            sensor_context = f"\n\nEnvironmental sensor readings: {', '.join(parts)}. Consider these when assessing plant health."

        species_list = ", ".join(self.cfg.classes)
        prompt = (
            f"Analyze this close-up plant image. "
            f"The possible species are: {species_list}. "
            f"Output bbox_2d as normalized 0-1000 coordinates [x1, y1, x2, y2] for the main plant. "
            f"Identify the species, health status "
            f"(healthy/mild_stress/stressed/severe_stress), "
            f"confidence (0.0-1.0), and a brief description of visual symptoms if any."
            f"{sensor_context}\n\n"
            f'Output as JSON: {{"bbox_2d": [...], "species": "...", "health": "...", '
            f'"confidence": 0.9, "description": "...", "is_plant": true}}'
        )

        msg = self._make_image_message(prompt, b64)
        result = await self._call([msg])
        log.info("VLM diagnose (%d chars): %.500s", len(result), result)

        r = _extract_json_object(result)
        if r:
            species = str(r.get("species") or r.get("type") or r.get("name") or "unknown")
            species = self.cfg.normalize_class(species) or species.lower().strip()
            bbox = r.get("bbox_2d") or r.get("bbox")
            return {
                "is_plant": r.get("is_plant", True),
                "type": species,
                "health": str(r.get("health") or "unknown"),
                "details": str(r.get("description") or r.get("details") or ""),
                "confidence": float(r.get("confidence", 0.8)),
                "bbox": bbox if isinstance(bbox, list) and len(bbox) == 4 else None,
            }

        return {"is_plant": False, "type": "unknown", "health": "unknown",
                "details": "LLM parse failed", "confidence": 0, "bbox": None}

    async def ask(self, prompt: str, image_bytes: bytes = None) -> str:
        """General-purpose LLM call, optionally with image."""
        if image_bytes:
            b64 = base64.b64encode(image_bytes).decode()
            msg = self._make_image_message(prompt, b64)
        else:
            msg = {"role": "user", "content": prompt}
        return await self._call([msg])
