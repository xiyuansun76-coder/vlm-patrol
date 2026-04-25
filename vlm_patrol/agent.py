"""Agent — auto analysis (image+sensor→diagnosis→commands) and auto care."""

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime

import httpx

log = logging.getLogger(__name__)


@dataclass
class Command:
    action: str         # water, light, vent, etc.
    enable: bool = True
    reason: str = ""
    duration_sec: int = 0
    state: str = "pending"  # pending, executed, dismissed

    def to_dict(self):
        return {
            "action": self.action, "enable": self.enable,
            "reason": self.reason, "duration_sec": self.duration_sec,
            "state": self.state,
        }


@dataclass
class AnalysisResult:
    timestamp: str = ""
    health_score: int = 0
    summary: str = ""
    commands: list = field(default_factory=list)
    vision_summary: str = ""

    def to_dict(self):
        return {
            "timestamp": self.timestamp, "health_score": self.health_score,
            "summary": self.summary, "vision_summary": self.vision_summary,
            "commands": [c.to_dict() for c in self.commands],
        }


class Agent:
    """Auto analysis + auto care loop."""

    def __init__(self, config, vlm, patrol):
        self.cfg = config
        self.vlm = vlm
        self.patrol = patrol
        self.http = httpx.AsyncClient(timeout=15)
        self._running = False
        self.history: list[AnalysisResult] = []
        self.auto_care = False
        self.sensor_data: dict = {}

    async def fetch_sensor_data(self) -> dict:
        """Fetch sensor data from configured URL, or return cached."""
        if self.cfg.sensor_url:
            try:
                resp = await self.http.get(self.cfg.sensor_url)
                resp.raise_for_status()
                self.sensor_data = resp.json()
            except Exception as e:
                log.warning("Sensor fetch failed: %s", e)
        return self.sensor_data

    def update_sensor_data(self, data: dict):
        """Push sensor data (e.g. from MQTT or WebSocket)."""
        self.sensor_data.update(data)

    async def analyze_once(self) -> AnalysisResult:
        """Run one analysis cycle: snapshot → VLM diagnose → parse commands."""
        result = AnalysisResult(timestamp=datetime.now().isoformat())

        # 1. Get sensor data
        sensor = await self.fetch_sensor_data()

        # 2. Get camera image
        image = await self.patrol.snapshot()

        # 3. VLM vision analysis
        vision_summary = ""
        if image:
            diag = await self.vlm.diagnose(image, sensor)
            vision_summary = (
                f"Species: {diag['type']}, Health: {diag['health']}, "
                f"Confidence: {diag['confidence']:.1%}. {diag['details']}"
            )
            result.vision_summary = vision_summary

        # 4. Ask LLM for decision + commands
        species_list = ", ".join(self.cfg.classes)
        sensor_text = json.dumps(sensor, ensure_ascii=False) if sensor else "No sensor data"

        decision_prompt = f"""You are a greenhouse management AI. Based on the following information, provide:
1. A health score (0-100)
2. A brief summary of plant status
3. Action commands if needed

Plant species in greenhouse: {species_list}
Sensor data: {sensor_text}
Vision analysis: {vision_summary if vision_summary else "No camera available"}

Output as JSON:
{{"health_score": 80, "summary": "Plants are generally healthy...",
  "commands": [{{"action": "water", "enable": true, "reason": "Soil moisture low", "duration_sec": 300}}]}}

Possible actions: water, light, vent, cooling, heating.
Only include commands that are actually needed. Output ONLY JSON."""

        resp = await self.vlm.ask(decision_prompt)
        log.info("Agent decision (%d chars): %.500s", len(resp), resp)

        # 5. Parse response
        try:
            import re
            m = re.search(r'\{.*\}', resp, re.DOTALL)
            if m:
                data = json.loads(m.group(0))
                result.health_score = int(data.get("health_score", 0))
                result.summary = data.get("summary", "")
                for cmd in data.get("commands", []):
                    result.commands.append(Command(
                        action=cmd.get("action", ""),
                        enable=cmd.get("enable", True),
                        reason=cmd.get("reason", ""),
                        duration_sec=cmd.get("duration_sec", 0),
                    ))
        except Exception as e:
            log.error("Failed to parse decision: %s", e)
            result.summary = resp

        # 6. Auto-care: execute commands if enabled
        if self.auto_care:
            for cmd in result.commands:
                await self._execute_command(cmd)

        self.history.append(result)
        # keep last 48 entries
        if len(self.history) > 48:
            self.history = self.history[-48:]

        return result

    async def _execute_command(self, cmd: Command):
        """
        Execute a care command by POSTing to the configured actuator URL.
        If no actuator_url is configured, just log the command.

        The actuator endpoint receives JSON:
          {"action": "water", "enable": true, "duration_sec": 300, "reason": "..."}
        and should return {"status": "ok"} or {"status": "error", "error": "..."}.

        Users can implement this endpoint on any platform:
        - Arduino/ESP32 HTTP server
        - Raspberry Pi Flask app
        - PLC gateway
        - Or any device that accepts HTTP POST
        """
        log.info("Command: %s (enable=%s, duration=%ds, reason=%s)",
                 cmd.action, cmd.enable, cmd.duration_sec, cmd.reason)

        actuator_url = getattr(self.cfg, 'actuator_url', '')
        if actuator_url:
            try:
                resp = await self.http.post(actuator_url, json={
                    "action": cmd.action,
                    "enable": cmd.enable,
                    "duration_sec": cmd.duration_sec,
                    "reason": cmd.reason,
                })
                resp.raise_for_status()
                result = resp.json()
                if result.get("status") == "ok":
                    cmd.state = "executed"
                    log.info("Command executed via %s", actuator_url)
                else:
                    cmd.state = "error"
                    log.warning("Actuator error: %s", result.get("error"))
            except Exception as e:
                cmd.state = "error"
                log.error("Actuator call failed: %s", e)
        else:
            cmd.state = "executed"
            log.info("No actuator_url configured, command logged only")

    async def run_continuous(self):
        """Run auto-analysis loop at configured interval."""
        self._running = True
        while self._running:
            try:
                result = await self.analyze_once()
                log.info("Analysis done: score=%d, %d commands",
                         result.health_score, len(result.commands))
            except Exception as e:
                log.error("Analysis error: %s", e)
            await asyncio.sleep(self.cfg.agent_interval * 60)

    def stop(self):
        self._running = False

    def get_status(self) -> dict:
        latest = self.history[-1].to_dict() if self.history else None
        return {
            "running": self._running,
            "auto_care": self.auto_care,
            "interval_minutes": self.cfg.agent_interval,
            "history_count": len(self.history),
            "latest": latest,
        }
