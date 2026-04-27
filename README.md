# VLM-Patrol

**Self-improving plant monitoring framework** — VLM-driven detection, automatic YOLO knowledge distillation, and closed-loop care.

Deploy on any machine (Windows / Ubuntu / macOS), configure a few URLs, and the system starts collecting data, training models, and caring for plants autonomously.

## How It Works

```
Camera Image → VLM Grounding Detection → Pseudo-labels
  → Auto-split train/val → YOLO Training → Model Hot-swap
  → VLM + Sensors → Diagnosis → Care Commands → Actuator
```

The VLM (Vision Language Model) acts as a teacher: it detects plants with bounding boxes and species labels, generating pseudo-labels that train a lightweight YOLO detector. As the YOLO model improves, it provides fast inference while the VLM continues refining labels — a self-improving loop.

## Quick Start

```bash
git clone https://github.com/xiyuansun76-coder/vlm-patrol.git
cd vlm-patrol
bash setup.sh            # creates venv, installs deps, copies config
source venv/bin/activate
python main.py
```

Or manually:

```bash
python3 -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp config.example.yaml config.yaml
python main.py
```

Open `http://localhost:8765` in your browser.

## Configuration

All settings can be edited from the web UI (Settings page), or directly in `config.yaml`:

| Setting | What to fill | Purpose |
|---------|-------------|---------|
| `llm.url` | Ollama or cloud API URL | VLM detection & diagnosis |
| `llm.model` | e.g. `qwen3-vl:8b` | Which model to use |
| `camera.snapshot_url` | Any URL returning JPEG | Image source |
| `actuator.url` | HTTP endpoint (optional) | Send care commands |
| `sensor.url` | HTTP endpoint (optional) | Read sensor data |
| `classes` | List of plant species | Detection targets |

### LLM API

Any OpenAI-compatible endpoint works:

- **Local Ollama**: `http://localhost:11434/v1/chat/completions`
- **NVIDIA API**: `https://integrate.api.nvidia.com/v1/chat/completions`
- **OpenAI**: `https://api.openai.com/v1/chat/completions`

Set `LLM_API_KEY` in `.env` for cloud providers.

### Camera

Any URL that returns a JPEG image:
- IP camera snapshot endpoint
- MJPEG frame grabber
- Raspberry Pi camera server
- Even a static image URL for testing

### Actuator (Optional)

The system sends care commands as HTTP POST:

```json
{"action": "water", "enable": true, "duration_sec": 300, "reason": "Soil moisture low"}
```

Implement this endpoint on any platform: Arduino/ESP32, Raspberry Pi, PLC gateway, etc.

## Architecture

```
┌─────────────┐     ┌──────────────────────────┐     ┌──────────────┐
│ Physical    │     │ VLM-Patrol Server        │     │ Training     │
│             │     │                          │     │ (background) │
│ Camera ─────┼────→│ VLM Engine ──→ Patrol    │     │              │
│             │     │   ↓ pseudo-labels        │     │              │
│ Sensors ────┼────→│ Agent ──→ Diagnosis      │     │ YOLO Train   │
│             │     │   ↓ commands             │←────┤ (auto-trigger)│
│ Actuators ←─┼─────│ Care Execution           │     │              │
│             │     │                          │     │              │
│ Greenhouse  │     │ Web UI (localhost:8765)  │     │              │
└─────────────┘     └──────────────────────────┘     └──────────────┘
```

## Project Structure

```
vlm-patrol/
├── config.example.yaml      # Configuration template
├── .env.example              # API keys
├── requirements.txt          # Python dependencies
├── main.py                   # FastAPI server entry point
├── vlm_patrol/
│   ├── config.py             # Config loader + class aliases
│   ├── vlm.py                # LLM API caller + grounding parser
│   ├── yolo.py               # YOLO auto-detect/install + train
│   ├── patrol.py             # Patrol loop + pseudo-label collection
│   └── agent.py              # Auto-analysis + auto-care
└── static/                   # Web frontend
    ├── index.html
    ├── css/style.css
    └── js/app.js
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Web UI |
| POST | `/api/patrol/start` | Run one patrol cycle |
| GET | `/api/patrol/status` | Patrol status + YOLO info |
| POST | `/api/agent/analyze` | Run one analysis cycle |
| POST | `/api/agent/start` | Start auto-analysis loop |
| POST | `/api/agent/auto-care` | Toggle auto-care |
| POST | `/api/sensor/push` | Push sensor data (JSON) |
| POST | `/api/vlm/detect` | Upload image for VLM detection |
| POST | `/api/vlm/diagnose` | Upload image for diagnosis |
| GET/POST | `/api/config` | Read/write configuration |
| GET | `/api/yolo/status` | YOLO model & dataset info |
| POST | `/api/yolo/train` | Trigger YOLO training |
| WS | `/ws` | WebSocket for real-time updates |

## Self-Improving Loop

1. **Patrol** captures images at configured intervals
2. **VLM** detects plants with bounding boxes and species labels (pseudo-labels)
3. **Quality filter** removes tiny boxes and unknown classes
4. **Auto-split** routes 80% to train, 20% to val
5. **YOLO training** triggers automatically when dataset reaches threshold
6. **Model hot-swap** replaces the running detector with newly trained weights
7. **Next patrol** uses both YOLO (fast) and VLM (accurate) — continuous improvement
8. **Agent** periodically analyzes images + sensor data → health score + care commands
9. **Actuator** receives commands to water, ventilate, etc.

## Requirements

- Python 3.10+
- A vision-capable LLM (local Ollama or cloud API)
- A camera with HTTP snapshot capability (any IP camera, webcam server, etc.)
- Optional: IoT sensors, actuators, GPU for YOLO training

## License

MIT
