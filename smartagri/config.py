"""SmartAgri configuration — all values from environment or .env file."""
import os
from pathlib import Path

# Load .env from project root if exists
_env_file = Path(__file__).parent.parent / ".env"
if _env_file.exists():
    for line in _env_file.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

BROKER_HOST    = os.environ.get("MQTT_BROKER_HOST", "127.0.0.1")
BROKER_PORT    = int(os.environ.get("MQTT_BROKER_PORT", "1883"))
DB_PATH        = os.environ.get("SMARTAGRI_DB_PATH", "smartagri.db")
API_PORT       = int(os.environ.get("SMARTAGRI_PORT", "8000"))
API_TOKEN      = os.environ.get("SMARTAGRI_API_TOKEN", "change-me-token")

NVIDIA_API_KEY = os.environ.get("NVIDIA_API_KEY", "")

LOGIN_USER     = os.environ.get("SMARTAGRI_USER", "admin")
LOGIN_PASS     = os.environ.get("SMARTAGRI_PASS", "admin")
SESSION_SECRET = os.environ.get("SMARTAGRI_SESSION_SECRET", "change-me-session-secret")

# Supabase (optional cloud sync)
SUPABASE_URL   = os.environ.get("SUPABASE_URL", "")
SUPABASE_KEY   = os.environ.get("SUPABASE_KEY", "")
