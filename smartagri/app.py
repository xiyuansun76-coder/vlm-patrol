"""SmartAgri web application entry point.

Run: uvicorn smartagri.app:app --host 0.0.0.0 --port 8000
Or:  python -m smartagri.app
"""
import hmac
import hashlib
import logging
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, RedirectResponse

# Ensure smartagri package is importable
sys.path.insert(0, str(Path(__file__).parent))

import database as db
import mqtt_bridge
import cloud_sync
import control_state
from config import API_TOKEN, LOGIN_USER, LOGIN_PASS, SESSION_SECRET
from routers import (sensors, devices, camera,
                     stream as stream_router, agent as agent_router,
                     hikvision as hik_router, patrol as patrol_router,
                     vlp as vlp_router)

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

_SESSION_TOKEN = hmac.new(
    SESSION_SECRET.encode(), b"authenticated", hashlib.sha256
).hexdigest()

TEMPLATES_DIR = Path(__file__).parent / "templates"


def _valid_session(request: Request) -> bool:
    return request.cookies.get("sid") == _SESSION_TOKEN


@asynccontextmanager
async def lifespan(app: FastAPI):
    db.init_db()
    log.info("Database initialized")
    control_state.init_from_db()
    mqtt_bridge.on_sensor_update = stream_router.broadcast
    mqtt_bridge.start(block=False)
    cloud_sync.flush_pending()
    yield
    log.info("Shutting down")


app = FastAPI(title="SmartAgri", lifespan=lifespan)

app.include_router(sensors.router)
app.include_router(devices.router)
app.include_router(camera.router)
app.include_router(stream_router.router)
app.include_router(agent_router.router)
app.include_router(hik_router.router)
app.include_router(patrol_router.router)
app.include_router(vlp_router.router)


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    if not _valid_session(request):
        return RedirectResponse("/login", status_code=302)
    html = (TEMPLATES_DIR / "index.html").read_text(encoding="utf-8")
    html = html.replace('"{{ token }}"', f'"{API_TOKEN}"')
    return HTMLResponse(html, headers={"Cache-Control": "no-cache, no-store, must-revalidate"})


@app.get("/login", response_class=HTMLResponse)
def login_page(request: Request):
    if _valid_session(request):
        return RedirectResponse("/", status_code=302)
    return HTMLResponse((TEMPLATES_DIR / "login.html").read_text(encoding="utf-8"))


@app.post("/login")
async def login(
    request: Request,
    username: str = Form(...),
    password: str = Form(...),
):
    if username == LOGIN_USER and password == LOGIN_PASS:
        response = RedirectResponse("/", status_code=302)
        response.set_cookie(
            key="sid",
            value=_SESSION_TOKEN,
            httponly=True,
            samesite="lax",
            max_age=86400 * 7,
        )
        return response
    html = (TEMPLATES_DIR / "login.html").read_text(encoding="utf-8")
    html = html.replace("<!--ERROR_PLACEHOLDER-->",
                        '<p id="login-err" class="text-error text-xs text-center font-mono tracking-wide">Invalid credentials</p>')
    return HTMLResponse(html, status_code=401)


@app.post("/logout")
def logout():
    response = RedirectResponse("/login", status_code=302)
    response.delete_cookie("sid")
    return response


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("SMARTAGRI_PORT", "8000"))
    uvicorn.run("smartagri.app:app", host="0.0.0.0", port=port, reload=False)
