"""PTZ dome camera control via Hikvision ISAPI protocol.

Works with most Hikvision / Dahua / ONVIF-compatible dome cameras
that support absolute positioning through ISAPI HTTP interface.

All parameters come from config.yaml — no hardcoded values.
"""

import math
import re
import logging
import asyncio
import httpx

log = logging.getLogger(__name__)


class PTZController:
    """Hikvision ISAPI PTZ controller. All params from config."""

    def __init__(self, config):
        self.cfg = config
        self.client = httpx.AsyncClient(
            auth=httpx.DigestAuth(config.ptz_user, config.ptz_pass),
            timeout=8,
        )
        self.base_url = config.ptz_url.rstrip("/")
        self.current_az = config.ptz_home_az
        self.current_el = config.ptz_home_el
        self.current_zoom = config.ptz_wide_zoom
        self.total_travel = 0.0

    # ── Position ──

    async def get_position(self) -> tuple[int, int, int]:
        """Read current az/el/zoom from camera."""
        try:
            r = await self.client.get(self.base_url + "/ISAPI/PTZCtrl/channels/1/status")
            if r.status_code == 200:
                az = int(re.search(r"<azimuth>(\d+)</azimuth>", r.text).group(1))
                el = int(re.search(r"<elevation>(\d+)</elevation>", r.text).group(1))
                zm = int(re.search(r"<absoluteZoom>(\d+)</absoluteZoom>", r.text).group(1))
                self.current_az, self.current_el, self.current_zoom = az, el, zm
                return az, el, zm
        except Exception as e:
            log.error("Failed to get PTZ position: %s", e)
        return self.current_az, self.current_el, self.current_zoom

    @staticmethod
    def calc_distance(az1: int, el1: int, az2: int, el2: int) -> float:
        """Angular distance in degrees between two PTZ positions."""
        daz = abs(az2 - az1)
        if daz > 1800:
            daz = 3600 - daz
        del_ = abs(el2 - el1)
        return math.sqrt((daz / 10.0) ** 2 + (del_ / 10.0) ** 2)

    # ── Movement ──

    async def goto(self, az: int, el: int, zoom: int, wait: float = 2.5) -> bool:
        """Move to absolute position. Units: az/el in 0.1°, zoom 10-100."""
        az = max(0, min(3600, az))
        el = max(0, min(900, el))
        zoom = max(10, min(100, zoom))

        dist = self.calc_distance(self.current_az, self.current_el, az, el)
        self.total_travel += dist

        xml = f'''<?xml version="1.0" encoding="UTF-8"?>
<PTZData version="2.0" xmlns="http://www.hikvision.com/ver20/XMLSchema">
<AbsoluteHigh>
<elevation>{el}</elevation>
<azimuth>{az}</azimuth>
<absoluteZoom>{zoom}</absoluteZoom>
</AbsoluteHigh>
</PTZData>'''
        try:
            r = await self.client.put(
                self.base_url + "/ISAPI/PTZCtrl/channels/1/absolute",
                content=xml.encode(),
                headers={"Content-Type": "application/xml"},
            )
            if r.status_code < 300:
                self.current_az, self.current_el, self.current_zoom = az, el, zoom
                await asyncio.sleep(wait)
                log.info("PTZ goto az=%d el=%d zoom=%d (dist=%.1f°)", az, el, zoom, dist)
                return True
        except Exception as e:
            log.error("PTZ move failed: %s", e)
        return False

    async def go_home(self) -> bool:
        return await self.goto(self.cfg.ptz_home_az, self.cfg.ptz_home_el,
                               self.cfg.ptz_wide_zoom, wait=3)

    def reset_travel(self):
        self.total_travel = 0.0

    # ── Coordinate conversion ──

    def bbox_to_ptz(self, bbox: list[int],
                    ref_az: int = None, ref_el: int = None) -> tuple[int, int]:
        """
        Convert pixel bounding box center to PTZ absolute coordinates.
        bbox: [x1, y1, x2, y2] in pixels
        Returns: (target_az, target_el)
        """
        if ref_az is None:
            ref_az = self.current_az
        if ref_el is None:
            ref_el = self.current_el

        img_w = self.cfg.ptz_img_w
        img_h = self.cfg.ptz_img_h
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2

        dx_frac = (cx - img_w / 2) / img_w
        dy_frac = (cy - img_h / 2) / img_h

        daz = int(dx_frac * self.cfg.ptz_fov_h * 10)
        del_ = int(-dy_frac * self.cfg.ptz_fov_v * 10)

        target_az = (ref_az + daz) % 3600
        target_el = max(0, min(900, ref_el + del_))

        log.info("bbox[%d,%d,%d,%d] → PTZ az=%d el=%d", x1, y1, x2, y2, target_az, target_el)
        return target_az, target_el

    async def focus_on_bbox(self, bbox: list[int],
                            ref_az: int = None, ref_el: int = None) -> bool:
        """Move and zoom to focus on a bounding box."""
        target_az, target_el = self.bbox_to_ptz(bbox, ref_az, ref_el)
        return await self.goto(target_az, target_el, self.cfg.ptz_close_zoom, wait=3)

    # ── Snapshot ──

    async def snapshot(self) -> bytes | None:
        """Capture JPEG from camera via ISAPI."""
        try:
            r = await self.client.get(self.base_url + "/ISAPI/Streaming/channels/101/picture")
            if r.status_code == 200:
                return r.content
        except Exception as e:
            log.error("PTZ snapshot failed: %s", e)
        return None

    # ── Scan grid ──

    def generate_scan_positions(self, cols: int = 5, rows: int = 2) -> list[tuple[int, int, int]]:
        """
        Generate grid scan positions around home.
        Returns [(az, el, zoom), ...] covering the greenhouse.
        """
        home_az = self.cfg.ptz_home_az
        home_el = self.cfg.ptz_home_el
        zoom = self.cfg.ptz_wide_zoom
        fov_h = self.cfg.ptz_fov_h

        # spread positions across fov_h * cols range centered on home
        step_az = int(fov_h * 10 * 0.8)  # 80% overlap
        step_el = int(self.cfg.ptz_fov_v * 10 * 0.8)

        start_az = home_az - (cols // 2) * step_az
        start_el = home_el - (rows // 2) * step_el

        positions = []
        for r in range(rows):
            for c in range(cols):
                az = (start_az + c * step_az) % 3600
                el = max(0, min(900, start_el + r * step_el))
                positions.append((az, el, zoom))
        return positions

    async def close(self):
        await self.client.aclose()
