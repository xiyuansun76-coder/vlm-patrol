// VLM-Patrol Frontend

const API = '';  // same origin

// ── Navigation ──

document.querySelectorAll('.nav-item').forEach(item => {
  item.addEventListener('click', e => {
    e.preventDefault();
    document.querySelectorAll('.nav-item').forEach(i => i.classList.remove('active'));
    document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
    item.classList.add('active');
    document.getElementById('page-' + item.dataset.page).classList.add('active');
  });
});

// ── WebSocket ──

let ws = null;
function connectWS() {
  const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
  ws = new WebSocket(`${proto}//${location.host}/ws`);
  ws.onopen = () => {
    document.getElementById('ws-status').className = 'status-dot online';
    document.querySelector('.nav-footer .status-text').textContent = 'Connected';
  };
  ws.onclose = () => {
    document.getElementById('ws-status').className = 'status-dot offline';
    document.querySelector('.nav-footer .status-text').textContent = 'Disconnected';
    setTimeout(connectWS, 3000);
  };
  ws.onmessage = (e) => {
    try {
      const msg = JSON.parse(e.data);
      handleWSMessage(msg);
    } catch {}
  };
}

function handleWSMessage(msg) {
  // handle real-time updates
  if (msg.type === 'patrol_update') refreshDashboard();
  if (msg.type === 'analysis_update') refreshDashboard();
}

// ── API helpers ──

async function api(path, opts = {}) {
  const resp = await fetch(API + path, opts);
  return resp.json();
}

async function apiPost(path, body = null) {
  const opts = { method: 'POST' };
  if (body) {
    opts.headers = { 'Content-Type': 'application/json' };
    opts.body = JSON.stringify(body);
  }
  return api(path, opts);
}

// ── Dashboard ──

async function refreshDashboard() {
  try {
    const [patrolStatus, agentStatus, config] = await Promise.all([
      api('/api/patrol/status'),
      api('/api/agent/status'),
      api('/api/config'),
    ]);

    // Health score
    const score = agentStatus.latest ? agentStatus.latest.health_score : '--';
    document.getElementById('health-score').textContent = score;

    // Plants count
    const plants = patrolStatus.session ? patrolStatus.session.plants.length : '--';
    document.getElementById('plants-count').textContent = plants;

    // Dataset
    document.getElementById('dataset-size').textContent = config.yolo.dataset_size;
    document.getElementById('yolo-model').textContent = config.yolo.model_path;

    // Latest analysis
    if (agentStatus.latest) {
      const a = agentStatus.latest;
      let html = `Score: ${a.health_score}/100\n${a.summary}`;
      if (a.commands.length > 0) {
        html += '\n\nCommands:';
        a.commands.forEach(c => {
          html += `\n  ${c.action}: ${c.reason} (${c.duration_sec}s) [${c.state}]`;
        });
      }
      document.getElementById('latest-analysis').textContent = html;
    }

    // Patrol result
    if (patrolStatus.session) {
      const s = patrolStatus.session;
      let html = `Session: ${s.session_id} | Status: ${s.status}\n`;
      html += `Plants found: ${s.plants.length} | Images collected: ${s.images_collected}\n`;
      s.plants.forEach(p => {
        html += `\n  ${p.type} — ${p.health} (${p.details || 'no details'})`;
      });
      document.getElementById('patrol-result').textContent = html;
    }

    // Settings page — populate form fields
    document.getElementById('cfg-llm-url').value = config.llm.url;
    document.getElementById('cfg-llm-model').value = config.llm.model;
    document.getElementById('cfg-camera-url').value = config.camera.snapshot_url;
    document.getElementById('cfg-stream-url').value = config.camera.stream_url;
    document.getElementById('cfg-sensor-url').value = config.sensor_url || '';
    document.getElementById('cfg-actuator-url').value = config.actuator_url || '';
    currentClasses = [...config.classes];
    renderClasses();
    document.getElementById('cfg-yolo-path').value = config.yolo.model_path;
    document.getElementById('cfg-yolo-datadir').value = config.yolo.data_dir || './data';
    document.getElementById('cfg-yolo-autotrain').checked = config.yolo.auto_train;
    document.getElementById('cfg-yolo-threshold').value = config.yolo.train_threshold;
    document.getElementById('cfg-ptz-enabled').checked = config.ptz_enabled || false;
    document.getElementById('cfg-ptz-url').value = config.ptz_url || '';
    document.getElementById('cfg-ptz-imgw').value = config.ptz_img_w || 1920;
    document.getElementById('cfg-ptz-imgh').value = config.ptz_img_h || 1080;
    document.getElementById('cfg-ptz-fovh').value = config.ptz_fov_h || 55;
    document.getElementById('cfg-ptz-fovv').value = config.ptz_fov_v || 32;
    document.getElementById('cfg-patrol-enabled').checked = config.patrol.enabled;
    document.getElementById('cfg-patrol-interval').value = config.patrol.interval;
    document.getElementById('cfg-patrol-strategy').value = config.patrol.strategy || 'single';
    document.getElementById('cfg-agent-enabled').checked = config.agent.auto_analysis;
    document.getElementById('cfg-agent-interval').value = config.agent.interval;
    document.getElementById('cfg-server-host').value = config.server_host || '0.0.0.0';
    document.getElementById('cfg-server-port').value = config.server_port || 8765;

  } catch (e) {
    console.error('Dashboard refresh failed:', e);
  }
}

// ── Patrol ──

async function runPatrol() {
  addChatMessage('system', 'Starting patrol...');
  const result = await apiPost('/api/patrol/start');
  addChatMessage('system', `Patrol: ${result.status}`);
  setTimeout(refreshDashboard, 5000);
}

// ── Agent ──

async function runAnalysis() {
  addChatMessage('system', 'Running analysis...');
  try {
    const result = await apiPost('/api/agent/analyze');
    addChatMessage('agent', `Health: ${result.health_score}/100\n${result.summary}`, result.commands);
    refreshDashboard();
  } catch (e) {
    addChatMessage('system', 'Analysis failed: ' + e.message);
  }
}

async function toggleAutoAnalysis() {
  const enabled = document.getElementById('auto-analysis-toggle').checked;
  if (enabled) {
    await apiPost('/api/agent/start');
    document.getElementById('agent-status-text').textContent = 'Running';
  } else {
    await apiPost('/api/agent/stop');
    document.getElementById('agent-status-text').textContent = 'Stopped';
  }
}

async function toggleAutoCare() {
  const enabled = document.getElementById('auto-care-toggle').checked;
  await apiPost('/api/agent/auto-care?enable=' + enabled);
}

async function sendChat() {
  const input = document.getElementById('chat-input');
  const text = input.value.trim();
  if (!text) return;
  input.value = '';

  addChatMessage('user', text);
  try {
    const result = await apiPost('/api/chat', { message: text });
    addChatMessage(result.role || 'agent', result.text, result.commands || []);
    // If setup assistant changed config, refresh dashboard to reflect changes
    if (result.is_setup) refreshDashboard();
  } catch (e) {
    addChatMessage('system', 'Error: ' + e.message);
  }
}

function addChatMessage(role, text, commands = []) {
  const box = document.getElementById('chat-messages');
  const div = document.createElement('div');
  div.className = 'chat-msg ' + role;
  let html = text.replace(/</g, '&lt;').replace(/\n/g, '<br>');
  if (commands && commands.length > 0) {
    commands.forEach(c => {
      html += `<div class="cmd-card">${c.action}: ${c.reason} (${c.duration_sec}s) [${c.state}]</div>`;
    });
  }
  div.innerHTML = html;
  box.appendChild(div);
  box.scrollTop = box.scrollHeight;
}

// ── Camera ──

let lastImageBytes = null;

async function captureSnapshot() {
  try {
    const resp = await fetch(API + '/api/patrol/status');
    const config = await api('/api/config');
    if (!config.camera.snapshot_url) {
      document.getElementById('camera-placeholder').textContent = 'No camera configured';
      return;
    }
    // Proxy through the server — fetch snapshot via camera URL
    // For now, just show the stream URL or snapshot URL directly
    const img = document.getElementById('camera-img');
    img.src = config.camera.snapshot_url + '?t=' + Date.now();
    img.style.display = 'block';
    document.getElementById('camera-placeholder').style.display = 'none';
  } catch (e) {
    console.error('Snapshot failed:', e);
  }
}

async function detectFromCamera() {
  document.getElementById('detection-results').textContent = 'Detecting...';
  try {
    // Upload current camera image for VLM detection
    const config = await api('/api/config');
    if (!config.camera.snapshot_url) {
      document.getElementById('detection-results').textContent = 'No camera configured';
      return;
    }
    const imgResp = await fetch(config.camera.snapshot_url);
    const blob = await imgResp.blob();

    const form = new FormData();
    form.append('file', blob, 'snapshot.jpg');
    const result = await fetch(API + '/api/vlm/detect', { method: 'POST', body: form });
    const data = await result.json();

    let text = `Found ${data.plants.length} plants:\n`;
    data.plants.forEach((p, i) => {
      text += `\n${i+1}. ${p.type} — ${p.health}\n   bbox: [${p.bbox.join(', ')}]\n   ${p.description}`;
    });
    document.getElementById('detection-results').textContent = text;
  } catch (e) {
    document.getElementById('detection-results').textContent = 'Error: ' + e.message;
  }
}

async function diagnoseFromCamera() {
  document.getElementById('detection-results').textContent = 'Diagnosing...';
  try {
    const config = await api('/api/config');
    if (!config.camera.snapshot_url) {
      document.getElementById('detection-results').textContent = 'No camera configured';
      return;
    }
    const imgResp = await fetch(config.camera.snapshot_url);
    const blob = await imgResp.blob();

    const form = new FormData();
    form.append('file', blob, 'snapshot.jpg');
    const result = await fetch(API + '/api/vlm/diagnose', { method: 'POST', body: form });
    const data = await result.json();

    let text = `Species: ${data.type}\nHealth: ${data.health}\nConfidence: ${(data.confidence * 100).toFixed(0)}%\n\n${data.details}`;
    document.getElementById('detection-results').textContent = text;
  } catch (e) {
    document.getElementById('detection-results').textContent = 'Error: ' + e.message;
  }
}

// ── Classes editor ──

let currentClasses = [];

function renderClasses() {
  document.getElementById('cfg-classes').innerHTML = currentClasses.map((c, i) =>
    `<span class="tag">${c} <span class="tag-remove" onclick="removeClass(${i})">×</span></span>`
  ).join('');
}

function addClass() {
  const input = document.getElementById('cfg-class-input');
  const name = input.value.trim().toLowerCase();
  if (name && !currentClasses.includes(name)) {
    currentClasses.push(name);
    renderClasses();
    input.value = '';
  }
}

function removeClass(idx) {
  currentClasses.splice(idx, 1);
  renderClasses();
}

// ── Save config ──

async function saveConfig() {
  const config = {
    llm: {
      url: document.getElementById('cfg-llm-url').value,
      model: document.getElementById('cfg-llm-model').value,
      api_key: document.getElementById('cfg-llm-key').value,
    },
    camera: {
      snapshot_url: document.getElementById('cfg-camera-url').value,
      stream_url: document.getElementById('cfg-stream-url').value,
    },
    sensor: {
      url: document.getElementById('cfg-sensor-url').value,
    },
    actuator: {
      url: document.getElementById('cfg-actuator-url').value,
    },
    classes: currentClasses,
    yolo: {
      model_path: document.getElementById('cfg-yolo-path').value,
      data_dir: document.getElementById('cfg-yolo-datadir').value,
      auto_train: document.getElementById('cfg-yolo-autotrain').checked,
      train_threshold: parseInt(document.getElementById('cfg-yolo-threshold').value) || 50,
    },
    ptz: {
      enabled: document.getElementById('cfg-ptz-enabled').checked,
      url: document.getElementById('cfg-ptz-url').value,
      image_width: parseInt(document.getElementById('cfg-ptz-imgw').value) || 1920,
      image_height: parseInt(document.getElementById('cfg-ptz-imgh').value) || 1080,
      fov_h_deg: parseInt(document.getElementById('cfg-ptz-fovh').value) || 55,
      fov_v_deg: parseInt(document.getElementById('cfg-ptz-fovv').value) || 32,
    },
    patrol: {
      enabled: document.getElementById('cfg-patrol-enabled').checked,
      interval_minutes: parseInt(document.getElementById('cfg-patrol-interval').value) || 60,
      strategy: document.getElementById('cfg-patrol-strategy').value,
    },
    agent: {
      auto_analysis: document.getElementById('cfg-agent-enabled').checked,
      interval_minutes: parseInt(document.getElementById('cfg-agent-interval').value) || 30,
    },
    server: {
      host: document.getElementById('cfg-server-host').value,
      port: parseInt(document.getElementById('cfg-server-port').value) || 8765,
    },
  };

  try {
    const result = await apiPost('/api/config', config);
    const status = document.getElementById('save-status');
    if (result.status === 'ok') {
      status.textContent = 'Saved and applied! (server restart needed for host/port changes)';
      status.style.color = 'var(--success)';
    } else {
      status.textContent = 'Error: ' + (result.error || 'unknown');
      status.style.color = 'var(--danger)';
    }
    setTimeout(() => { status.textContent = ''; }, 5000);
  } catch (e) {
    document.getElementById('save-status').textContent = 'Save failed: ' + e.message;
  }
}

// ── Init ──

connectWS();
refreshDashboard();
setInterval(refreshDashboard, 30000);
