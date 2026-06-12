const canvas = document.getElementById("board");
const ctx = canvas.getContext("2d");
const statusEl = document.getElementById("status");
const stepStatusEl = document.getElementById("stepStatus");
const playerStatusEl = document.getElementById("playerStatus");
const detailsEl = document.getElementById("details");
const lineupEl = document.getElementById("lineup");
const logBotEl = document.getElementById("logBot");
const botLogEl = document.getElementById("botLog");
const seedInput = document.getElementById("seed");
const matchModeEl = document.getElementById("matchMode");
const stepButton = document.getElementById("step");
const autoButton = document.getElementById("auto");

const colors = {
  "-1": "#888888",
  0: "#0072B2",
  1: "#D55E00",
  2: "#009E73",
  3: "#F0E442",
};

const groupColors = {
  1: "rgba(60, 170, 255, 0.95)",
  2: "rgba(255, 210, 70, 0.95)",
  3: "rgba(120, 235, 150, 0.95)",
  4: "rgba(255, 105, 150, 0.95)",
};

const CENTER = 50;
const ROTATION_RADIUS_LIMIT = 50;

let state = null;
let selected = null;
let autoTimer = null;
let stepInFlight = false;

function api(path, body) {
  return fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body || {}),
  }).then((res) => {
    if (!res.ok) return res.json().then((data) => Promise.reject(data));
    return res.json();
  });
}

function worldToCanvas(x, y) {
  const scale = canvas.width / 100;
  return [x * scale, y * scale];
}

function canvasToWorld(evt) {
  const rect = canvas.getBoundingClientRect();
  const scaleX = canvas.width / rect.width;
  const scaleY = canvas.height / rect.height;
  return [
    ((evt.clientX - rect.left) * scaleX) / (canvas.width / 100),
    ((evt.clientY - rect.top) * scaleY) / (canvas.height / 100),
  ];
}

function planetAt(x, y) {
  if (!state) return null;
  let best = null;
  for (const p of state.observation.planets) {
    const dx = p[2] - x;
    const dy = p[3] - y;
    const hitRadius = Math.max(p[4], 2.2);
    if (Math.hypot(dx, dy) <= hitRadius) best = p;
  }
  return best;
}

function draw() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = "#000000";
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  const [sunX, sunY] = worldToCanvas(50, 50);
  const sunR = 10 * (canvas.width / 100);
  const glow = ctx.createRadialGradient(sunX, sunY, sunR * 0.5, sunX, sunY, sunR * 2.5);
  glow.addColorStop(0, "rgba(255, 200, 50, 0.6)");
  glow.addColorStop(0.5, "rgba(255, 150, 20, 0.2)");
  glow.addColorStop(1, "rgba(255, 100, 0, 0)");
  ctx.fillStyle = glow;
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  drawCircle(50, 50, 10, "#FFB800", true);
  drawCircle(50, 50, 10, "#FFD700", false);

  if (!state) return;

  const obs = state.observation;
  const cometIds = new Set(obs.comet_planet_ids || []);

  drawOpeningGroups();

  if (obs.comets) {
    for (const group of obs.comets) {
      const idx = group.path_index;
      group.planet_ids.forEach((pid, i) => {
        void pid;
        const path = group.paths[i];
        const tailLen = Math.min(idx + 1, path.length, 3);
        if (tailLen < 2) return;
        for (let t = 1; t < tailLen; t++) {
          const pi = idx - t;
          if (pi < 0) break;
          const alpha = 0.5 * (1 - t / tailLen);
          const [x0, y0] = worldToCanvas(path[pi][0], path[pi][1]);
          const [x1, y1] = worldToCanvas(path[pi + 1][0], path[pi + 1][1]);
          const width = (3 - (2 * t) / tailLen) * (canvas.width / 100);
          ctx.beginPath();
          ctx.moveTo(x1, y1);
          ctx.lineTo(x0, y0);
          ctx.strokeStyle = `rgba(200, 220, 255, ${alpha})`;
          ctx.lineWidth = width;
          ctx.lineCap = "round";
          ctx.stroke();
        }
      });
    }
  }

  for (const p of obs.planets || []) {
    const id = p[0];
    const owner = p[1];
    const x = p[2];
    const y = p[3];
    const r = p[4];
    const ships = p[5];
    const color = colors[owner] || "#ffffff";
    const isSelected = selected && selected[0] === id;

    drawCircle(x, y, r, color, true);
    if (cometIds.has(id)) {
      drawCircle(x, y, r, "#FFFFFF", false);
    }
    if (isSelected) {
      drawCircle(x, y, r + 0.55, "#FFFFFF", false);
    }
    drawText(String(Math.floor(ships)), x, y, "#FFFFFF", 8);
  }

  for (const f of obs.fleets || []) {
    drawFleet(f);
  }

  drawSelectedTrajectory();

  ctx.fillStyle = "#FFFFFF";
  ctx.font = "14px sans-serif";
  ctx.textAlign = "left";
  ctx.textBaseline = "top";
  ctx.fillText(`Step: ${obs.step || 0}`, 10, 10);
}

function drawOpeningGroups() {
  if (!state || state.matchMode !== "dev2p" || !state.quadrantRegions) return;
  if ((state.observation.step || 0) <= 0) return;

  const planetById = new Map((state.observation.planets || []).map((p) => [p[0], p]));
  for (const group of state.quadrantRegions.priorityGroups || []) {
    const color = groupColors[group.rank] || "rgba(255, 255, 255, 0.65)";
    for (const planetId of group.planet_ids || []) {
      const planet = planetById.get(planetId);
      if (!planet) continue;
      drawRegionRing(planet, color, group.rank === 1);
    }
  }
  for (const group of state.quadrantRegions.rotatingGroups || []) {
    const color = groupColors[group.rank] || "rgba(255, 255, 255, 0.65)";
    for (const planetId of group.planet_ids || []) {
      const planet = planetById.get(planetId);
      if (!planet) continue;
      drawRegionRing(planet, color, group.rank === 1);
    }
  }
}

function drawRegionRing(planet, color, isMyRegion) {
  const scale = canvas.width / 100;
  const [cx, cy] = worldToCanvas(planet[2], planet[3]);
  const radius = (planet[4] + (isMyRegion ? 1.25 : 0.8)) * scale;

  ctx.save();
  ctx.beginPath();
  ctx.arc(cx, cy, radius, 0, Math.PI * 2);
  ctx.strokeStyle = color;
  ctx.lineWidth = isMyRegion ? 5 : 3;
  ctx.globalAlpha = isMyRegion ? 0.9 : 0.55;
  ctx.stroke();
  ctx.restore();
}

function drawSelectedTrajectory() {
  if (!state || !selected) return;
  const obs = state.observation;
  const planet = obs.planets.find((p) => p[0] === selected[0]);
  if (!planet) return;

  if ((obs.comet_planet_ids || []).includes(planet[0])) {
    drawCometTrajectory(planet);
    return;
  }

  const initial = (obs.initial_planets || []).find((p) => p[0] === planet[0]);
  if (!initial) return;

  const radius = Math.hypot(initial[2] - CENTER, initial[3] - CENTER);
  if (radius + planet[4] >= ROTATION_RADIUS_LIMIT) {
    drawStaticSelection(planet);
    return;
  }

  const omega = obs.angular_velocity || 0;
  const step = obs.step || 0;
  const initialAngle = Math.atan2(initial[3] - CENTER, initial[2] - CENTER);
  const points = [];
  const future = [];
  const samples = 180;
  for (let i = 0; i <= samples; i++) {
    const a = initialAngle + (i / samples) * Math.PI * 2;
    points.push([CENTER + radius * Math.cos(a), CENTER + radius * Math.sin(a)]);
  }
  for (let t = 0; t <= 80; t += 8) {
    const a = initialAngle + omega * (step + t);
    future.push([CENTER + radius * Math.cos(a), CENTER + radius * Math.sin(a), t]);
  }

  drawPath(points, "rgba(255, 255, 255, 0.22)", 1.1, [7, 8]);
  drawPath(
    future.map((p) => [p[0], p[1]]),
    "rgba(0, 255, 180, 0.85)",
    2.2,
    []
  );

  for (const [x, y, t] of future) {
    const [cx, cy] = worldToCanvas(x, y);
    ctx.beginPath();
    ctx.arc(cx, cy, t === 0 ? 5 : 3.5, 0, Math.PI * 2);
    ctx.fillStyle = t === 0 ? "rgba(255, 255, 255, 0.95)" : "rgba(0, 255, 180, 0.85)";
    ctx.fill();
  }
}

function drawStaticSelection(planet) {
  const [x, y] = worldToCanvas(planet[2], planet[3]);
  ctx.beginPath();
  ctx.arc(x, y, planet[4] * (canvas.width / 100) + 10, 0, Math.PI * 2);
  ctx.strokeStyle = "rgba(255, 255, 255, 0.35)";
  ctx.lineWidth = 2;
  ctx.setLineDash([6, 7]);
  ctx.stroke();
  ctx.setLineDash([]);
}

function drawCometTrajectory(planet) {
  const obs = state.observation;
  for (const group of obs.comets || []) {
    const index = group.planet_ids.indexOf(planet[0]);
    if (index < 0) continue;
    const path = group.paths[index] || [];
    const current = Math.max(0, group.path_index || 0);
    const full = path.map((p) => [p[0], p[1]]);
    const future = path.slice(current, current + 45).map((p) => [p[0], p[1]]);
    drawPath(full, "rgba(255, 255, 255, 0.18)", 1.1, [5, 8]);
    drawPath(future, "rgba(120, 190, 255, 0.9)", 2.5, []);
    for (let i = current; i < Math.min(path.length, current + 45); i += 6) {
      const [x, y] = worldToCanvas(path[i][0], path[i][1]);
      ctx.beginPath();
      ctx.arc(x, y, i === current ? 5 : 3.5, 0, Math.PI * 2);
      ctx.fillStyle = i === current ? "rgba(255, 255, 255, 0.95)" : "rgba(120, 190, 255, 0.9)";
      ctx.fill();
    }
    return;
  }
}

function drawPath(points, color, width, dash) {
  if (!points || points.length < 2) return;
  ctx.save();
  ctx.beginPath();
  points.forEach(([x, y], index) => {
    const [cx, cy] = worldToCanvas(x, y);
    if (index === 0) {
      ctx.moveTo(cx, cy);
    } else {
      ctx.lineTo(cx, cy);
    }
  });
  ctx.strokeStyle = color;
  ctx.lineWidth = width;
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  ctx.setLineDash(dash);
  ctx.stroke();
  ctx.restore();
}

function drawCircle(x, y, r, color, fill = false) {
  const scale = canvas.width / 100;
  const [cx, cy] = worldToCanvas(x, y);
  ctx.beginPath();
  ctx.arc(cx, cy, r * scale, 0, 2 * Math.PI);
  ctx.fillStyle = color;
  ctx.strokeStyle = color;
  ctx.lineWidth = 2;
  if (fill) {
    ctx.fill();
    ctx.stroke();
  } else {
    ctx.stroke();
  }
}

function drawText(text, x, y, color, sizePt = 12) {
  const scale = canvas.width / 100;
  const [cx, cy] = worldToCanvas(x, y);
  ctx.font = `${(sizePt * scale) / 5.8}px sans-serif`;
  ctx.fillStyle = color;
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillText(text, cx, cy);
}

function drawFleet(f) {
  const scale = canvas.width / 100;
  const owner = f[1];
  const [x, y] = worldToCanvas(f[2], f[3]);
  const angle = f[4];
  const ships = Math.max(1, f[6]);
  const color = colors[owner] || "#ffffff";
  const sz = (0.5 + (2.5 * Math.log(ships)) / Math.log(1000)) * scale;

  ctx.save();
  ctx.translate(x, y);
  ctx.rotate(angle);

  ctx.beginPath();
  ctx.moveTo(sz, 0);
  ctx.lineTo(-sz, -sz * 0.7);
  ctx.lineTo(-sz * 0.3, 0);
  ctx.lineTo(-sz, sz * 0.7);
  ctx.closePath();
  ctx.fillStyle = color;
  ctx.fill();

  ctx.strokeStyle = "rgba(255, 255, 255, 0.55)";
  ctx.lineWidth = sz * 0.15;
  ctx.lineCap = "round";
  if (owner === 1 || owner === 3) {
    ctx.beginPath();
    ctx.moveTo(sz * 0.8, 0);
    ctx.lineTo(-sz * 0.2, 0);
    ctx.stroke();
  }
  if (owner === 2 || owner === 3) {
    ctx.beginPath();
    ctx.moveTo(sz * 0.6, -sz * 0.15);
    ctx.lineTo(-sz * 0.7, -sz * 0.5);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(sz * 0.6, sz * 0.15);
    ctx.lineTo(-sz * 0.7, sz * 0.5);
    ctx.stroke();
  }

  ctx.restore();

  const labelOffset = f[3] >= 50 ? -3 : 3;
  drawText(String(ships), f[2], f[3] + labelOffset, color, 6);
}

function renderDetails() {
  if (!state) return;
  const obs = state.observation;
  if (state.matchMode && matchModeEl.value !== state.matchMode) {
    matchModeEl.value = state.matchMode;
  }
  const seedText = state.seed === null || state.seed === undefined ? "unknown" : state.seed;
  stepStatusEl.textContent = `Seed ${seedText} | Step ${obs.step || 0} / 500${selectionLabel(obs)}`;
  playerStatusEl.innerHTML = state.players
    .map(
      (p) =>
        `<div class="statusRow"><span class="swatch" style="background:${colors[p.player]}"></span><span>P${p.player}: ${p.name}</span><strong>${p.score}</strong><span>${p.status}</span></div>`
    )
    .join("");

  lineupEl.innerHTML = state.players
    .map(
      (p) =>
        `<li><span class="swatch" style="background:${colors[p.player]}"></span><span>P${p.player}: ${p.name}</span></li>`
    )
    .join("");

  renderBotSelector();
  renderBotLog();

  detailsEl.innerHTML = obs.planets
    .slice()
    .sort((a, b) => a[0] - b[0])
    .map((p) => {
      const owner = p[1] === -1 ? "N" : `P${p[1]}`;
      const cluster = clusterLabelForPlanet(p[0]);
      const planetType = planetTypeLabel(p[0]);
      return `<div class="planetRow"><strong>#${p[0]}</strong><span class="muted">${owner}</span><span>${Math.floor(p[5])} ships</span><span>+${p[6]}</span><span class="clusterTag">${cluster}</span><span class="typeTag">${planetType}</span></div>`;
    })
    .join("");

}

function clusterLabelForPlanet(planetId) {
  if (!state || state.matchMode !== "dev2p" || !state.quadrantRegions) return "-";
  const priorityGroup = (state.quadrantRegions.priorityGroups || []).find((group) =>
    (group.planet_ids || []).includes(planetId)
  );
  if (priorityGroup) return `G${priorityGroup.rank}`;
  const rotatingGroup = (state.quadrantRegions.rotatingGroups || []).find((group) =>
    (group.planet_ids || []).includes(planetId)
  );
  if (rotatingGroup) return `R${rotatingGroup.rank}`;
  return "-";
}

function planetTypeLabel(planetId) {
  if (!state || state.matchMode !== "dev2p" || !state.quadrantRegions) return "-";
  for (const group of state.quadrantRegions.priorityGroups || []) {
    if ((group.planet_ids || []).includes(planetId)) return "S";
  }
  for (const group of state.quadrantRegions.rotatingGroups || []) {
    if ((group.planet_ids || []).includes(planetId)) return "R";
  }
  return "-";
}

function renderBotSelector() {
  const current = logBotEl.value;
  logBotEl.innerHTML = state.players
    .map((p) => `<option value="${p.player}">P${p.player}: ${p.name}</option>`)
    .join("");
  if ([...logBotEl.options].some((option) => option.value === current)) {
    logBotEl.value = current;
  }
}

function renderBotLog() {
  const selectedPlayer = Number(logBotEl.value || 0);
  const entries = (state.actionHistory || []).filter((entry) => entry.player === selectedPlayer);
  if (!entries.length) {
    botLogEl.innerHTML = `<div class="logEmpty">No logged turns yet.</div>`;
    return;
  }
  botLogEl.innerHTML = entries
    .slice(-80)
    .reverse()
    .map((entry) => {
      if (entry.source === null || entry.source === undefined) {
        return `<div class="logRow"><span class="muted">T${entry.step}</span><span>No launch</span></div>`;
      }
      const target = entry.target === null ? "open space" : `#${entry.target}`;
      return `<div class="logRow"><span class="muted">T${entry.step}</span><span>#${entry.source} -> ${target}</span><strong>${entry.ships}</strong><span>${entry.angleDeg} deg</span></div>`;
    })
    .join("");
}

function selectionLabel(obs) {
  if (!selected) return "";
  const planet = obs.planets.find((p) => p[0] === selected[0]);
  if (!planet) return "";
  if ((obs.comet_planet_ids || []).includes(planet[0])) {
    return ` | Selected #${planet[0]} comet`;
  }
  const initial = (obs.initial_planets || []).find((p) => p[0] === planet[0]);
  if (!initial) return ` | Selected #${planet[0]}`;
  const r = Math.hypot(initial[2] - CENTER, initial[3] - CENTER);
  const kind = r + planet[4] < ROTATION_RADIUS_LIMIT ? "orbiting" : "static";
  return ` | Selected #${planet[0]} ${kind}`;
}

function render() {
  renderDetails();
  draw();
}

function setAutoMode(enabled) {
  if (enabled && !autoTimer) {
    autoButton.classList.add("active");
    autoButton.textContent = "Auto On";
    autoTimer = window.setInterval(() => {
      stepMatch();
    }, 450);
    stepMatch();
    return;
  }
  if (!enabled && autoTimer) {
    window.clearInterval(autoTimer);
    autoTimer = null;
    autoButton.classList.remove("active");
    autoButton.textContent = "Auto";
  }
}

async function stepMatch() {
  if (stepInFlight || (state && state.done)) {
    if (state && state.done) setAutoMode(false);
    return;
  }
  stepInFlight = true;
  stepButton.disabled = true;
  autoButton.disabled = false;
  try {
    state = await api("/api/step", {});
    selected = null;
    render();
    if (state.done) setAutoMode(false);
  } catch (err) {
    setAutoMode(false);
    stepStatusEl.textContent = err && err.detail ? err.detail : "Step failed";
  } finally {
    stepInFlight = false;
    stepButton.disabled = false;
  }
}

canvas.addEventListener("click", (evt) => {
  const [x, y] = canvasToWorld(evt);
  const p = planetAt(x, y);
  if (!p) {
    selected = null;
    render();
    return;
  }
  selected = p;
  render();
});

document.getElementById("newGame").addEventListener("click", async () => {
  setAutoMode(false);
  selected = null;
  state = await api("/api/new", { seed: seedInput.value, matchMode: matchModeEl.value });
  render();
});

stepButton.addEventListener("click", async () => {
  await stepMatch();
});

autoButton.addEventListener("click", () => {
  setAutoMode(!autoTimer);
});

logBotEl.addEventListener("change", renderBotLog);

api("/api/new", { matchMode: matchModeEl.value }).then((data) => {
  state = data;
  render();
});
