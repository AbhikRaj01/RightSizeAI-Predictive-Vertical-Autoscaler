const state = {
  timer: null,
  lastPrediction: null,
  metrics: [],
};

const els = {
  backendState: document.getElementById("backendState"),
  deploymentInput: document.getElementById("deploymentInput"),
  allocatedCpuInput: document.getElementById("allocatedCpuInput"),
  cpuCostInput: document.getElementById("cpuCostInput"),
  simulateInput: document.getElementById("simulateInput"),
  refreshButton: document.getElementById("refreshButton"),
  spikeButton: document.getElementById("spikeButton"),
  enableButton: document.getElementById("enableButton"),
  applyButton: document.getElementById("applyButton"),
  disableButton: document.getElementById("disableButton"),
  actionMessage: document.getElementById("actionMessage"),
  currentCpu: document.getElementById("currentCpu"),
  recommendedCpu: document.getElementById("recommendedCpu"),
  confidenceValue: document.getElementById("confidenceValue"),
  confidenceBar: document.getElementById("confidenceBar"),
  monthlySavings: document.getElementById("monthlySavings"),
  cpuSaved: document.getElementById("cpuSaved"),
  statusValue: document.getElementById("statusValue"),
  trendValue: document.getElementById("trendValue"),
  explanationText: document.getElementById("explanationText"),
  modelSource: document.getElementById("modelSource"),
  cacheState: document.getElementById("cacheState"),
  forecastTime: document.getElementById("forecastTime"),
  historyBody: document.getElementById("historyBody"),
  decisionBody: document.getElementById("decisionBody"),
  chart: document.getElementById("cpuChart"),
};

function params() {
  const deployment = els.deploymentInput.value.trim() || "demo-app";
  const allocated = Number.parseInt(els.allocatedCpuInput.value, 10) || 1000;
  const cost = Number.parseFloat(els.cpuCostInput.value) || 3.5;
  const simulate = els.simulateInput.checked;
  return { deployment, allocated, cost, simulate };
}

function apiUrl(path, query) {
  const url = new URL(path, window.location.origin);
  Object.entries(query).forEach(([key, value]) => {
    if (value !== undefined && value !== null && value !== "") {
      url.searchParams.set(key, value);
    }
  });
  return url;
}

async function getJson(path, query = {}) {
  const response = await fetch(apiUrl(path, query));
  if (!response.ok) {
    throw new Error(`${response.status} ${response.statusText}`);
  }
  return response.json();
}

async function postJson(path, query = {}) {
  const response = await fetch(apiUrl(path, query), { method: "POST" });
  if (!response.ok) {
    let detail = `${response.status} ${response.statusText}`;
    try {
      const body = await response.json();
      detail = body.detail || detail;
    } catch {
      detail = `${response.status} ${response.statusText}`;
    }
    throw new Error(detail);
  }
  return response.json();
}

function formatCpu(value) {
  if (value === undefined || value === null || Number.isNaN(Number(value))) {
    return "--";
  }
  return `${Math.round(Number(value))}m`;
}

function formatMoney(value) {
  if (value === undefined || value === null || Number.isNaN(Number(value))) {
    return "--";
  }
  return `INR ${Number(value).toLocaleString("en-IN", {
    maximumFractionDigits: 2,
  })}`;
}

function formatPercent(value) {
  if (value === undefined || value === null || Number.isNaN(Number(value))) {
    return "--";
  }
  return `${Math.round(Number(value) * 100)}%`;
}

function formatTime(value) {
  if (!value) {
    return "--";
  }
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return "--";
  }
  return date.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}

function setBackendState(text, mode) {
  els.backendState.textContent = text;
  els.backendState.className = `run-state ${mode || ""}`.trim();
}

function setActionMessage(text, mode) {
  els.actionMessage.textContent = text;
  els.actionMessage.className = mode || "";
}

function statusFromPrediction(prediction) {
  if (!prediction) {
    return "Idle";
  }
  if (prediction.confidence >= 0.8) {
    return "Safe";
  }
  if (prediction.confidence >= 0.5) {
    return "Review";
  }
  return "Hold";
}

function updateSummary(prediction, cost) {
  state.lastPrediction = prediction;

  els.currentCpu.textContent = formatCpu(prediction.current_cpu_millicores);
  els.recommendedCpu.textContent = prediction.recommended_cpu_limit || formatCpu(prediction.recommended_cpu_millicores);
  els.confidenceValue.textContent = formatPercent(prediction.confidence);
  els.confidenceBar.style.width = `${Math.max(0, Math.min(100, prediction.confidence * 100))}%`;
  els.monthlySavings.textContent = formatMoney(cost.estimated_monthly_savings);
  els.cpuSaved.textContent = `${formatCpu(cost.saved_cpu_millicores)} saved`;
  els.statusValue.textContent = statusFromPrediction(prediction);
  els.trendValue.textContent = `${prediction.trend || "stable"} trend`;
  els.explanationText.textContent = prediction.explanation || "No explanation returned.";
  els.modelSource.textContent = prediction.model?.source || "--";
  els.cacheState.textContent = prediction.model?.retrained ? "Retrained now" : "Cached model reused";
  els.forecastTime.textContent = formatTime(prediction.prediction?.timestamp);
}

function updateHistory(items) {
  if (!items || items.length === 0) {
    els.historyBody.innerHTML = '<tr><td colspan="5">No scaling actions yet.</td></tr>';
    return;
  }

  const rows = items
    .slice()
    .reverse()
    .map((item) => `
      <tr>
        <td>${formatTime(item.timestamp)}</td>
        <td>${item.namespace || "default"}/${item.deployment || "--"}</td>
        <td>${item.old_cpu || "--"}</td>
        <td>${item.new_cpu || "--"}</td>
        <td>${item.reason || "Predictive adjustment"}</td>
      </tr>
    `);

  els.historyBody.innerHTML = rows.join("");
}

function updateDecisions(items) {
  if (!items || items.length === 0) {
    els.decisionBody.innerHTML = '<tr><td colspan="5">No decisions recorded yet.</td></tr>';
    return;
  }

  const rows = items
    .slice()
    .reverse()
    .map((item) => `
      <tr>
        <td>${formatTime(item.timestamp)}</td>
        <td>${item.action || "--"}</td>
        <td>${item.namespace || "default"}/${item.deployment || "--"}</td>
        <td>${item.current_cpu || "--"} -> ${item.desired_cpu || item.recommended_cpu || "--"}</td>
        <td>${item.reason || "No reason recorded"}</td>
      </tr>
    `);

  els.decisionBody.innerHTML = rows.join("");
}

function resizeCanvas(canvas) {
  const rect = canvas.getBoundingClientRect();
  const scale = window.devicePixelRatio || 1;
  canvas.width = Math.max(320, Math.floor(rect.width * scale));
  canvas.height = Math.max(240, Math.floor(rect.height * scale));
  return { width: canvas.width, height: canvas.height, scale };
}

function drawLine(ctx, points, color, dashed = false) {
  if (points.length < 2) {
    return;
  }
  ctx.save();
  ctx.strokeStyle = color;
  ctx.lineWidth = 3;
  ctx.setLineDash(dashed ? [10, 8] : []);
  ctx.beginPath();
  points.forEach((point, index) => {
    if (index === 0) {
      ctx.moveTo(point.x, point.y);
    } else {
      ctx.lineTo(point.x, point.y);
    }
  });
  ctx.stroke();
  ctx.restore();
}

function drawChart() {
  const canvas = els.chart;
  const ctx = canvas.getContext("2d");
  const { width, height } = resizeCanvas(canvas);
  const padding = { top: 28, right: 28, bottom: 44, left: 58 };
  const plotWidth = width - padding.left - padding.right;
  const plotHeight = height - padding.top - padding.bottom;

  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = "#111111";
  ctx.fillRect(0, 0, width, height);

  const actual = state.metrics
    .map((point) => ({
      timestamp: new Date(point.timestamp).getTime(),
      value: Number(point.cpu_millicores),
    }))
    .filter((point) => Number.isFinite(point.timestamp) && Number.isFinite(point.value))
    .slice(-96);

  const predictionValue = Number(state.lastPrediction?.prediction?.yhat_upper || state.lastPrediction?.recommended_cpu_millicores);
  const values = actual.map((point) => point.value);
  if (Number.isFinite(predictionValue)) {
    values.push(predictionValue);
  }

  if (actual.length < 2 || values.length === 0) {
    ctx.fillStyle = "#a9a9a9";
    ctx.font = "16px system-ui, sans-serif";
    ctx.fillText("Waiting for metrics.", padding.left, padding.top + 28);
    return;
  }

  const minValue = Math.max(0, Math.min(...values) * 0.84);
  const maxValue = Math.max(...values) * 1.16 || 100;
  const firstTime = actual[0].timestamp;
  const lastTime = actual[actual.length - 1].timestamp;
  const forecastTime = lastTime + 60 * 60 * 1000;

  const xFor = (time) => padding.left + ((time - firstTime) / (forecastTime - firstTime)) * plotWidth;
  const yFor = (value) => padding.top + (1 - (value - minValue) / (maxValue - minValue)) * plotHeight;

  ctx.strokeStyle = "#2c2c2c";
  ctx.lineWidth = 1;
  ctx.fillStyle = "#a9a9a9";
  ctx.font = "12px system-ui, sans-serif";

  for (let i = 0; i <= 4; i += 1) {
    const y = padding.top + (plotHeight / 4) * i;
    const value = maxValue - ((maxValue - minValue) / 4) * i;
    ctx.beginPath();
    ctx.moveTo(padding.left, y);
    ctx.lineTo(width - padding.right, y);
    ctx.stroke();
    ctx.fillText(`${Math.round(value)}m`, 10, y + 4);
  }

  const actualPoints = actual.map((point) => ({
    x: xFor(point.timestamp),
    y: yFor(point.value),
  }));
  drawLine(ctx, actualPoints, "#20d3b0");

  if (Number.isFinite(predictionValue)) {
    const lastActual = actual[actual.length - 1];
    drawLine(
      ctx,
      [
        { x: xFor(lastActual.timestamp), y: yFor(lastActual.value) },
        { x: xFor(forecastTime), y: yFor(predictionValue) },
      ],
      "#ff6b6b",
      true
    );

    ctx.fillStyle = "#ff6b6b";
    ctx.beginPath();
    ctx.arc(xFor(forecastTime), yFor(predictionValue), 6, 0, Math.PI * 2);
    ctx.fill();
  }

  ctx.fillStyle = "#a9a9a9";
  ctx.fillText(formatTime(actual[0].timestamp), padding.left, height - 16);
  ctx.fillText("Next hour", width - padding.right - 64, height - 16);
}

async function refreshDashboard() {
  const { deployment, allocated, cost, simulate } = params();
  setBackendState("Refreshing", "");

  try {
    const [prediction, metrics, estimate, history, decisions] = await Promise.all([
      getJson("/predict", { target_deployment: deployment, simulate }),
      getJson("/metrics", { target_deployment: deployment, simulate, limit: 288 }),
      getJson("/cost-estimate", {
        target_deployment: deployment,
        allocated_cpu_millicores: allocated,
        cpu_hour_cost: cost,
        simulate,
      }),
      getJson("/scaling-history", { deployment, limit: 20 }),
      getJson("/operator-decisions", { deployment, limit: 20 }),
    ]);

    state.metrics = metrics.points || [];
    updateSummary(prediction, estimate);
    updateHistory(history.items || []);
    updateDecisions(decisions.items || []);
    drawChart();
    setBackendState("Live", "ok");
  } catch (error) {
    setBackendState("Backend offline", "error");
    els.explanationText.textContent = `Start FastAPI, then refresh. ${error.message}`;
    drawChart();
  }
}

async function runAction(label, action) {
  const originalText = label.textContent;
  label.disabled = true;
  setActionMessage("Working...", "");
  try {
    const result = await action();
    setActionMessage(result.message || "Action completed.", "ok");
    await refreshDashboard();
  } catch (error) {
    setActionMessage(error.message, "error");
  } finally {
    label.disabled = false;
    label.textContent = originalText;
  }
}

function restartTimer() {
  if (state.timer) {
    window.clearInterval(state.timer);
  }
  state.timer = window.setInterval(refreshDashboard, 5000);
}

els.refreshButton.addEventListener("click", refreshDashboard);
els.spikeButton.addEventListener("click", () => {
  const { deployment } = params();
  runAction(els.spikeButton, () => postJson("/simulate/spike", { target_deployment: deployment }));
});
els.enableButton.addEventListener("click", () => {
  const { deployment, simulate } = params();
  runAction(els.enableButton, () =>
    postJson("/autoscaling/enable", {
      target_deployment: deployment,
      mode: "auto",
      min_cpu: "100m",
      max_cpu: "2000m",
      cooldown_seconds: 60,
      change_threshold_percent: 5,
      dry_run: simulate,
    })
  );
});
els.applyButton.addEventListener("click", () => {
  const { deployment, allocated, simulate } = params();
  runAction(els.applyButton, () =>
    postJson("/apply-recommendation", {
      target_deployment: deployment,
      allocated_cpu_millicores: allocated,
      simulate,
      dry_run: simulate,
    })
  );
});
els.disableButton.addEventListener("click", () => {
  const { deployment, simulate } = params();
  runAction(els.disableButton, () =>
    postJson("/autoscaling/disable", {
      target_deployment: deployment,
      dry_run: simulate,
    })
  );
});
els.deploymentInput.addEventListener("change", refreshDashboard);
els.allocatedCpuInput.addEventListener("change", refreshDashboard);
els.cpuCostInput.addEventListener("change", refreshDashboard);
els.simulateInput.addEventListener("change", refreshDashboard);
window.addEventListener("resize", drawChart);

refreshDashboard();
restartTimer();
