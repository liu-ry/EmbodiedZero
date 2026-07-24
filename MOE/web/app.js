const PALETTE = ["#66e3c4", "#74a7ff", "#ffb86b", "#f487ba", "#b59bff", "#66c9ff", "#f7e989", "#f39090"];

let trace = makeDemoTrace();
let activeLayer = 0;
let selectedToken = 0;
let timer = null;
let activeStep = 0;

const STEPS = [
  ["01", "Token 表示"], ["02", "Router"], ["03", "Top-k"], ["04", "Expert 合并"],
];

const $ = (selector) => document.querySelector(selector);
const fmt = (value) => Number(value).toFixed(2);
const colorFor = (expert) => PALETTE[expert % PALETTE.length];

function rng(seed) {
  let state = seed >>> 0;
  return () => ((state = (state * 1664525 + 1013904223) >>> 0) / 4294967296);
}

function makeDemoTrace(config) {
  const random = rng(config && config.seed !== undefined ? config.seed : Math.floor(Math.random() * 4294967295));
  const numExperts = config && config.experts || 4;
  const numTokens = config && config.tokens || 12;
  const topK = Math.min(config && config.topK || 2, numExperts);
  const numLayers = config && config.layers || 3;
  const layers = [];
  for (let layer = 0; layer < numLayers; layer++) {
    const tokenExperts = [], tokenWeights = [], routerProbabilities = [];
    const expertLoad = Array(numExperts).fill(0);
    for (let token = 0; token < numTokens; token++) {
      const raw = Array.from({ length: numExperts }, (_, expert) => 0.15 + random() + (expert === (token + layer) % numExperts ? 0.7 : 0));
      const total = raw.reduce((a, b) => a + b, 0);
      const probs = raw.map((x) => x / total);
      const selected = [...probs.keys()].sort((a, b) => probs[b] - probs[a]).slice(0, topK);
      const selectedTotal = selected.reduce((sum, id) => sum + probs[id], 0);
      tokenExperts.push(selected);
      tokenWeights.push(selected.map((id) => probs[id] / selectedTotal));
      routerProbabilities.push(probs);
      selected.forEach((id) => expertLoad[id]++);
    }
    layers.push({
      layer, capacity: Math.ceil(numTokens * topK / numExperts * 1.25), dropped_assignments: 0, expert_load: expertLoad,
      expert_importance: Array.from({ length: numExperts }, (_, id) => routerProbabilities.reduce((sum, probs) => sum + probs[id], 0) / numTokens),
      token_experts: tokenExperts, token_weights: tokenWeights, router_probabilities: routerProbabilities,
    });
  }
  return { schema_version: 1, source: "Browser random simulation", num_experts: numExperts, num_tokens: numTokens, top_k: topK, layers };
}

function validateTrace(data) {
  if (!data || !Array.isArray(data.layers) || !data.layers.length || !data.num_experts || !data.num_tokens || !data.top_k) {
    throw new Error("JSON 不是 EmbodiedZero/MOE 路由轨迹格式。");
  }
  for (const layer of data.layers) {
    if (!Array.isArray(layer.token_experts) || !Array.isArray(layer.token_weights)) {
      throw new Error("路由轨迹缺少 token_experts 或 token_weights。");
    }
  }
}

function render() {
  const layer = trace.layers[activeLayer];
  selectedToken = Math.min(selectedToken, trace.num_tokens - 1);
  syncConfigControls();
  $("#trace-source").textContent = trace.source || "Imported trace";
  $("#flow-title").textContent = `Layer ${activeLayer} · MoE FFN 内部：Router → Experts → Combine`;
  renderWalkthrough(layer); renderTabs(); renderLegend(); renderFlow(layer); renderTokenDetails(layer); renderHealth(layer);
}

function syncConfigControls() {
  $("#config-layers").value = trace.layers.length;
  $("#config-tokens").value = trace.num_tokens;
  $("#config-experts").value = trace.num_experts;
  $("#config-topk").value = trace.top_k;
}

function integerInput(id, fallback, min, max) {
  const value = Number.parseInt($(id).value, 10);
  return Math.max(min, Math.min(max, Number.isFinite(value) ? value : fallback));
}

function regenerateFromControls() {
  const config = {
    layers: integerInput("#config-layers", 3, 1, 12),
    tokens: integerInput("#config-tokens", 12, 1, 32),
    experts: integerInput("#config-experts", 4, 1, 16),
    topK: integerInput("#config-topk", 2, 1, 16),
  };
  config.topK = Math.min(config.topK, config.experts);
  trace = makeDemoTrace(config);
  activeLayer = 0;
  selectedToken = 0;
  stopAutoplay();
  render();
}

function renderWalkthrough(layer) {
  const probs = layer.router_probabilities && layer.router_probabilities[selectedToken] || [];
  const selected = layer.token_experts[selectedToken];
  const gates = layer.token_weights[selectedToken];
  $("#step-tabs").innerHTML = STEPS.map((step, id) => `<button class="step-tab ${id === activeStep ? "active" : ""}" data-step="${id}"><span>${step[0]}</span><strong>${step[1]}</strong></button>`).join("");
  document.querySelectorAll(".step-tab").forEach((button) => button.addEventListener("click", () => { activeStep = Number(button.dataset.step); renderWalkthrough(layer); }));

  let title, copy, visual;
  if (activeStep === 0) {
    title = `Token ${selectedToken} 的隐藏表示 x`;
    copy = "每层都会把 token 的当前隐藏向量送入 Router。这个向量不是词表 ID，而是 Transformer 前一层计算后的连续特征。";
    visual = `<div class="vector">${Array.from({length: 18}, (_, i) => `<i style="opacity:${.25 + ((i * 17 + selectedToken * 11) % 70) / 100}"></i>`).join("")}</div><span class="arrow">→</span><div class="router-box">Wᵣ · x</div>`;
  } else if (activeStep === 1) {
    title = "Router 对所有 Expert 计算 softmax 概率";
    copy = "Router 是一个很小的线性层。它先产生每个 Expert 的 logit，再经过 softmax 得到总和为 1 的概率分布。这里显示的是实际记录的概率。";
    visual = probabilityRows(probs, selected, false);
  } else if (activeStep === 2) {
    title = `只保留概率最高的 Top-${trace.top_k} 个 Expert`;
    copy = "未入选的 Expert 不执行 FFN，因此节省计算。入选概率会重新归一化为 gate 权重；这正是每条路由连线粗细不同的原因。";
    visual = probabilityRows(probs, selected, true);
  } else {
    title = "Expert 输出按 gate 权重加权相加";
    copy = `Token ${selectedToken} 的最终 MoE 输出为 ${selected.map((expert, slot) => `${fmt(gates[slot])} × Expert ${expert}(x)`).join(" + ")}。随后它会通过残差连接进入下一层。`;
    visual = `<div class="route-list">${selected.map((expert, slot) => `<span class="route-chip" style="color:${colorFor(expert)};border-color:${colorFor(expert)}">E${expert} × ${fmt(gates[slot])}</span>`).join("")}</div><span class="arrow">→</span><div class="result-box">MoE output y</div>`;
  }
  $("#walkthrough-stage").innerHTML = `<div class="stage-copy"><h3>${title}</h3><p>${copy}</p></div><div class="stage-visual">${visual}</div>`;
}

function probabilityRows(probs, selected, showTopk) {
  return `<div class="mini-probs">${probs.map((probability, expert) => {
    const rank = selected.indexOf(expert);
    const retained = rank !== -1;
    return `<div class="mini-prob ${showTopk && !retained ? "muted-expert" : ""}"><span>E${expert}</span><div class="bar-track"><div class="bar-fill" style="width:${probability * 100}%;background:${colorFor(expert)}"></div></div><span>${showTopk && retained ? `<b class="topk-tag">${fmt(trace.layers[activeLayer].token_weights[selectedToken][rank])}</b>` : fmt(probability)}</span></div>`;
  }).join("")}</div>`;
}

function renderTabs() {
  $("#layer-tabs").innerHTML = trace.layers.map((_, id) => `<button class="layer-tab ${id === activeLayer ? "active" : ""}" data-layer="${id}">Layer ${id}</button>`).join("");
  document.querySelectorAll(".layer-tab").forEach((button) => button.addEventListener("click", () => { activeLayer = Number(button.dataset.layer); render(); }));
}

function renderLegend() {
  $("#flow-legend").innerHTML = Array.from({ length: trace.num_experts }, (_, id) => `<span class="legend-item"><i class="legend-dot" style="background:${colorFor(id)}"></i>Expert ${id}</span>`).join("");
}

function renderFlow(layer) {
  const tokenNodes = Array.from({ length: trace.num_tokens }, (_, id) => `<button class="token-node ${id === selectedToken ? "selected" : ""}" data-token="${id}">Token ${id}</button>`).join("");
  const experts = Array.from({ length: trace.num_experts }, (_, id) => `<div class="expert-node" data-expert="${id}" style="border-color:${colorFor(id)}"><span>Expert ${id}</span><small>${layer.expert_load[id]} routes</small></div>`).join("");
  $("#flow-view").innerHTML = `
    <svg class="flow-svg" id="flow-svg" aria-hidden="true"></svg>
    <div class="flow-columns">
      <div><p class="column-label">TOKEN STREAM</p><div class="token-list">${tokenNodes}</div></div>
      <div class="middle-copy"><strong>Router</strong><br>每个 Token 计算全部 Expert 的 softmax 概率，仅保留 Top-${trace.top_k}，再按 gate 权重送往对应 Expert。</div>
      <div class="expert-side"><p class="column-label">EXPERT POOL</p><div class="expert-list">${experts}</div></div>
    </div>`;
  document.querySelectorAll(".token-node").forEach((node) => node.addEventListener("click", () => { selectedToken = Number(node.dataset.token); render(); }));
  requestAnimationFrame(() => drawEdges(layer));
}

function drawEdges(layer) {
  const host = $("#flow-view"), svg = $("#flow-svg"), hostBox = host.getBoundingClientRect();
  const edges = [];
  for (let token = 0; token < trace.num_tokens; token++) {
    const from = host.querySelector(`.token-node[data-token="${token}"]`).getBoundingClientRect();
    layer.token_experts[token].forEach((expert, slot) => {
      const to = host.querySelector(`.expert-node[data-expert="${expert}"]`).getBoundingClientRect();
      const weight = layer.token_weights[token][slot];
      const active = token === selectedToken;
      const x1 = from.right - hostBox.left, y1 = from.top + from.height / 2 - hostBox.top;
      const x2 = to.left - hostBox.left, y2 = to.top + to.height / 2 - hostBox.top;
      const bend = Math.max(70, (x2 - x1) * .42);
      edges.push(`<path d="M ${x1} ${y1} C ${x1 + bend} ${y1}, ${x2 - bend} ${y2}, ${x2} ${y2}" fill="none" stroke="${colorFor(expert)}" stroke-width="${1 + weight * 5}" stroke-opacity="${active ? 1 : .10 + weight * .18}" ${active ? "" : ""}/>`);
    });
  }
  svg.innerHTML = edges.join("");
}

function renderTokenDetails(layer) {
  $("#token-title").textContent = `Token ${selectedToken}`;
  const routes = layer.token_experts[selectedToken].map((expert, slot) => `<span class="route-chip" style="color:${colorFor(expert)};border-color:${colorFor(expert)}">Expert ${expert} · gate ${fmt(layer.token_weights[selectedToken][slot])}</span>`).join("");
  $("#token-routes").innerHTML = routes;
  const probs = (layer.router_probabilities && layer.router_probabilities[selectedToken]) || Array.from({ length: trace.num_experts }, (_, id) => layer.token_experts[selectedToken].includes(id) ? layer.token_weights[selectedToken][layer.token_experts[selectedToken].indexOf(id)] : 0);
  $("#probability-bars").innerHTML = probs.map((probability, expert) => `<div class="prob-row"><span>E${expert}</span><div class="bar-track"><div class="bar-fill" style="width:${probability * 100}%;background:${colorFor(expert)}"></div></div><span>${fmt(probability)}</span></div>`).join("");
}

function renderHealth(layer) {
  const totalRoutes = layer.expert_load.reduce((a, b) => a + b, 0);
  const capacity = layer.capacity === null || layer.capacity === undefined ? "∞" : layer.capacity;
  $("#layer-health").innerHTML = `<div class="health-stat"><span>Capacity / expert</span><strong>${capacity}</strong></div><div class="health-stat"><span>Dropped assignments</span><strong>${layer.dropped_assignments}</strong></div><div class="health-stat"><span>Total assignments</span><strong>${totalRoutes}</strong></div><div class="health-stat"><span>Mean importance</span><strong>${fmt(layer.expert_importance.reduce((a,b)=>a+b,0) / trace.num_experts)}</strong></div>`;
  const maximum = Math.max(...layer.expert_load, 1);
  $("#load-bars").innerHTML = layer.expert_load.map((load, expert) => `<div class="load-row"><span>E${expert}</span><div class="bar-track"><div class="bar-fill" style="width:${load / maximum * 100}%;background:${colorFor(expert)}"></div></div><span>${load}</span></div>`).join("");
}

$("#trace-upload").addEventListener("change", async (event) => {
  const file = event.target.files[0];
  if (!file) return;
  try {
    const data = JSON.parse(await file.text());
    validateTrace(data); trace = data; activeLayer = 0; selectedToken = 0; stopAutoplay(); render();
  } catch (error) { window.alert(`无法读取路由 JSON：${error.message}`); }
  event.target.value = "";
});
$("#reset-button").addEventListener("click", regenerateFromControls);
document.querySelectorAll(".config-card input").forEach((input) => input.addEventListener("change", regenerateFromControls));
document.querySelectorAll(".config-card input").forEach((input) => input.addEventListener("keydown", (event) => { if (event.key === "Enter") regenerateFromControls(); }));
$("#previous-layer").addEventListener("click", () => { activeLayer = (activeLayer - 1 + trace.layers.length) % trace.layers.length; render(); });
$("#next-layer").addEventListener("click", () => { activeLayer = (activeLayer + 1) % trace.layers.length; render(); });
function stopAutoplay() { if (timer) clearInterval(timer); timer = null; $("#autoplay").textContent = "播放"; }
$("#autoplay").addEventListener("click", () => {
  if (timer) return stopAutoplay();
  $("#autoplay").textContent = "暂停";
  timer = setInterval(() => { activeLayer = (activeLayer + 1) % trace.layers.length; render(); }, 1200);
});
window.addEventListener("resize", () => render());
render();
