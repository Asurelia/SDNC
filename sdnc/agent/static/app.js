const state = {
  busy: false,
  files: [],
  lastEventId: 0,
  lastResult: null,
  ruleLinks: [],
  rules: [],
};

const modes = ["fast", "think", "max"];
const modeLabels = ["Rapide", "Pensée", "Max"];
const statusLabels = {
  queued: "À traiter",
  running: "En cours",
  done: "Terminé",
  failed: "À revoir",
  held: "Pause",
};

const $ = (id) => document.getElementById(id);

const controls = {
  batchSlider: $("batch-slider"),
  chooseFilesBtn: $("choose-files-btn"),
  compactBtn: $("compact-btn"),
  dropzone: $("dropzone"),
  fileInput: $("file-input"),
  improveBtn: $("improve-btn"),
  learnBtn: $("learn-btn"),
  learnToggle: $("learn-toggle"),
  modeSlider: $("mode-slider"),
  observeBtn: $("observe-btn"),
  processBatchBtn: $("process-batch-btn"),
  processNextBtn: $("process-next-btn"),
  refreshBtn: $("refresh-btn"),
  rulesConsolidateBtn: $("rules-consolidate-btn"),
  sendBtn: $("send-btn"),
  toolsToggle: $("tools-toggle"),
};

const promptForm = $("prompt-form");
const promptInput = $("prompt");
const sensoryForm = $("sensory-form");
const sensorModality = $("sensor-modality");
const sensorLabel = $("sensor-label");
const sensorText = $("sensor-text");

promptForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const text = promptInput.value.trim();
  if (!text || state.busy) return;
  promptInput.value = "";
  addLog("interaction", text);
  await postJson("/api/interact", { text, mode: currentMode() }, (payload) => {
    state.lastResult = payload.result;
    renderResult(payload.result);
    renderStatus(payload.status);
  });
});

sensoryForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const text = sensorText.value.trim();
  const label = sensorLabel.value.trim();
  if ((!text && !label) || state.busy) return;
  sensorText.value = "";
  addLog("signal", text || label);
  await postJson("/api/observe", {
    mode: currentMode(),
    learn: controls.learnToggle.checked,
    use_tools: controls.toolsToggle.checked,
    samples: [{
      modality: sensorModality.value,
      text,
      content: text,
      label: label || undefined,
      source: "web-ui",
    }],
  }, (payload) => {
    state.lastResult = payload.result;
    renderResult(payload.result);
    renderStatus(payload.status);
  });
});

controls.modeSlider.addEventListener("input", renderMode);
controls.batchSlider.addEventListener("input", () => {
  $("batch-label").textContent = controls.batchSlider.value;
});
controls.refreshBtn.addEventListener("click", refreshAll);
controls.improveBtn.addEventListener("click", () => {
  if (state.busy) return;
  postJson("/api/improve", {}, (payload) => {
    addLog("amélioration", payload.report.summary);
    renderStatus(payload.status);
  });
});
controls.learnBtn.addEventListener("click", () => {
  if (state.busy) return;
  postJson("/api/learn-gap", {}, (payload) => {
    addLog("lacune", payload.report.summary);
    renderStatus(payload.status);
  });
});
controls.compactBtn.addEventListener("click", () => {
  if (state.busy) return;
  postJson("/api/compact", {
    batch_size: Number(controls.batchSlider.value || 1),
    preview: false,
  }, (payload) => {
    addLog("compaction", payload.report.summary);
    renderStatus(payload.status);
  });
});
controls.processNextBtn.addEventListener("click", () => processNext(1));
controls.processBatchBtn.addEventListener("click", () => processNext(Number(controls.batchSlider.value || 1)));
controls.rulesConsolidateBtn.addEventListener("click", consolidateRules);
controls.chooseFilesBtn.addEventListener("click", () => controls.fileInput.click());
controls.fileInput.addEventListener("change", () => handleFiles([...controls.fileInput.files]));

for (const name of ["dragenter", "dragover"]) {
  controls.dropzone.addEventListener(name, (event) => {
    event.preventDefault();
    controls.dropzone.classList.add("dragging");
  });
}
for (const name of ["dragleave", "drop"]) {
  controls.dropzone.addEventListener(name, (event) => {
    event.preventDefault();
    controls.dropzone.classList.remove("dragging");
  });
}
controls.dropzone.addEventListener("drop", (event) => {
  handleFiles([...event.dataTransfer.files]);
});
controls.dropzone.addEventListener("keydown", (event) => {
  if (event.key === "Enter" || event.key === " ") controls.fileInput.click();
});

async function handleFiles(files) {
  if (!files.length || state.busy) return;
  setBusy(true);
  $("upload-state").textContent = `${files.length} fichier(s)`;
  try {
    for (const file of files) {
      const dataBase64 = await readFileBase64(file);
      const payload = {
        name: file.name,
        content_type: file.type || "application/octet-stream",
        data_base64: dataBase64,
        origin: "drag-drop",
      };
      const response = await fetch("/api/files/upload", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const json = await response.json();
      if (!response.ok) throw new Error(json.error || response.statusText);
      state.files = json.files || state.files;
      renderFiles(json.summary);
      renderStatus(json.status);
      await refreshRules();
      addLog("fichier", `${file.name} ajouté`);
    }
    $("upload-state").textContent = "stocké";
  } catch (error) {
    addLog("erreur", error.message);
    $("upload-state").textContent = "erreur";
  } finally {
    setBusy(false);
  }
}

async function processNext(count) {
  if (state.busy) return;
  await postJson("/api/files/process-next", {
    count,
    mode: currentMode(),
    learn: controls.learnToggle.checked,
    use_tools: controls.toolsToggle.checked,
  }, (payload) => {
    state.files = payload.files || state.files;
    renderFiles(payload.summary);
    renderStatus(payload.status);
    if (payload.processed.length) {
      const result = payload.processed[payload.processed.length - 1];
      state.lastResult = result;
      renderResult(result);
      addLog("traitement", `${payload.processed.length} fichier(s) traité(s)`);
    } else {
      addLog("traitement", "aucun fichier en attente");
    }
  });
}

async function processFile(fileId) {
  if (state.busy) return;
  await postJson("/api/files/process", {
    file_id: fileId,
    mode: currentMode(),
    learn: controls.learnToggle.checked,
    use_tools: controls.toolsToggle.checked,
  }, (payload) => {
    state.files = payload.files || state.files;
    renderFiles(payload.summary);
    renderStatus(payload.status);
    state.lastResult = payload.result;
    renderResult(payload.result);
    addLog("traitement", "fichier traité");
  });
}

async function moveFile(fileId, status) {
  if (state.busy) return;
  await postJson("/api/files/status", { file_id: fileId, status }, (payload) => {
    state.files = payload.files || state.files;
    renderFiles(payload.summary);
    renderStatus(payload.status);
    addLog("statut", statusLabels[status] || status);
  });
}

async function postJson(path, payload, onSuccess) {
  setBusy(true);
  try {
    const response = await fetch(path, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const json = await response.json();
    if (!response.ok) throw new Error(json.error || response.statusText);
    onSuccess(json);
    await renderRecent();
    await refreshRules();
  } catch (error) {
    addLog("erreur", error.message);
  } finally {
    setBusy(false);
  }
}

async function refreshAll() {
  const [status, files, rules] = await Promise.all([
    fetchJson("/api/status"),
    fetchJson("/api/files"),
    fetchJson("/api/rules"),
  ]);
  renderStatus(status);
  state.files = files.files || [];
  renderFiles(files.summary);
  renderRules(rules);
  await renderRecent();
}

async function refreshRules() {
  const data = await fetchJson("/api/rules");
  renderRules(data);
}

async function renderRecent() {
  const data = await fetchJson("/api/recent");
  renderMemory(data.memories || []);
  renderCompactions(data.memory_compactions || []);
  renderSensory(data.sensory_bindings || []);
  if (!state.lastResult) renderCognitivePanel((data.cognitive_traces || [])[0]?.payload || null);
}

async function fetchJson(path) {
  const response = await fetch(path);
  const json = await response.json();
  if (!response.ok) throw new Error(json.error || response.statusText);
  return json;
}

function renderStatus(status) {
  $("circuit-count").textContent = status.n_circuits;
  $("sparsity").textContent = `${(status.sparsity_ratio * 100).toFixed(2)}%`;
  $("max-active").textContent = status.max_active_circuits;
  $("hot-vram").textContent = status.resource_budget
    ? `${Number(status.resource_budget.hot_vram_gb || 0).toFixed(2)}GB`
    : "--";
  $("hot-experts").textContent = status.expert_summary
    ? `${status.expert_summary.hot}/${status.expert_summary.total}`
    : "--";
  $("workspace-slots").textContent = status.cognitive_core
    ? status.cognitive_core.workspace_slots
    : "--";
  renderSummary(status.file_queue || {});
  $("tool-list").replaceChildren(...(status.tools || []).map((name) => chip(name)));
}

function renderRules(data) {
  state.rules = data.rules || [];
  state.ruleLinks = data.links || [];
  const list = $("rule-list");
  if (!state.rules.length) {
    list.innerHTML = '<span class="empty">aucune règle</span>';
  } else {
    list.replaceChildren(...state.rules.slice(0, 10).map(renderRuleCard));
  }

  const links = $("rule-link-list");
  if (!state.ruleLinks.length) {
    links.innerHTML = '<span class="empty">aucun lien</span>';
    return;
  }
  links.replaceChildren(...state.ruleLinks.slice(0, 8).map((link) => {
    const el = document.createElement("article");
    el.className = "rule-link";
    const label = link.payload?.target_name || link.payload?.prototype_key || link.target_id;
    el.innerHTML = `<strong>${escapeHtml(link.target_kind)} · ${escapeHtml(label)}</strong><p>${escapeHtml(link.relation)} · ${fmt(link.confidence)} · ${link.provenance.length} preuves</p>`;
    return el;
  }));
}

function renderRuleCard(rule) {
  const el = document.createElement("article");
  el.className = `rule-card ${rule.status !== "enabled" ? "muted-card" : ""}`;
  const evidence = `${rule.provenance.length}/${rule.counterexamples.length}`;
  const actions = document.createElement("div");
  actions.className = "rule-actions";
  actions.append(button("+", () => giveRuleFeedback(rule.id, 1), "mini"));
  actions.append(button("-", () => giveRuleFeedback(rule.id, -1), "mini ghost"));
  if (rule.status === "enabled") {
    actions.append(button("Désactiver", () => setRuleStatus(rule.id, "disabled"), "mini ghost"));
  } else {
    actions.append(button("Activer", () => setRuleStatus(rule.id, "enabled"), "mini ghost"));
  }
  actions.append(button("Rejeter", () => setRuleStatus(rule.id, "rejected"), "mini ghost"));

  const body = document.createElement("div");
  body.className = "rule-body";
  body.innerHTML = `
    <div class="rule-head">
      <strong>${escapeHtml(rule.name)}</strong>
      <span>${escapeHtml(rule.status)}</span>
    </div>
    <p>${escapeHtml(rule.trigger_pattern)} → ${escapeHtml(rule.action_tool)}</p>
    <div class="rule-meta">conf ${fmt(rule.confidence)} · preuves ${evidence}</div>
  `;
  el.append(body, actions);
  return el;
}

async function consolidateRules() {
  if (state.busy) return;
  await postJson("/api/rules/consolidate", {}, (payload) => {
    addLog("règles", payload.report.summary);
    renderStatus(payload.status);
  });
}

async function setRuleStatus(ruleId, status) {
  if (state.busy) return;
  await postJson("/api/rules/status", { rule_id: ruleId, status, reason: "web-ui" }, (payload) => {
    renderRules({ rules: payload.rules, links: payload.links });
    renderStatus(payload.status);
    addLog("règles", `${status} · ${payload.rule.name}`);
  });
}

async function giveRuleFeedback(ruleId, score) {
  if (state.busy) return;
  await postJson("/api/rules/feedback", { rule_id: ruleId, score, note: "web-ui" }, (payload) => {
    renderRules({ rules: payload.rules, links: payload.links });
    renderStatus(payload.status);
    addLog("règles", `${score > 0 ? "renforcée" : "affaiblie"} · ${payload.rule.name}`);
  });
}

function renderFiles(summary = null) {
  const grouped = { queued: [], running: [], done: [], failed: [] };
  for (const file of state.files) {
    const key = file.status === "held" ? "failed" : file.status;
    if (grouped[key]) grouped[key].push(file);
  }
  for (const [status, files] of Object.entries(grouped)) {
    const list = $(`${status}-list`);
    if (!files.length) {
      list.innerHTML = '<span class="empty">vide</span>';
      continue;
    }
    list.replaceChildren(...files.map(renderFileCard));
  }
  renderSummary(summary || summarizeFiles(state.files));
}

function renderFileCard(file) {
  const template = $("file-card-template");
  const node = template.content.firstElementChild.cloneNode(true);
  node.querySelector(".file-name").textContent = file.name;
  node.querySelector(".file-modality").textContent = file.modality;
  node.querySelector(".file-preview").textContent = file.preview || file.content_type;
  node.querySelector(".file-meta").textContent = `${formatBytes(file.size_bytes)} · ${statusLabels[file.status] || file.status}`;
  const actions = node.querySelector(".file-actions");
  if (file.status === "queued" || file.status === "failed" || file.status === "held") {
    actions.append(button("Traiter", () => processFile(file.id), "mini"));
  }
  if (file.status === "queued") {
    actions.append(button("Pause", () => moveFile(file.id, "held"), "mini ghost"));
  }
  if (file.status === "held" || file.status === "failed") {
    actions.append(button("Reprendre", () => moveFile(file.id, "queued"), "mini ghost"));
  }
  if (file.status === "running") {
    actions.append(button("Terminer", () => moveFile(file.id, "done"), "mini ghost"));
  }
  if (file.status === "done") {
    actions.append(button("Revoir", () => moveFile(file.id, "queued"), "mini ghost"));
  }
  if (file.error) node.classList.add("has-error");
  return node;
}

function renderSummary(summary) {
  const queued = Number(summary.queued || 0);
  const running = Number(summary.running || 0);
  const done = Number(summary.done || 0);
  const failed = Number(summary.failed || 0);
  const held = Number(summary.held || 0);
  const total = Number(summary.total || queued + running + done + failed + held);
  $("queued-count").textContent = queued + held;
  $("running-count").textContent = running;
  $("done-count").textContent = done;
  $("failed-count").textContent = failed;
  $("queued-pill").textContent = queued + held;
  $("running-pill").textContent = running;
  $("done-pill").textContent = done;
  $("failed-pill").textContent = failed;
  const progress = total ? Math.round((done / total) * 100) : 0;
  $("progress-label").textContent = `${progress}%`;
  $("progress-fill").style.width = `${progress}%`;
}

function renderResult(result) {
  const activation = result.activation || {};
  const metadata = result.metadata || {};
  $("confidence-pill").textContent = `${fmt(activation.confidence)} conf.`;
  $("active-count-pill").textContent = `${activation.active_count || 0} actifs`;
  $("salience-value").textContent = fmt(metadata.salience);
  $("novelty-value").textContent = fmt(activation.novelty);
  renderCircuits(activation);
  renderTraces(result.tool_results || []);
  renderCognitivePanel(metadata.cognitive_core || null);
  const sensory = metadata.sensory_event
    ? `${metadata.sensory_event.modalities.join("+")} bind ${fmt(metadata.sensory_event.binding_score)}`
    : "interaction";
  addLog("résultat", `${sensory} · conf ${fmt(activation.confidence)} · sal ${fmt(metadata.salience)}`);
}

function renderCognitivePanel(workspace) {
  if (!workspace) {
    $("workspace-action").textContent = "--";
    $("uncertainty-value").textContent = "--";
    $("surprise-value").textContent = "--";
    $("workspace-mode").textContent = "--";
    $("workspace-focus").innerHTML = '<span class="empty">aucun état</span>';
    return;
  }
  const prediction = workspace.prediction || {};
  $("workspace-action").textContent = prediction.predicted_action || "--";
  $("uncertainty-value").textContent = fmt(workspace.uncertainty);
  $("surprise-value").textContent = fmt(workspace.surprise);
  $("workspace-slots").textContent = workspace.slot_count || (workspace.slots || []).length || "--";
  $("workspace-mode").textContent = workspace.mode || "--";
  const focus = workspace.attention_focus || [];
  if (!focus.length) {
    $("workspace-focus").innerHTML = '<span class="empty">aucun focus</span>';
    return;
  }
  $("workspace-focus").replaceChildren(...focus.slice(0, 6).map((item, index) => {
    const el = document.createElement("article");
    el.className = "trace";
    el.innerHTML = `<strong>focus ${index + 1}</strong><p>${escapeHtml(item)}</p>`;
    return el;
  }));
}

function renderCircuits(activation) {
  const indices = activation.indices || [];
  const total = Math.max(72, Math.min(180, indices.length * 18 || 72));
  const active = new Map();
  indices.forEach((id) => active.set(id % total, id));
  const cells = [];
  for (let i = 0; i < total; i += 1) {
    const cell = document.createElement("span");
    cell.className = active.has(i) ? "cell active" : "cell";
    cell.title = active.has(i) ? `circuit ${active.get(i)}` : "";
    cells.push(cell);
  }
  $("circuit-grid").replaceChildren(...cells);
}

function renderTraces(tools) {
  if (!tools.length) {
    $("tool-traces").innerHTML = '<span class="empty">aucune trace</span>';
    return;
  }
  $("tool-traces").replaceChildren(...tools.map((tool) => {
    const el = document.createElement("article");
    el.className = "trace";
    const status = tool.success ? "ok" : "erreur";
    el.innerHTML = `<strong>${escapeHtml(tool.tool_name)} · ${status}</strong><p>${escapeHtml(tool.content).slice(0, 360)}</p>`;
    return el;
  }));
}

function renderMemory(memories) {
  if (!memories.length) {
    $("memory-list").innerHTML = '<span class="empty">mémoire vide</span>';
    return;
  }
  $("memory-list").replaceChildren(...memories.map((memory) => {
    const el = document.createElement("article");
    el.className = "memory";
    const score = memory.similarity ? `sim ${fmt(memory.similarity)}` : `sal ${fmt(memory.salience)}`;
    el.innerHTML = `<strong>${score}</strong><p>${escapeHtml(memory.text).slice(0, 160)}</p>`;
    return el;
  }));
}

function renderCompactions(compactions) {
  if (!compactions.length) {
    $("compaction-list").innerHTML = '<span class="empty">aucune compaction</span>';
    return;
  }
  $("compaction-list").replaceChildren(...compactions.map((record) => {
    const el = document.createElement("article");
    el.className = "memory";
    const protectedCount = (record.protected_episode_ids || []).length;
    el.innerHTML = `<strong>${escapeHtml(record.key)} · ${record.episode_count} ep · ${protectedCount} protégés</strong><p>${escapeHtml(record.summary).slice(0, 180)}</p>`;
    return el;
  }));
}

function renderSensory(bindings) {
  if (!bindings.length) {
    $("sensory-list").innerHTML = '<span class="empty">aucune observation</span>';
    return;
  }
  $("sensory-list").replaceChildren(...bindings.map((binding) => {
    const el = document.createElement("article");
    el.className = "memory";
    el.innerHTML = `<strong>${escapeHtml(binding.modalities.join("+"))} · ${fmt(binding.binding_score)}</strong><p>${escapeHtml(binding.summary).slice(0, 160)}</p>`;
    return el;
  }));
}

function addLog(kind, message) {
  const row = document.createElement("article");
  row.className = `log-row ${kind === "erreur" ? "bad" : ""}`;
  const time = new Date().toLocaleTimeString("fr-FR", { hour: "2-digit", minute: "2-digit", second: "2-digit" });
  row.innerHTML = `<strong>${escapeHtml(kind)}</strong><span>${escapeHtml(time)}</span><p>${escapeHtml(message).slice(0, 220)}</p>`;
  const log = $("event-log");
  log.prepend(row);
  while (log.children.length > 80) log.lastElementChild.remove();
}

function handleEvent(event) {
  const payload = JSON.parse(event.data);
  state.lastEventId = Number(payload.id || state.lastEventId);
  if (payload.event_type === "file") {
    const body = payload.payload || {};
    addLog("fichier", `${body.action || "event"} ${body.name || ""}`);
    refreshFilesOnly().catch(() => {});
  } else if (payload.event_type === "interaction") {
    addLog("interaction", payload.payload.text || "cycle");
  } else if (payload.event_type === "observation") {
    addLog("observation", (payload.payload.modalities || [payload.payload.modality || "signal"]).join("+"));
  } else if (payload.event_type === "improvement") {
    addLog("amélioration", payload.payload.summary || "cycle terminé");
  } else if (payload.event_type === "learning") {
    addLog("lacune", payload.payload.summary || "apprentissage terminé");
  } else if (payload.event_type === "compaction") {
    addLog("compaction", payload.payload.summary || "cycle terminé");
  } else if (payload.event_type === "rules") {
    addLog("règles", payload.payload.summary || payload.payload.action || "mise à jour");
    refreshRules().catch(() => {});
  }
  renderRecent().catch(() => {});
}

async function refreshFilesOnly() {
  const data = await fetchJson("/api/files");
  state.files = data.files || [];
  renderFiles(data.summary);
}

function connectEventStream() {
  if (!window.EventSource) return;
  const source = new EventSource(`/api/stream?after=${state.lastEventId}`);
  source.addEventListener("file", handleEvent);
  source.addEventListener("interaction", handleEvent);
  source.addEventListener("observation", handleEvent);
  source.addEventListener("feedback", handleEvent);
  source.addEventListener("improvement", handleEvent);
  source.addEventListener("learning", handleEvent);
  source.addEventListener("compaction", handleEvent);
  source.addEventListener("rules", handleEvent);
  source.onerror = () => {
    source.close();
    window.setTimeout(connectEventStream, 1800);
  };
}

function setBusy(value) {
  state.busy = value;
  for (const element of [
    controls.improveBtn,
    controls.learnBtn,
    controls.compactBtn,
    controls.observeBtn,
    controls.processBatchBtn,
    controls.processNextBtn,
    controls.refreshBtn,
    controls.rulesConsolidateBtn,
    controls.sendBtn,
  ]) {
    element.disabled = value;
  }
}

function currentMode() {
  return modes[Number(controls.modeSlider.value || 0)] || "fast";
}

function renderMode() {
  $("mode-label").textContent = modeLabels[Number(controls.modeSlider.value || 0)] || "Rapide";
}

function chip(text) {
  const el = document.createElement("span");
  el.className = "chip";
  el.textContent = text;
  return el;
}

function button(label, onClick, className = "") {
  const el = document.createElement("button");
  el.type = "button";
  el.className = className;
  el.textContent = label;
  el.addEventListener("click", onClick);
  return el;
}

function summarizeFiles(files) {
  const summary = { queued: 0, running: 0, done: 0, failed: 0, held: 0, total: files.length };
  for (const file of files) summary[file.status] = (summary[file.status] || 0) + 1;
  return summary;
}

function readFileBase64(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result).split(",", 2)[1] || "");
    reader.onerror = () => reject(reader.error || new Error("lecture impossible"));
    reader.readAsDataURL(file);
  });
}

function formatBytes(value) {
  const bytes = Number(value || 0);
  if (bytes < 1024) return `${bytes} o`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} Ko`;
  return `${(bytes / 1024 / 1024).toFixed(1)} Mo`;
}

function fmt(value) {
  return Number(value || 0).toFixed(3);
}

function escapeHtml(value) {
  return String(value).replace(/[&<>"']/g, (char) => ({
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    '"': "&quot;",
    "'": "&#039;",
  }[char]));
}

renderMode();
$("batch-label").textContent = controls.batchSlider.value;
addLog("système", "interface prête");
refreshAll().catch((error) => addLog("erreur", error.message));
connectEventStream();
