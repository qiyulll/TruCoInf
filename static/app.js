const state = {
  models: [],
  selected: new Set(["qwen-new-7B", "Qwen1.5-7B-Chat"]),
  configPath: "",
  requestId: "",
  evidenceModels: [],
};

const $ = (id) => document.getElementById(id);

function esc(value) {
  return String(value ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function log(message, data) {
  const stamp = new Date().toLocaleTimeString();
  const text = data === undefined ? message : `${message}\n${typeof data === "string" ? data : JSON.stringify(data, null, 2)}`;
  $("logOutput").textContent += `[${stamp}] ${text}\n\n`;
  $("logOutput").scrollTop = $("logOutput").scrollHeight;
}

async function api(path, options = {}) {
  const res = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  const text = await res.text();
  let data;
  try {
    data = text ? JSON.parse(text) : {};
  } catch {
    data = { raw: text };
  }
  if (!res.ok) {
    const detail = data.detail || data.raw || res.statusText;
    const err = new Error(typeof detail === "string" ? detail : (detail.message || JSON.stringify(detail)));
    err.detail = detail;
    throw err;
  }
  return data;
}

function setBadge(el, text, cls = "") {
  el.className = `badge ${cls}`.trim();
  el.textContent = text;
}

function setStep(id, text, cls = "idle") {
  const el = $(id);
  if (!el) return;
  el.className = `step-state ${cls}`;
  el.textContent = text;
}

function setButtonLoading(button, loading, label) {
  if (!button) return;
  if (loading) {
    button.dataset.label = button.textContent;
    button.textContent = label || "运行中...";
    button.classList.add("loading");
    button.disabled = true;
  } else {
    button.textContent = button.dataset.label || button.textContent;
    button.classList.remove("loading");
    button.disabled = false;
  }
}

function mb(bytes) {
  return (Number(bytes || 0) / 1024 / 1024).toFixed(2);
}

async function copyText(text) {
  if (!text) return;
  await navigator.clipboard.writeText(text);
  log("已复制", text);
}

function addCopyHandler(button, value) {
  button.addEventListener("click", () => copyText(typeof value === "function" ? value() : value));
}

function renderPath(value) {
  return `
    <div class="path-row">
      <code class="path">${esc(value)}</code>
      <button class="copy-inline" data-copy="${esc(value)}">复制</button>
    </div>
  `;
}

function bindCopyButtons(root = document) {
  root.querySelectorAll("[data-copy]").forEach((btn) => {
    btn.addEventListener("click", () => copyText(btn.dataset.copy));
  });
}

function showCommand(targetId, label, command) {
  if (!command) return;
  const target = $(targetId);
  target.classList.remove("hidden");
  target.innerHTML = `
    <div class="command-title">${esc(label)}</div>
    <pre>${esc(command)}</pre>
    <button data-copy="${esc(command)}">复制命令</button>
  `;
  bindCopyButtons(target);
  recordCommand(label, command);
}

function recordCommand(label, command) {
  if (!command) return;
  const row = document.createElement("div");
  row.className = "command-item";
  row.innerHTML = `
    <div class="command-title">${esc(label)}</div>
    <pre>${esc(command)}</pre>
    <button data-copy="${esc(command)}">复制命令</button>
  `;
  $("commandHistory").prepend(row);
  bindCopyButtons(row);
}

function diagnoseError(errorOrText) {
  const text = String(errorOrText?.message || errorOrText || "");
  const lower = text.toLowerCase();
  const hints = [];
  if (lower.includes("out of memory") || lower.includes("cuda oom")) {
    hints.push("CUDA OOM：降低 max_new_tokens、减少同时加载模型、先停止融合服务释放显存，或把 proof 放到独立 GPU。");
  }
  if (lower.includes("chat_template") || lower.includes("chat template")) {
    hints.push("tokenizer.chat_template 缺失：关闭 apply_chat_template，或给对应 tokenizer 配置 chat_template。");
  }
  if (lower.includes("zk_weights") || lower.includes("weights 不存在")) {
    hints.push("zk_weights 缺失：请确认 zk_weights/<model-name> 已存在；不要把它放进 zk_evidence，也不要在清理 evidence 时删除。");
  }
  if (lower.includes("transformer_prove") && (lower.includes("not found") || lower.includes("no such file") || lower.includes("找不到"))) {
    hints.push("transformer_prove 找不到：检查 zkhook/transformer_prove 是否已编译，以及 verify.py 中 CUDA 可执行文件路径。");
  }
  if (lower.includes("tables_7b.bin") || lower.includes("lookup table 文件不存在") || lower.includes("tables 文件不存在")) {
    hints.push("tables_7B.bin 缺失：检查 zkhook/tables_7B.bin 是否存在；7B 模型 verify 默认读取这个表。");
  }
  if (lower.includes("rayactorerror") || lower.includes("illegal memory access") || lower.includes("zksoftmax")) {
    hints.push("Ray 残留或融合服务占用 GPU：先停止融合服务并执行 ray stop --force，再重新生成 proof。之前测试中 proof 与常驻模型服务并发会触发 zkSoftmax 非法访存。");
  }
  if (lower.includes("connection refused") || lower.includes("failed to establish a new connection")) {
    hints.push("融合 API 未启动：Web UI 在 7860，生成服务在 8001；请先启动服务并等待 API ready。");
  }
  if (!hints.length) hints.push("未匹配到内置提示：请查看实际命令、stdout/stderr 和服务日志定位。");
  return hints;
}

function showErrorHints(errorOrText) {
  const text = String(errorOrText?.message || errorOrText || "");
  const hints = diagnoseError(errorOrText);
  $("errorHints").innerHTML = `
    <div class="error-title">${esc(text || "运行失败")}</div>
    ${hints.map((hint) => `<div class="error-hint">${esc(hint)}</div>`).join("")}
  `;
}

function renderPreflight(data) {
  const cls = data.safe_to_run ? "ok" : (data.ok ? "warn" : "err");
  return `
    <div class="preflight ${cls}">
      <div class="preflight-head">
        <strong>${data.safe_to_run ? "前置检查通过" : (data.ok ? "有警告，建议处理后再运行" : "前置检查失败")}</strong>
        <span>errors=${data.errors.length} warnings=${data.warnings.length}</span>
      </div>
      <div class="preflight-list">
        ${data.checks.map((item) => `
          <div class="preflight-item ${item.ok ? "ok" : item.severity}">
            <span>${item.ok ? "OK" : item.severity.toUpperCase()}</span>
            <p>${esc(item.message)}</p>
          </div>
        `).join("")}
      </div>
    </div>
  `;
}

function showPreflight(card, data) {
  const box = card.querySelector(".preflight-box");
  box.classList.remove("hidden");
  box.innerHTML = renderPreflight(data);
  recordCommand("proof 前置检查", data.command);
  if (!data.safe_to_run) {
    const text = [
      ...data.errors.map((x) => x.message),
      ...data.warnings.map((x) => x.message),
    ].join("\n");
    showErrorHints(text);
  }
}

function modelCard(model) {
  const selected = state.selected.has(model.name);
  return `
    <div class="model-card ${selected ? "selected" : ""}" data-model="${esc(model.name)}">
      <div class="model-title">
        <input type="checkbox" class="model-check" ${selected ? "checked" : ""} />
        <strong>${esc(model.name)}</strong>
      </div>
      <div class="model-grid">
        <label class="wide">weight
          <input class="model-weight" value="${esc(model.weight)}" />
        </label>
        <label>score
          <input class="model-score" type="number" step="0.01" value="${model.score}" />
        </label>
        <label>priority
          <select class="model-priority">
            <option value="supportive" ${model.priority === "supportive" ? "selected" : ""}>supportive</option>
            <option value="primary" ${model.priority === "primary" ? "selected" : ""}>primary</option>
          </select>
        </label>
        <div class="segmented-field">
          <span>quantization</span>
          <div class="segmented" role="radiogroup" aria-label="quantization">
            ${["none", "8bit", "4bit"].map((q) => `
              <label>
                <input type="radio" name="quantization-${esc(model.name)}" value="${q}" ${model.quantization === q ? "checked" : ""} />
                <span>${q}</span>
              </label>
            `).join("")}
          </div>
        </div>
        <label>max_memory
          <input class="model-memory" value="${esc(model.max_memory)}" />
        </label>
        <label>num_gpus
          <input class="model-gpus" type="number" step="0.5" value="${model.num_gpus}" />
        </label>
      </div>
    </div>
  `;
}

function bindModelEvents() {
  document.querySelectorAll(".model-card").forEach((card) => {
    const name = card.dataset.model;
    const check = card.querySelector(".model-check");
    check.addEventListener("change", () => {
      if (check.checked) state.selected.add(name);
      else state.selected.delete(name);
      card.classList.toggle("selected", check.checked);
      renderProofRows();
    });
  });
}

function collectSelectedModels() {
  const models = [];
  document.querySelectorAll(".model-card").forEach((card) => {
    const name = card.dataset.model;
    if (!state.selected.has(name)) return;
    models.push({
      name,
      weight: card.querySelector(".model-weight").value,
      score: Number(card.querySelector(".model-score").value),
      priority: card.querySelector(".model-priority").value,
      quantization: card.querySelector('input[type="radio"]:checked').value,
      max_memory: card.querySelector(".model-memory").value,
      num_gpus: Number(card.querySelector(".model-gpus").value),
    });
  });
  return models;
}

async function loadModels() {
  const data = await api("/ui/models");
  state.models = data.models;
  $("modelList").innerHTML = data.models.map(modelCard).join("");
  bindModelEvents();
  renderProofRows();
  log("模型列表已加载");
}

async function refreshStatus() {
  const data = await api("/ui/status");
  $("projectRoot").textContent = data.project_root;
  const svc = data.service;
  if (svc.ready) setBadge($("serviceBadge"), `API: ready :${svc.port}`, "ok");
  else if (svc.running) setBadge($("serviceBadge"), `API: starting :${svc.port}`, "warn");
  else setBadge($("serviceBadge"), "API: stopped", "");
  if (svc.config_path) $("configPath").textContent = state.configPath || svc.config_path;
  if (svc.log_tail) renderTokenTrace(svc.log_tail);
  return data;
}

async function checkGpu() {
  const data = await api("/ui/gpu");
  const used = data.gpus.reduce((sum, g) => sum + g.memory_used_mb, 0);
  const total = data.gpus.reduce((sum, g) => sum + g.memory_total_mb, 0);
  setBadge($("gpuBadge"), `GPU: ${used}/${total} MB`, used > 0 ? "warn" : "ok");
  log("GPU 状态", data);
}

function checkRay() {
  setBadge($("rayBadge"), "Ray: 通过服务日志/Ray stop 检查", "warn");
  log("Ray 检查：当前 MVP 通过启动/停止服务和日志判断；如需精确状态可扩展 /ui/ray。");
}

async function generateYaml() {
  setStep("serviceStep", "生成配置中...", "loading");
  const models = collectSelectedModels();
  const primaryCount = models.filter((m) => m.priority === "primary").length;
  if (models.length < 2) throw new Error("至少选择两个模型。");
  if (primaryCount > 1) throw new Error("最多只能有一个 primary 模型。");
  const data = await api("/ui/config", {
    method: "POST",
    body: JSON.stringify({
      models,
      norm_type: $("normType").value,
      threshold: Number($("threshold").value),
    }),
  });
  state.configPath = data.path;
  $("configPath").textContent = data.path;
  $("yamlPreview").value = data.yaml;
  showCommand("serviceCommand", "生成 YAML", data.command);
  setStep("serviceStep", "配置已生成", "ok");
  log("YAML 已生成", data.path);
}

async function startServer() {
  setStep("serviceStep", "启动服务中...", "loading");
  if (!state.configPath) await generateYaml();
  const data = await api("/ui/start-server", {
    method: "POST",
    body: JSON.stringify({
      config_path: state.configPath,
      host: $("serverHost").value,
      port: Number($("serverPort").value),
      use_model_name_evidence: $("useModelNameEvidence").value === "true",
    }),
  });
  log("融合服务启动中", data);
  showCommand("serviceCommand", "启动融合服务", `${data.command}\n# env: ZK_EVIDENCE_USE_MODEL_NAME=${data.env?.ZK_EVIDENCE_USE_MODEL_NAME ?? "1"}`);
  setBadge($("serviceBadge"), "API: starting", "warn");
  setStep("serviceStep", `启动中 pid=${data.pid}`, "loading");
  setTimeout(refreshStatus, 2500);
}

async function stopServer() {
  setStep("serviceStep", "停止服务并清理 Ray...", "loading");
  const data = await api("/ui/stop-server", { method: "POST", body: "{}" });
  log("融合服务已停止", data);
  showCommand("serviceCommand", "停止融合服务并清理 Ray", `${data.command}\n${data.ray_stop?.command || ""}`);
  setStep("serviceStep", "已停止，Ray 已清理", data.ok ? "ok" : "err");
  await refreshStatus();
}

async function generateAnswer() {
  setStep("generateStep", "生成中...", "loading");
  let rid = $("requestId").value.trim();
  if (!rid) {
    rid = `ui_${Date.now()}`;
    $("requestId").value = rid;
  }
  state.requestId = rid;
  $("requestBadge").textContent = `request_id: ${rid}`;
  const until = $("until").value.trim();
  const data = await api("/ui/generate", {
    method: "POST",
    body: JSON.stringify({
      prompt: $("prompt").value,
      request_id: rid,
      max_new_tokens: Number($("maxNewTokens").value),
      min_new_tokens: Math.min(Number($("minNewTokens").value), Number($("maxNewTokens").value)),
      apply_chat_template: $("applyChatTemplate").value === "true",
      until: until ? until.split(",").map((s) => s.trim()).filter(Boolean) : null,
      port: Number($("serverPort").value),
    }),
  });
  state.requestId = data.request_id;
  $("requestId").value = data.request_id;
  $("requestBadge").textContent = `request_id: ${data.request_id}`;
  $("answer").textContent = data.answer || "(空回答)";
  renderTokenTrace(data.log_tail || "");
  showCommand("generateCommand", "发送生成请求", `${data.command}\n# upstream: POST ${data.upstream_url}`);
  setStep("generateStep", `生成成功，${data.elapsed_seconds}s`, "ok");
  log("生成完成", data);
  await scanEvidence();
  await refreshMetrics();
}

function renderTokenTrace(logTail) {
  const lines = logTail.split("\n").filter((line) =>
    line.includes("Token from Model") || line.includes("Token chosen by GaC") || line.includes("Generated text")
  );
  if (lines.length) $("tokenTrace").textContent = lines.slice(-80).join("\n");
}

function checkItem(label, ok) {
  return `<span class="check ${ok ? "ok" : ""}">${ok ? "OK" : "MISS"} ${label}</span>`;
}

async function scanEvidence() {
  setStep("generateStep", "扫描证据中...", "loading");
  const rid = $("requestId").value.trim() || state.requestId;
  if (!rid) throw new Error("缺少 request_id。");
  state.requestId = rid;
  const data = await api(`/ui/evidence/${encodeURIComponent(rid)}`);
  state.evidenceModels = data.models || [];
  $("evidencePath").textContent = data.path;
  $("copyEvidencePathBtn").dataset.copy = data.path;
  showCommand("evidenceCommand", "扫描 evidence", data.command);
  $("evidenceWarning").classList.toggle("hidden", !data.has_weight_bins);
  $("evidenceSummary").innerHTML = state.evidenceModels.map((m) => `
    <div class="info-card ${m.weight_bin_count > 0 ? "warning" : ""}">
      <h2>${esc(m.model_name)}</h2>
      ${renderPath(m.path)}
      ${m.weight_bin_count > 0 ? `<div class="inline-warning">证据中包含重复权重副本，会导致证明开销异常，请检查 hook 逻辑。</div>` : ""}
      <p>文件 ${m.file_count} 个，大小 ${esc(m.human_size)}</p>
      <p>weight_*.bin: <strong>${m.weight_bin_count}</strong></p>
      <p>seq=${esc(m.meta.seq_len || "-")} embed=${esc(m.meta.embed_dim || "-")} hidden=${esc(m.meta.hidden_dim || "-")}</p>
      <div class="check-grid">
        ${checkItem("input_int", m.checks.input_int)}
        ${checkItem("rms_inv", m.checks.rms_inv)}
        ${checkItem("rmsnorm", m.checks.rmsnorm)}
        ${checkItem("attention", m.checks.attention)}
        ${checkItem("ffn", m.checks.ffn)}
        ${checkItem("meta", m.checks.attn_meta && m.checks.ffn_meta)}
      </div>
    </div>
  `).join("") || `<div class="info-card"><h2>未找到证据</h2><p>请先发送生成请求。</p></div>`;
  bindCopyButtons($("evidenceSummary"));
  setStep("generateStep", data.exists ? "证据扫描成功" : "未找到证据", data.exists ? "ok" : "err");
  renderProofRows();
  log("证据扫描完成", data);
}

function renderProofRows() {
  const names = state.evidenceModels.length
    ? state.evidenceModels.map((m) => m.model_name)
    : Array.from(state.selected);
  $("proofList").innerHTML = names.map((name) => `
    <div class="proof-card" data-proof-model="${esc(name)}">
      <h2>${esc(name)}</h2>
      <label>zk_weights 目录
        <input class="weights-dir" value="zk_weights/${esc(name)}" />
      </label>
      <div class="note small">保留 zk_weights：这里读取长期权重，不会从 evidence 重新生成权重副本。</div>
      <div class="actions">
        <button class="preflight-btn">前置检查</button>
        <button class="prove-btn primary">生成 proof</button>
        <button class="reprove-btn">重新生成 proof</button>
        <button class="verify-btn">只验证已有 proof</button>
      </div>
      ${renderPath(`zk_evidence/${state.requestId || "request_id"}/transformer_proof_${name}.bin`)}
      <code class="proof-path hidden">zk_evidence/${esc(state.requestId || "request_id")}/transformer_proof_${esc(name)}.bin</code>
      <div class="preflight-box hidden"></div>
      <div class="proof-command command-box hidden"></div>
      <p class="proof-status">未运行</p>
    </div>
  `).join("");
  document.querySelectorAll(".proof-card").forEach((card) => {
    card.querySelector(".preflight-btn").addEventListener("click", () => runPreflight(card, { manual: true }));
    card.querySelector(".prove-btn").addEventListener("click", () => runProve(card));
    card.querySelector(".reprove-btn").addEventListener("click", () => runProve(card));
    card.querySelector(".verify-btn").addEventListener("click", () => runVerify(card));
  });
  bindCopyButtons($("proofList"));
}

async function runPreflight(card, options = {}) {
  const model = card.dataset.proofModel;
  const rid = $("requestId").value.trim() || state.requestId;
  if (!rid) throw new Error("缺少 request_id。");
  const btn = card.querySelector(".preflight-btn");
  const status = card.querySelector(".proof-status");
  setButtonLoading(btn, true, "检查中...");
  status.className = "proof-status loading";
  status.textContent = "proof 前置检查中...";
  try {
    const data = await api("/ui/prove/preflight", {
      method: "POST",
      body: JSON.stringify({
        request_id: rid,
        model_name: model,
        weights_dir: card.querySelector(".weights-dir").value,
      }),
    });
    showPreflight(card, data);
    if (data.safe_to_run) {
      status.className = "proof-status ok";
      status.textContent = "前置检查通过";
    } else if (data.ok) {
      status.className = "proof-status err";
      status.textContent = "前置检查有警告";
    } else {
      status.className = "proof-status err";
      status.textContent = "前置检查失败";
    }
    log(`proof 前置检查：${model}`, data);
    return data;
  } finally {
    setButtonLoading(btn, false);
  }
}

async function runProve(card) {
  const model = card.dataset.proofModel;
  const rid = $("requestId").value.trim() || state.requestId;
  if (!rid) throw new Error("缺少 request_id。");
  const btn = card.querySelector(".prove-btn");
  const status = card.querySelector(".proof-status");
  setButtonLoading(btn, true, "生成中...");
  status.className = "proof-status loading";
  status.textContent = "生成 proof 中...";
  try {
    const preflight = await runPreflight(card);
    if (!preflight.ok) {
      throw new Error("proof 前置检查失败，请先修复红色 ERROR 项。");
    }
    if (!preflight.safe_to_run) {
      const proceed = confirm("proof 前置检查存在 WARNING。建议先停止融合服务并清理 Ray/GPU。仍要继续生成 proof 吗？");
      if (!proceed) {
        status.className = "proof-status err";
        status.textContent = "已取消 proof";
        return;
      }
    }
    setButtonLoading(btn, true, "生成中...");
    status.className = "proof-status loading";
    status.textContent = "生成 proof 中...";
    const data = await api("/ui/prove", {
      method: "POST",
      body: JSON.stringify({
        request_id: rid,
        model_name: model,
        weights_dir: card.querySelector(".weights-dir").value,
      }),
    });
    card.querySelector(".proof-path").textContent = data.proof_file;
    const pathRow = card.querySelector(".path-row");
    pathRow.outerHTML = renderPath(data.proof_file);
    bindCopyButtons(card);
    showInlineCommand(card, "生成/重新生成 proof", `${data.ui_command || ""}\n${data.command || ""}`);
    status.className = `proof-status ${data.ok ? "ok" : "err"}`;
    status.textContent = data.ok ? `proof 完成，大小 ${data.proof_human_size}` : "proof 失败";
    if (!data.ok) showErrorHints(`${data.stdout || ""}\n${data.stderr || ""}`);
    log(`proof 结果：${model}`, data.stdout + data.stderr);
    await refreshMetrics();
  } catch (err) {
    status.className = "proof-status err";
    status.textContent = `proof 失败：${err.message}`;
    showErrorHints(err);
    log(`proof 失败：${model}`, err.message);
  } finally {
    setButtonLoading(btn, false);
  }
}

async function runVerify(card) {
  const model = card.dataset.proofModel;
  const proofFile = card.querySelector(".proof-path").textContent;
  const btn = card.querySelector(".verify-btn");
  const status = card.querySelector(".proof-status");
  setButtonLoading(btn, true, "验证中...");
  status.className = "proof-status loading";
  status.textContent = "验证中...";
  try {
    const data = await api("/ui/verify", {
      method: "POST",
      body: JSON.stringify({ proof_file: proofFile, model_name: model }),
    });
    showInlineCommand(card, "只验证已有 proof", `${data.ui_command || ""}\n${data.command || ""}`);
    status.className = `proof-status ${data.verified ? "ok" : "err"}`;
    status.textContent = data.verified ? "验证通过" : "验证失败";
    if (!data.verified) showErrorHints(`${data.stdout || ""}\n${data.stderr || ""}`);
    log(`verify 结果：${model}`, data.stdout + data.stderr);
    await refreshMetrics();
  } catch (err) {
    status.className = "proof-status err";
    status.textContent = `验证失败：${err.message}`;
    showErrorHints(err);
    log(`verify 失败：${model}`, err.message);
  } finally {
    setButtonLoading(btn, false);
  }
}

async function refreshMetrics() {
  const rid = $("requestId").value.trim() || state.requestId;
  if (!rid) return;
  const data = await api(`/ui/metrics/${encodeURIComponent(rid)}`);
  $("evidenceBytes").textContent = data.evidence_human_size;
  $("proofBytes").textContent = data.proof_human_size;
  $("totalBytes").textContent = data.total_human_size;
  $("evidencePerToken").textContent = data.evidence_per_token_human;
  $("proofPerToken").textContent = data.proof_per_token_human;
  $("totalPerToken").textContent = data.total_per_token_human;
  $("metricsTableBody").innerHTML = `
    <tr>
      <td>${mb(data.evidence_bytes)}</td>
      <td>${mb(data.proof_bytes)}</td>
      <td>${mb(data.total_bytes)}</td>
      <td>${data.generated_tokens}</td>
      <td>${mb(data.total_bytes_per_token)}</td>
    </tr>
  `;
  recordCommand("刷新开销统计", data.command);
  log("开销统计已刷新", data);
}

function showInlineCommand(card, label, command) {
  const box = card.querySelector(".proof-command");
  box.classList.remove("hidden");
  box.innerHTML = `
    <div class="command-title">${esc(label)}</div>
    <pre>${esc(command.trim())}</pre>
    <button data-copy="${esc(command.trim())}">复制命令</button>
  `;
  bindCopyButtons(box);
  recordCommand(label, command.trim());
}

async function cleanupEvidence() {
  const rid = $("requestId").value.trim() || state.requestId;
  if (!rid) throw new Error("缺少 request_id。");
  if (!confirm(`只删除 zk_evidence/${rid}，不会删除 zk_weights。确认清理？`)) return;
  setStep("cleanupStep", "清理中...", "loading");
  const data = await api("/ui/cleanup-evidence", {
    method: "POST",
    body: JSON.stringify({ request_id: rid }),
  });
  setStep("cleanupStep", data.deleted ? `已清理 ${data.human_deleted}` : "目录不存在", "ok");
  $("evidenceSummary").innerHTML = "";
  $("evidenceWarning").classList.add("hidden");
  showCommand("evidenceCommand", "清理本次 evidence", data.command);
  log("本次 evidence 已清理，zk_weights 已保留", data);
}

function wire(id, fn) {
  $(id).addEventListener("click", async () => {
    const btn = $(id);
    try {
      setButtonLoading(btn, true);
      await fn();
    } catch (err) {
      if (id === "generateBtn") setStep("generateStep", `失败：${err.message}`, "err");
      if (["startServerBtn", "stopServerBtn", "generateYamlBtn"].includes(id)) setStep("serviceStep", `失败：${err.message}`, "err");
      if (id === "cleanupEvidenceBtn") setStep("cleanupStep", `失败：${err.message}`, "err");
      showErrorHints(err);
      log(`错误：${err.message}`);
      alert(err.message);
    } finally {
      setButtonLoading(btn, false);
    }
  });
}

function init() {
  wire("loadModelsBtn", loadModels);
  wire("generateYamlBtn", generateYaml);
  wire("startServerBtn", startServer);
  wire("stopServerBtn", stopServer);
  wire("statusBtn", refreshStatus);
  wire("gpuBtn", checkGpu);
  wire("rayBtn", checkRay);
  wire("generateBtn", generateAnswer);
  wire("scanEvidenceBtn", scanEvidence);
  wire("metricsBtn", refreshMetrics);
  wire("cleanupEvidenceBtn", cleanupEvidence);
  addCopyHandler($("copyRequestBtn"), () => $("requestId").value.trim() || state.requestId);
  addCopyHandler($("copyEvidencePathBtn"), () => $("copyEvidencePathBtn").dataset.copy || $("evidencePath").textContent);
  $("clearCommandsBtn").addEventListener("click", () => $("commandHistory").innerHTML = "");
  $("clearErrorsBtn").addEventListener("click", () => $("errorHints").textContent = "暂无错误。");
  $("clearLogBtn").addEventListener("click", () => $("logOutput").textContent = "");
  loadModels().then(refreshStatus).catch((err) => log(`初始化失败：${err.message}`));
}

init();
