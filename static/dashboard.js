/* =============================================
   Context Analyzer Dashboard — Logic
   ============================================= */

(() => {
  "use strict";

  // ---- State ----
  let currentSession = null;
  let turnData = [];        // from /api/session/{id}/turns -> .turns array
  let blocksData = [];      // from /api/session/{id}/blocks -> .blocks array
  let sessionMeta = {};     // model, turn_count, block_count from turns response
  let currentTurn = 0;
  let maxTurn = 0;
  let playing = false;
  let playInterval = null;
  let playSpeed = 1;        // 1, 2, 4
  let sedimentChart = null;

  // ---- DOM refs ----
  const $sessionSelect = document.getElementById("session-select");
  const $healthDot = document.getElementById("health-dot");
  const $healthLabel = document.getElementById("health-label");
  const $scDeadweight = document.getElementById("sc-deadweight");
  const $scDeadweightSub = document.getElementById("sc-deadweight-sub");
  const $scContext = document.getElementById("sc-context");
  const $scContextSub = document.getElementById("sc-context-sub");
  const $scCache = document.getElementById("sc-cache");
  const $scCacheSub = document.getElementById("sc-cache-sub");
  const $scTools = document.getElementById("sc-tools");
  const $scToolsSub = document.getElementById("sc-tools-sub");
  const $playBtn = document.getElementById("play-btn");
  const $playIcon = document.getElementById("play-icon");
  const $pauseIcon = document.getElementById("pause-icon");
  const $turnSlider = document.getElementById("turn-slider");
  const $turnLabel = document.getElementById("turn-label");
  const $speedBtn = document.getElementById("speed-btn");
  const $tapeRows = document.getElementById("tape-rows");
  const $recsRecoverable = document.getElementById("recs-recoverable");
  const $recsList = document.getElementById("recs-list");
  const $turnMessages = document.getElementById("turn-messages");
  const $drilldownLink = document.getElementById("drilldown-link");
  const $modalOverlay = document.getElementById("modal-overlay");
  const $modalTitle = document.getElementById("modal-title");
  const $modalClose = document.getElementById("modal-close");
  const $modalPills = document.getElementById("modal-pills");
  const $modalStats = document.getElementById("modal-stats");
  const $modalMessages = document.getElementById("modal-messages");
  const $modalStale = document.getElementById("modal-stale");
  const $modalPrev = document.getElementById("modal-prev");
  const $modalNext = document.getElementById("modal-next");

  // ---- API helpers ----
  async function api(path) {
    const resp = await fetch(path);
    if (!resp.ok) throw new Error(`API ${path}: ${resp.status}`);
    return resp.json();
  }

  // ---- Formatting ----
  function fmtTokens(n) {
    if (n >= 1_000_000) return (n / 1_000_000).toFixed(1) + "M";
    if (n >= 1_000) return (n / 1_000).toFixed(0) + "K";
    return String(n);
  }

  function fmtPct(ratio) {
    return (ratio * 100).toFixed(1) + "%";
  }

  function fmtCost(dollars) {
    if (dollars < 0.01) return "<$0.01";
    return "$" + dollars.toFixed(2);
  }

  // ---- Init ----
  async function init() {
    try {
      const sessions = await api("/api/sessions");
      $sessionSelect.innerHTML = "";
      if (!sessions || sessions.length === 0) {
        $sessionSelect.innerHTML = '<option value="">No sessions found</option>';
        return;
      }
      sessions.forEach((sid) => {
        const opt = document.createElement("option");
        opt.value = sid;
        opt.textContent = sid.length > 30 ? sid.slice(0, 12) + "..." + sid.slice(-12) : sid;
        $sessionSelect.appendChild(opt);
      });
      // Load first session
      loadSession(sessions[0]);
    } catch (e) {
      console.error("Failed to load sessions:", e);
      $sessionSelect.innerHTML = '<option value="">Error loading sessions</option>';
    }
  }

  // ---- Load session ----
  async function loadSession(sessionId) {
    currentSession = sessionId;
    $sessionSelect.value = sessionId;

    try {
      const [turnsResp, blocksResp] = await Promise.all([
        api(`/api/session/${encodeURIComponent(sessionId)}/turns`),
        api(`/api/session/${encodeURIComponent(sessionId)}/blocks`),
      ]);
      turnData = turnsResp.turns || [];
      sessionMeta = turnsResp;
      blocksData = blocksResp.blocks || [];
    } catch (e) {
      console.error("Failed to load session:", e);
      turnData = [];
      blocksData = [];
      sessionMeta = {};
    }

    maxTurn = Math.max(0, turnData.length - 1);
    currentTurn = maxTurn; // Start at the latest turn

    $turnSlider.min = 0;
    $turnSlider.max = maxTurn;
    $turnSlider.value = maxTurn;

    updateScorecards();
    buildSedimentChart();
    updateTurnView(maxTurn);
    updateHealthDot();
  }

  // ---- Update scorecards ----
  function updateScorecards() {
    if (turnData.length === 0) {
      $scDeadweight.textContent = "--";
      $scDeadweightSub.textContent = "";
      $scContext.textContent = "--";
      $scContextSub.textContent = "";
      $scCache.textContent = "--";
      $scCacheSub.textContent = "";
      $scTools.textContent = "--";
      $scToolsSub.textContent = "";
      return;
    }

    const snap = turnData[currentTurn];
    if (!snap) return;

    // Dead weight ratio
    const deadRatio = snap.stale_tokens / Math.max(1, snap.total_tokens);
    $scDeadweight.textContent = fmtPct(deadRatio);
    $scDeadweightSub.textContent = fmtTokens(snap.stale_tokens) + " stale tokens";

    // Context utilization
    $scContext.textContent = fmtTokens(snap.total_tokens);
    const model = sessionMeta.model || "";
    const windowSize = model.includes("[1m]") ? 1_000_000 : 200_000;
    $scContextSub.textContent = fmtPct(snap.total_tokens / windowSize) + " of " + fmtTokens(windowSize);

    // Cache hit rate (cumulative up to current turn)
    let totalCacheRead = 0;
    let totalAll = 0;
    for (let i = 0; i <= currentTurn; i++) {
      const t = turnData[i];
      totalCacheRead += t.cache_read_tokens || 0;
      totalAll += (t.cache_read_tokens || 0) + (t.cache_creation_tokens || 0) + (t.input_tokens || 0);
    }
    const cacheRate = totalAll > 0 ? totalCacheRead / totalAll : 0;
    $scCache.textContent = fmtPct(cacheRate);
    $scCacheSub.textContent = fmtTokens(totalCacheRead) + " cache read tokens";

    // Block count
    $scTools.textContent = String(snap.block_count);
    $scToolsSub.textContent = snap.stale_block_count + " stale blocks";
  }

  // ---- Health dot ----
  function updateHealthDot() {
    if (turnData.length === 0) {
      $healthDot.className = "health-dot";
      $healthLabel.textContent = "--";
      return;
    }
    const snap = turnData[currentTurn];
    if (!snap) {
      $healthDot.className = "health-dot";
      $healthLabel.textContent = "--";
      return;
    }
    const deadRatio = snap.stale_tokens / Math.max(1, snap.total_tokens);
    if (deadRatio < 0.15) {
      $healthDot.className = "health-dot green";
      $healthLabel.textContent = "Healthy";
    } else if (deadRatio < 0.35) {
      $healthDot.className = "health-dot yellow";
      $healthLabel.textContent = "Degrading";
    } else {
      $healthDot.className = "health-dot red";
      $healthLabel.textContent = "Attention needed";
    }
  }

  // ---- Sediment chart ----
  function buildSedimentChart() {
    const canvas = document.getElementById("sediment-chart");
    if (!canvas) return;

    if (sedimentChart) {
      sedimentChart.destroy();
      sedimentChart = null;
    }

    if (turnData.length === 0) return;

    const labels = turnData.map((t) => t.turn);
    const systemData = turnData.map((t) => t.system_tokens);
    const activeData = turnData.map((t) => t.active_tokens);
    const staleData = turnData.map((t) => t.stale_tokens);

    sedimentChart = new Chart(canvas, {
      type: "line",
      data: {
        labels: labels,
        datasets: [
          {
            label: "System + Skills",
            data: systemData,
            backgroundColor: "rgba(167, 139, 250, 0.6)",
            borderColor: "rgba(167, 139, 250, 0.8)",
            borderWidth: 1,
            fill: true,
            tension: 0.3,
            pointRadius: 0,
            order: 3,
          },
          {
            label: "Active Context",
            data: activeData,
            backgroundColor: "rgba(52, 211, 153, 0.5)",
            borderColor: "rgba(5, 150, 105, 0.8)",
            borderWidth: 1,
            fill: true,
            tension: 0.3,
            pointRadius: 0,
            order: 2,
          },
          {
            label: "Stale Context",
            data: staleData,
            backgroundColor: "rgba(252, 165, 165, 0.5)",
            borderColor: "rgba(220, 38, 38, 0.7)",
            borderWidth: 1,
            fill: true,
            tension: 0.3,
            pointRadius: 0,
            order: 1,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        interaction: {
          mode: "index",
          intersect: false,
        },
        scales: {
          x: {
            title: { display: true, text: "Turn", font: { size: 11, family: "'Inter'" } },
            grid: { display: false },
            ticks: { font: { size: 10, family: "'JetBrains Mono'" }, maxTicksLimit: 20 },
          },
          y: {
            stacked: true,
            title: { display: true, text: "Tokens", font: { size: 11, family: "'Inter'" } },
            grid: { color: "#f3f4f6" },
            ticks: {
              font: { size: 10, family: "'JetBrains Mono'" },
              callback: (v) => fmtTokens(v),
            },
          },
        },
        plugins: {
          legend: {
            position: "bottom",
            labels: {
              usePointStyle: true,
              pointStyle: "rectRounded",
              padding: 16,
              font: { size: 11, family: "'Inter'" },
            },
          },
          tooltip: {
            callbacks: {
              label: (ctx) => ctx.dataset.label + ": " + fmtTokens(ctx.raw),
            },
          },
        },
      },
      plugins: [
        {
          id: "turnMarker",
          afterDraw(chart) {
            if (maxTurn === 0) return;
            const meta = chart.getDatasetMeta(0);
            if (!meta.data[currentTurn]) return;
            const x = meta.data[currentTurn].x;
            const ctx = chart.ctx;
            const yTop = chart.chartArea.top;
            const yBottom = chart.chartArea.bottom;

            ctx.save();
            ctx.setLineDash([4, 4]);
            ctx.strokeStyle = "#374151";
            ctx.lineWidth = 1.5;
            ctx.beginPath();
            ctx.moveTo(x, yTop);
            ctx.lineTo(x, yBottom);
            ctx.stroke();
            ctx.restore();

            // Turn number label
            ctx.save();
            ctx.font = '500 10px "JetBrains Mono"';
            ctx.fillStyle = "#374151";
            ctx.textAlign = "center";
            ctx.fillText("T" + (turnData[currentTurn]?.turn || currentTurn), x, yTop - 4);
            ctx.restore();
          },
        },
      ],
    });
  }

  // ---- Update turn marker on chart ----
  function updateChartMarker() {
    if (sedimentChart) {
      sedimentChart.update("none"); // triggers plugin re-draw without animation
    }
  }

  // ---- Context tape ----
  function updateTape(turnIndex) {
    if (blocksData.length === 0) {
      $tapeRows.innerHTML = '<div class="tape-empty">No blocks for this session</div>';
      return;
    }

    // The turnNumber from the API is 1-based
    const turnNumber = turnData[turnIndex] ? turnData[turnIndex].turn : turnIndex + 1;

    // Filter blocks visible at this turn
    const visibleBlocks = blocksData.filter((b) => b.turn_entered <= turnNumber);

    // Sort: pinned first, then stale (need attention), then by size descending
    visibleBlocks.sort((a, b) => {
      if (a.is_pinned && !b.is_pinned) return -1;
      if (!a.is_pinned && b.is_pinned) return 1;
      const aStale = a.staleness_label === "stale" || a.staleness_label === "dead_weight";
      const bStale = b.staleness_label === "stale" || b.staleness_label === "dead_weight";
      if (aStale && !bStale) return -1;
      if (!aStale && bStale) return 1;
      return b.size_tokens_est - a.size_tokens_est;
    });

    let html = "";
    visibleBlocks.slice(0, 15).forEach((block) => {
      const label = block.tool_name
        ? `${block.tool_name}: ${block.resource || "unknown"}`
        : block.block_type;
      const age = turnNumber - block.turn_entered;
      const barPct = Math.min(100, ((age + 1) / Math.max(1, turnNumber)) * 100);
      const status = block.is_pinned ? "pinned" : block.staleness_label;
      const barClass = block.is_pinned
        ? "pinned"
        : status === "stale" || status === "dead_weight"
          ? "stale"
          : status === "warm"
            ? "warm"
            : "active";

      html += `<div class="tape-row">
        <div class="tape-block-label" title="${escapeHtml(label)}">${escapeHtml(label)}</div>
        <div class="tape-bar-container">
          <div class="tape-bar ${barClass}" style="width:${barPct}%"></div>
        </div>
        <div class="tape-size">${fmtTokens(block.size_tokens_est)}</div>
        <div class="tape-status ${status}">${status.replace("_", " ")}</div>
      </div>`;
    });

    if (visibleBlocks.length > 15) {
      html += `<div class="tape-empty">${visibleBlocks.length - 15} more blocks...</div>`;
    }

    $tapeRows.innerHTML = html || '<div class="tape-empty">No blocks</div>';
  }

  // ---- Recommendations ----
  function updateRecommendations(turnIndex) {
    const turnNumber = turnData[turnIndex] ? turnData[turnIndex].turn : turnIndex + 1;

    const staleBlocks = blocksData.filter(
      (b) => !b.is_pinned && (b.staleness_label === "stale" || b.staleness_label === "dead_weight") && b.turn_entered <= turnNumber
    );

    const recoverable = staleBlocks.reduce((s, b) => s + b.size_tokens_est, 0);
    $recsRecoverable.textContent = fmtTokens(recoverable);

    if (staleBlocks.length === 0) {
      $recsList.innerHTML = '<div class="tape-empty">No stale blocks to reclaim</div>';
      return;
    }

    const sorted = [...staleBlocks].sort((a, b) => b.size_tokens_est - a.size_tokens_est).slice(0, 8);

    let html = "";
    sorted.forEach((block) => {
      const label = block.tool_name
        ? `${block.tool_name}: ${block.resource || "unknown"}`
        : block.block_type;
      const staleTurns = turnNumber - block.turn_entered;
      html += `<div class="rec-item">
        <div class="rec-tool">${escapeHtml(label)}</div>
        <div class="rec-meta">
          <span>entered T${block.turn_entered}</span>
          <span>stale ${staleTurns} turns</span>
          <span>${fmtTokens(block.size_tokens_est)}</span>
        </div>
      </div>`;
    });

    $recsList.innerHTML = html;
  }

  // ---- Turn details (lazy load messages) ----
  async function updateTurnDetails(turnIndex) {
    const turnNumber = turnData[turnIndex] ? turnData[turnIndex].turn : turnIndex + 1;

    try {
      const data = await api(
        `/api/session/${encodeURIComponent(currentSession)}/turn/${turnNumber}/messages`
      );
      const msgs = data.messages || [];

      if (msgs.length === 0) {
        $turnMessages.innerHTML = '<div class="tape-empty">No messages for this turn</div>';
        return;
      }

      let html = "";
      msgs.forEach((msg) => {
        const roleClass = msg.block_type.replace(" ", "_");
        const roleLabel = msg.block_type === "tool_use"
          ? "TOOL USE"
          : msg.block_type === "tool_result"
            ? "TOOL RESULT"
            : msg.block_type.toUpperCase().replace("_", " ");
        const preview = msg.content.length > 120
          ? msg.content.slice(0, 120) + "..."
          : msg.content;

        html += `<div class="turn-msg">
          <span class="role-badge ${roleClass}">${roleLabel}</span>
          <span class="msg-preview">${escapeHtml(preview)}</span>
          <span class="msg-size">${fmtTokens(msg.size_tokens_est)}</span>
        </div>`;
      });

      $turnMessages.innerHTML = html;
    } catch (e) {
      $turnMessages.innerHTML = '<div class="tape-empty">Could not load messages</div>';
    }
  }

  // ---- Drilldown modal ----
  function openDrilldown(turnIndex) {
    $modalOverlay.classList.add("open");
    document.body.style.overflow = "hidden";
    renderDrilldown(turnIndex);
  }

  function closeDrilldown() {
    $modalOverlay.classList.remove("open");
    document.body.style.overflow = "";
  }

  async function renderDrilldown(turnIndex) {
    const snap = turnData[turnIndex];
    const turnNumber = snap ? snap.turn : turnIndex + 1;

    $modalTitle.textContent = "Turn " + turnNumber + (snap ? " \u2014 " + fmtTokens(snap.total_tokens) + " tokens" : "");

    // Compute stale blocks at this turn
    const staleBlocks = blocksData.filter(
      (b) => !b.is_pinned && (b.staleness_label === "stale" || b.staleness_label === "dead_weight") && b.turn_entered <= turnNumber
    );

    // Pills
    const pills = [
      { label: "Blocks", value: snap ? snap.block_count : 0 },
      { label: "API calls", value: snap ? snap.api_call_count : 0 },
      { label: "Stale blocks", value: staleBlocks.length },
    ];
    $modalPills.innerHTML = pills
      .map((p) => `<span class="pill">${p.label} <span class="pill-value">${p.value}</span></span>`)
      .join("");

    // Stats panels
    if (snap) {
      const model = sessionMeta.model || "";
      const windowSize = model.includes("[1m]") ? 1_000_000 : 200_000;
      const recoverable = staleBlocks.reduce((s, b) => s + b.size_tokens_est, 0);

      $modalStats.innerHTML = `
        <div class="stat-panel">
          <div class="stat-panel-title">Composition</div>
          <div class="stat-panel-rows">
            <div class="stat-row"><span class="stat-row-label">System</span><span class="stat-row-value">${fmtTokens(snap.system_tokens)}</span></div>
            <div class="stat-row"><span class="stat-row-label">Active</span><span class="stat-row-value">${fmtTokens(snap.active_tokens)}</span></div>
            <div class="stat-row"><span class="stat-row-label">Stale</span><span class="stat-row-value">${fmtTokens(snap.stale_tokens)}</span></div>
          </div>
        </div>
        <div class="stat-panel">
          <div class="stat-panel-title">This Turn</div>
          <div class="stat-panel-rows">
            <div class="stat-row"><span class="stat-row-label">Blocks</span><span class="stat-row-value">${snap.block_count}</span></div>
            <div class="stat-row"><span class="stat-row-label">Total tokens</span><span class="stat-row-value">${fmtTokens(snap.total_tokens)}</span></div>
            <div class="stat-row"><span class="stat-row-label">Utilization</span><span class="stat-row-value">${fmtPct(snap.total_tokens / windowSize)}</span></div>
          </div>
        </div>
        <div class="stat-panel">
          <div class="stat-panel-title">Staleness at Turn</div>
          <div class="stat-panel-rows">
            <div class="stat-row"><span class="stat-row-label">Dead weight</span><span class="stat-row-value">${fmtPct(snap.stale_tokens / Math.max(1, snap.total_tokens))}</span></div>
            <div class="stat-row"><span class="stat-row-label">Stale blocks</span><span class="stat-row-value">${staleBlocks.length}</span></div>
            <div class="stat-row"><span class="stat-row-label">Recoverable</span><span class="stat-row-value">${fmtTokens(recoverable)}</span></div>
          </div>
        </div>
        <div class="stat-panel">
          <div class="stat-panel-title">API Tokens</div>
          <div class="stat-panel-rows">
            <div class="stat-row"><span class="stat-row-label">Input</span><span class="stat-row-value">${fmtTokens(snap.input_tokens)}</span></div>
            <div class="stat-row"><span class="stat-row-label">Output</span><span class="stat-row-value">${fmtTokens(snap.output_tokens)}</span></div>
            <div class="stat-row"><span class="stat-row-label">Cache read</span><span class="stat-row-value">${fmtTokens(snap.cache_read_tokens)}</span></div>
          </div>
        </div>`;
    } else {
      $modalStats.innerHTML = "";
    }

    // Messages (lazy fetch)
    $modalMessages.innerHTML = '<div class="tape-empty">Loading messages...</div>';
    try {
      const data = await api(
        `/api/session/${encodeURIComponent(currentSession)}/turn/${turnNumber}/messages`
      );
      const msgs = data.messages || [];
      let msgHtml = "";
      msgs.forEach((msg) => {
        const roleClass = msg.block_type.replace(" ", "_");
        const roleLabel = msg.block_type === "tool_use"
          ? "TOOL USE"
          : msg.block_type === "tool_result"
            ? "TOOL RESULT"
            : msg.block_type.toUpperCase().replace("_", " ");
        const maxSize = 8000;
        const barPct = Math.min(100, (msg.size_tokens_est / maxSize) * 100);

        msgHtml += `<div class="modal-msg">
          <div class="modal-msg-header">
            <span class="role-badge ${roleClass}">${roleLabel}</span>
            <div class="modal-msg-size-bar"><div class="modal-msg-size-fill" style="width:${barPct}%"></div></div>
            <span class="modal-msg-size-text">${fmtTokens(msg.size_tokens_est)}</span>
          </div>
          <div class="modal-msg-body">${escapeHtml(msg.content)}${msg.is_truncated ? "\n\n[truncated]" : ""}</div>
        </div>`;
      });
      $modalMessages.innerHTML = msgHtml || '<div class="tape-empty">No messages</div>';
    } catch (e) {
      $modalMessages.innerHTML = '<div class="tape-empty">Could not load messages</div>';
    }

    // Stale blocks section
    if (staleBlocks.length > 0) {
      const showCount = Math.min(3, staleBlocks.length);
      const sorted = [...staleBlocks].sort((a, b) => b.size_tokens_est - a.size_tokens_est);
      let staleHtml = `<div class="modal-stale-header">Stale blocks (${staleBlocks.length})</div><div class="modal-stale-list">`;
      sorted.slice(0, showCount).forEach((b) => {
        const label = b.tool_name ? `${b.tool_name}: ${b.resource || "unknown"}` : b.block_type;
        staleHtml += `<div class="rec-item">
          <div class="rec-tool">${escapeHtml(label)}</div>
          <div class="rec-meta"><span>${fmtTokens(b.size_tokens_est)}</span> <span>entered T${b.turn_entered}</span></div>
        </div>`;
      });
      if (staleBlocks.length > showCount) {
        staleHtml += `<div class="modal-stale-header">Show ${staleBlocks.length - showCount} more</div>`;
      }
      staleHtml += "</div>";
      $modalStale.innerHTML = staleHtml;
    } else {
      $modalStale.innerHTML = "";
    }

    // Nav buttons
    $modalPrev.disabled = turnIndex <= 0;
    $modalNext.disabled = turnIndex >= maxTurn;
  }

  // ---- Composite turn update ----
  function updateTurnView(turnIndex) {
    currentTurn = turnIndex;
    $turnSlider.value = turnIndex;
    const turnNumber = turnData[turnIndex] ? turnData[turnIndex].turn : turnIndex + 1;
    $turnLabel.textContent = `Turn ${turnNumber} / ${turnData.length}`;

    updateScorecards();
    updateTape(turnIndex);
    updateRecommendations(turnIndex);
    updateTurnDetails(turnIndex);
    updateHealthDot();
    updateChartMarker();
  }

  // ---- Play/pause ----
  function togglePlay() {
    if (playing) {
      stopPlay();
    } else {
      startPlay();
    }
  }

  function startPlay() {
    if (maxTurn === 0) return;
    playing = true;
    $playIcon.style.display = "none";
    $pauseIcon.style.display = "block";
    const intervalMs = Math.max(50, 500 / playSpeed);
    playInterval = setInterval(() => {
      if (currentTurn >= maxTurn) {
        stopPlay();
        return;
      }
      updateTurnView(currentTurn + 1);
    }, intervalMs);
  }

  function stopPlay() {
    playing = false;
    $playIcon.style.display = "block";
    $pauseIcon.style.display = "none";
    if (playInterval) {
      clearInterval(playInterval);
      playInterval = null;
    }
  }

  function cycleSpeed() {
    const speeds = [1, 2, 4];
    const idx = speeds.indexOf(playSpeed);
    playSpeed = speeds[(idx + 1) % speeds.length];
    $speedBtn.textContent = playSpeed + "x";
    if (playing) {
      stopPlay();
      startPlay();
    }
  }

  // ---- Escape HTML ----
  function escapeHtml(text) {
    const div = document.createElement("div");
    div.textContent = text;
    return div.innerHTML;
  }

  // ---- Event listeners ----
  $sessionSelect.addEventListener("change", (e) => {
    if (e.target.value) {
      stopPlay();
      loadSession(e.target.value);
    }
  });

  $playBtn.addEventListener("click", togglePlay);
  $speedBtn.addEventListener("click", cycleSpeed);

  $turnSlider.addEventListener("input", (e) => {
    stopPlay();
    updateTurnView(parseInt(e.target.value, 10));
  });

  $drilldownLink.addEventListener("click", () => {
    openDrilldown(currentTurn);
  });

  $modalClose.addEventListener("click", closeDrilldown);
  $modalOverlay.addEventListener("click", (e) => {
    if (e.target === $modalOverlay) closeDrilldown();
  });

  $modalPrev.addEventListener("click", () => {
    if (currentTurn > 0) {
      currentTurn--;
      updateTurnView(currentTurn);
      renderDrilldown(currentTurn);
    }
  });

  $modalNext.addEventListener("click", () => {
    if (currentTurn < maxTurn) {
      currentTurn++;
      updateTurnView(currentTurn);
      renderDrilldown(currentTurn);
    }
  });

  // Keyboard shortcuts
  document.addEventListener("keydown", (e) => {
    if (e.target.tagName === "INPUT" || e.target.tagName === "SELECT" || e.target.tagName === "TEXTAREA") return;

    switch (e.key) {
      case " ":
        e.preventDefault();
        togglePlay();
        break;
      case "ArrowLeft":
        e.preventDefault();
        stopPlay();
        if (currentTurn > 0) updateTurnView(currentTurn - 1);
        if ($modalOverlay.classList.contains("open")) renderDrilldown(currentTurn);
        break;
      case "ArrowRight":
        e.preventDefault();
        stopPlay();
        if (currentTurn < maxTurn) updateTurnView(currentTurn + 1);
        if ($modalOverlay.classList.contains("open")) renderDrilldown(currentTurn);
        break;
      case "Escape":
        closeDrilldown();
        break;
    }
  });

  // ---- Boot ----
  document.addEventListener("DOMContentLoaded", init);
})();
