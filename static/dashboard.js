/* =============================================
   Context Analyzer Dashboard — Logic
   ============================================= */

(() => {
  "use strict";

  // ---- State ----
  let currentSession = null;
  let sessionData = null;       // from /api/session/{id}/turns
  let currentTurn = 0;
  let maxTurn = 0;
  let playing = false;
  let playInterval = null;
  let playSpeed = 1;            // 1, 2, 4
  let sedimentChart = null;

  // TODO: These use placeholder data. Wire to real API when per-turn/per-block
  // endpoints are available.
  let demoBlocks = [];
  let demoTurnSnapshots = [];
  let demoMessages = [];

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
      sessionData = await api(`/api/session/${encodeURIComponent(sessionId)}/turns`);
    } catch (e) {
      console.error("Failed to load session:", e);
      sessionData = null;
    }

    const turnCount = sessionData ? sessionData.turn_count : 0;
    maxTurn = Math.max(0, turnCount - 1);
    currentTurn = 0;

    // Set up scrubber
    $turnSlider.min = 0;
    $turnSlider.max = maxTurn;
    $turnSlider.value = 0;

    // Generate placeholder data for this session
    generateDemoData(turnCount);

    // Update everything
    updateScorecards(sessionData);
    buildSedimentChart();
    updateTurnView(0);
    updateHealthDot();
  }

  // ---- Placeholder data generation ----
  // TODO: Replace with real API data when per-block/per-turn endpoints are wired.
  function generateDemoData(turnCount) {
    if (turnCount === 0) {
      demoBlocks = [];
      demoTurnSnapshots = [];
      demoMessages = [];
      return;
    }

    const blockTypes = [
      { type: "system", tool: "System Prompt", resource: "system", pinned: true },
      { type: "system", tool: "CLAUDE.md", resource: "CLAUDE.md", pinned: true },
    ];

    // Generate some tool-use blocks across turns
    const tools = ["Read", "Edit", "Grep", "Bash", "Write", "Glob"];
    const files = [
      "src/main.py", "src/config.ts", "tests/test_api.py", "README.md",
      "src/models.py", "src/utils.js", "package.json", "Makefile",
      "src/server.py", "src/parser.py", "docs/guide.md", "src/types.ts",
    ];

    let blockId = 2;
    for (let t = 0; t < turnCount && blockId < 30; t++) {
      if (Math.random() < 0.6) {
        const tool = tools[Math.floor(Math.random() * tools.length)];
        const file = files[Math.floor(Math.random() * files.length)];
        blockTypes.push({
          type: "tool_result",
          tool: tool,
          resource: file,
          pinned: false,
          entered: t,
          size: Math.floor(500 + Math.random() * 8000),
        });
        blockId++;
      }
    }

    demoBlocks = blockTypes.map((bt, i) => ({
      id: "block_" + i,
      type: bt.type,
      tool: bt.tool,
      resource: bt.resource,
      pinned: bt.pinned,
      entered: bt.entered || 0,
      size: bt.size || (bt.pinned ? 2000 + Math.floor(Math.random() * 3000) : 1000),
    }));

    // Build per-turn snapshots for the sediment chart
    demoTurnSnapshots = [];
    const contextWindow = 200000; // tokens
    for (let t = 0; t <= maxTurn; t++) {
      const activeBlocks = demoBlocks.filter((b) => b.entered <= t);
      const systemTokens = activeBlocks
        .filter((b) => b.pinned)
        .reduce((s, b) => s + b.size, 0);
      // Active context grows then stabilizes
      const activeRatio = Math.min(1, t / Math.max(1, maxTurn * 0.4));
      const activeContext = Math.floor(systemTokens + 20000 * activeRatio + Math.random() * 5000);
      // Stale context grows as a wedge
      const staleRatio = Math.max(0, (t - maxTurn * 0.2) / Math.max(1, maxTurn * 0.8));
      const staleContext = Math.floor(15000 * staleRatio * staleRatio + Math.random() * 2000);

      demoTurnSnapshots.push({
        turn: t,
        system: systemTokens,
        active: activeContext,
        stale: staleContext,
        total: systemTokens + activeContext + staleContext,
        window: contextWindow,
      });
    }

    // Build per-turn messages
    const roles = ["user", "assistant", "tool_use", "tool_result"];
    demoMessages = [];
    for (let t = 0; t <= maxTurn; t++) {
      const msgs = [];
      // User message
      msgs.push({
        role: "user",
        content: "User prompt for turn " + t + ": " + randomPhrase(),
        size: 50 + Math.floor(Math.random() * 200),
        stale: false,
      });
      // Assistant
      msgs.push({
        role: "assistant",
        content: "I'll help with that. Let me " + randomAction() + "...",
        size: 200 + Math.floor(Math.random() * 1500),
        stale: false,
      });
      // Possibly tool calls
      if (Math.random() < 0.7) {
        const tool = tools[Math.floor(Math.random() * tools.length)];
        const file = files[Math.floor(Math.random() * files.length)];
        msgs.push({
          role: "tool_use",
          content: tool + '("' + file + '")',
          size: 80 + Math.floor(Math.random() * 200),
          stale: false,
        });
        msgs.push({
          role: "tool_result",
          content: "File content from " + file + ":\n" + randomCode(),
          size: 500 + Math.floor(Math.random() * 5000),
          stale: t < currentTurn - 5,
        });
      }
      demoMessages.push(msgs);
    }
  }

  function randomPhrase() {
    const phrases = [
      "Can you refactor the parser module?",
      "Fix the failing test in test_api.py",
      "Add error handling to the server",
      "Update the configuration schema",
      "Review the type definitions",
      "Optimize the query performance",
      "Add logging to the pipeline",
      "Clean up unused imports",
    ];
    return phrases[Math.floor(Math.random() * phrases.length)];
  }

  function randomAction() {
    const actions = [
      "read the file", "check the tests", "analyze the code",
      "look at the implementation", "review the changes",
    ];
    return actions[Math.floor(Math.random() * actions.length)];
  }

  function randomCode() {
    return 'def process(data):\n    result = []\n    for item in data:\n        if item.is_valid():\n            result.append(transform(item))\n    return result';
  }

  // ---- Update scorecards ----
  function updateScorecards(data) {
    if (!data) {
      $scDeadweight.textContent = "—";
      $scContext.textContent = "—";
      $scCache.textContent = "—";
      $scTools.textContent = "—";
      return;
    }

    // TODO: Wire to real per-turn data when API returns staleness breakdown.
    // For now, derive approximate values from the snapshot data.
    const snap = demoTurnSnapshots[currentTurn] || demoTurnSnapshots[0];
    if (snap) {
      const deadWeightRatio = snap.stale / Math.max(1, snap.total);
      $scDeadweight.textContent = fmtPct(deadWeightRatio);
      $scDeadweightSub.textContent = fmtTokens(snap.stale) + " stale tokens";

      $scContext.textContent = fmtTokens(snap.total);
      $scContextSub.textContent = fmtPct(snap.total / snap.window) + " of " + fmtTokens(snap.window) + " window";
    }

    // TODO: Cache hit rate and tool calls need real API data.
    const cacheRate = 0.45 + Math.random() * 0.3;
    $scCache.textContent = fmtPct(cacheRate);
    $scCacheSub.textContent = fmtCost(0.02 * (currentTurn + 1)) + " session cost";

    $scTools.textContent = String(data.block_count || 0);
    $scToolsSub.textContent = "blocks in store";
  }

  // ---- Health dot ----
  function updateHealthDot() {
    // TODO: Wire to real urgency score from API.
    const snap = demoTurnSnapshots[currentTurn];
    if (!snap) {
      $healthDot.className = "health-dot";
      $healthLabel.textContent = "—";
      return;
    }
    const deadRatio = snap.stale / Math.max(1, snap.total);
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

    if (demoTurnSnapshots.length === 0) return;

    const labels = demoTurnSnapshots.map((s) => s.turn);
    const systemData = demoTurnSnapshots.map((s) => s.system);
    const activeData = demoTurnSnapshots.map((s) => s.active);
    const staleData = demoTurnSnapshots.map((s) => s.stale);

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
            ctx.fillText("T" + currentTurn, x, yTop - 4);
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
  function updateTape(turnNumber) {
    // TODO: Wire to real per-block staleness API when available.
    if (demoBlocks.length === 0) {
      $tapeRows.innerHTML = '<div class="tape-empty">No blocks for this session</div>';
      return;
    }

    const visibleBlocks = demoBlocks.filter((b) => b.entered <= turnNumber);

    let html = "";
    visibleBlocks.forEach((block) => {
      const age = turnNumber - block.entered;
      const barPct = Math.min(100, ((age + 1) / Math.max(1, maxTurn + 1)) * 100);
      let status = "active";
      if (block.pinned) {
        status = "pinned";
      } else if (age > maxTurn * 0.5) {
        status = "stale";
      } else if (age > maxTurn * 0.25) {
        status = "warm";
      }
      const barClass = block.pinned ? "pinned" : status === "stale" || status === "dead_weight" ? "stale" : "active";

      html += `<div class="tape-row">
        <div class="tape-block-label" title="${block.tool}: ${block.resource}">${block.tool}: ${block.resource}</div>
        <div class="tape-bar-container">
          <div class="tape-bar ${barClass}" style="width:${barPct}%"></div>
        </div>
        <div class="tape-size">${fmtTokens(block.size)}</div>
        <div class="tape-status ${status}">${status.replace("_", " ")}</div>
      </div>`;
    });

    $tapeRows.innerHTML = html;
  }

  // ---- Recommendations ----
  function updateRecommendations(turnNumber) {
    // TODO: Wire to real recommendations API when available.
    const staleBlocks = demoBlocks.filter(
      (b) => !b.pinned && b.entered <= turnNumber && turnNumber - b.entered > maxTurn * 0.5
    );

    const recoverable = staleBlocks.reduce((s, b) => s + b.size, 0);
    $recsRecoverable.textContent = fmtTokens(recoverable);

    if (staleBlocks.length === 0) {
      $recsList.innerHTML = '<div class="tape-empty">No stale blocks to reclaim</div>';
      return;
    }

    // Sort by size descending
    const sorted = [...staleBlocks].sort((a, b) => b.size - a.size).slice(0, 8);

    let html = "";
    sorted.forEach((block) => {
      const staleTurns = turnNumber - block.entered;
      html += `<div class="rec-item">
        <div class="rec-tool">${block.tool}: ${block.resource}</div>
        <div class="rec-meta">
          <span>entered T${block.entered}</span>
          <span>stale ${staleTurns} turns</span>
          <span>${fmtTokens(block.size)}</span>
        </div>
      </div>`;
    });

    $recsList.innerHTML = html;
  }

  // ---- Turn details ----
  function updateTurnDetails(turnNumber) {
    // TODO: Wire to real per-turn message API when available.
    const msgs = demoMessages[turnNumber];
    if (!msgs || msgs.length === 0) {
      $turnMessages.innerHTML = '<div class="tape-empty">No messages for this turn</div>';
      return;
    }

    let html = "";
    msgs.forEach((msg) => {
      const roleClass = msg.role.replace(" ", "_");
      const roleLabel = msg.role === "tool_use" ? "TOOL USE" : msg.role === "tool_result" ? "TOOL RESULT" : msg.role.toUpperCase();
      const preview = msg.content.length > 120 ? msg.content.slice(0, 120) + "..." : msg.content;

      html += `<div class="turn-msg">
        <span class="role-badge ${roleClass}">${roleLabel}</span>
        <span class="msg-preview">${escapeHtml(preview)}</span>
        <span class="msg-size">${fmtTokens(msg.size)}</span>
      </div>`;
    });

    $turnMessages.innerHTML = html;
  }

  // ---- Drilldown modal ----
  function openDrilldown(turnNumber) {
    $modalOverlay.classList.add("open");
    document.body.style.overflow = "hidden";
    renderDrilldown(turnNumber);
  }

  function closeDrilldown() {
    $modalOverlay.classList.remove("open");
    document.body.style.overflow = "";
  }

  function renderDrilldown(turnNumber) {
    const snap = demoTurnSnapshots[turnNumber];
    const msgs = demoMessages[turnNumber] || [];

    $modalTitle.textContent = "Turn " + turnNumber + (snap ? " \u2014 " + fmtTokens(snap.total) + " tokens" : "");

    // Pills
    const staleBlocks = demoBlocks.filter(
      (b) => !b.pinned && b.entered <= turnNumber && turnNumber - b.entered > maxTurn * 0.5
    );
    const pills = [
      { label: "Messages", value: msgs.length },
      { label: "Token delta", value: snap ? "+" + fmtTokens(Math.abs(snap.active - (demoTurnSnapshots[turnNumber - 1]?.active || 0))) : "—" },
      { label: "Stale blocks", value: staleBlocks.length },
    ];
    $modalPills.innerHTML = pills
      .map((p) => `<span class="pill">${p.label} <span class="pill-value">${p.value}</span></span>`)
      .join("");

    // Stats panels
    if (snap) {
      $modalStats.innerHTML = `
        <div class="stat-panel">
          <div class="stat-panel-title">Composition</div>
          <div class="stat-panel-rows">
            <div class="stat-row"><span class="stat-row-label">System</span><span class="stat-row-value">${fmtTokens(snap.system)}</span></div>
            <div class="stat-row"><span class="stat-row-label">Active</span><span class="stat-row-value">${fmtTokens(snap.active)}</span></div>
            <div class="stat-row"><span class="stat-row-label">Stale</span><span class="stat-row-value">${fmtTokens(snap.stale)}</span></div>
          </div>
        </div>
        <div class="stat-panel">
          <div class="stat-panel-title">This Turn</div>
          <div class="stat-panel-rows">
            <div class="stat-row"><span class="stat-row-label">Messages</span><span class="stat-row-value">${msgs.length}</span></div>
            <div class="stat-row"><span class="stat-row-label">Total tokens</span><span class="stat-row-value">${fmtTokens(snap.total)}</span></div>
            <div class="stat-row"><span class="stat-row-label">Utilization</span><span class="stat-row-value">${fmtPct(snap.total / snap.window)}</span></div>
          </div>
        </div>
        <div class="stat-panel">
          <div class="stat-panel-title">Staleness at Turn</div>
          <div class="stat-panel-rows">
            <div class="stat-row"><span class="stat-row-label">Dead weight</span><span class="stat-row-value">${fmtPct(snap.stale / Math.max(1, snap.total))}</span></div>
            <div class="stat-row"><span class="stat-row-label">Stale blocks</span><span class="stat-row-value">${staleBlocks.length}</span></div>
            <div class="stat-row"><span class="stat-row-label">Recoverable</span><span class="stat-row-value">${fmtTokens(staleBlocks.reduce((s, b) => s + b.size, 0))}</span></div>
          </div>
        </div>
        <div class="stat-panel">
          <div class="stat-panel-title">Cost</div>
          <div class="stat-panel-rows">
            <div class="stat-row"><span class="stat-row-label">This turn</span><span class="stat-row-value">${fmtCost(0.02)}</span></div>
            <div class="stat-row"><span class="stat-row-label">Cumulative</span><span class="stat-row-value">${fmtCost(0.02 * (turnNumber + 1))}</span></div>
            <div class="stat-row"><span class="stat-row-label">Wasted</span><span class="stat-row-value">${fmtCost(0.005 * staleBlocks.length)}</span></div>
          </div>
        </div>`;
    } else {
      $modalStats.innerHTML = "";
    }

    // Messages
    let msgHtml = "";
    msgs.forEach((msg) => {
      const roleClass = msg.role.replace(" ", "_");
      const roleLabel = msg.role === "tool_use" ? "TOOL USE" : msg.role === "tool_result" ? "TOOL RESULT" : msg.role.toUpperCase();
      const staleTag = msg.stale
        ? '<span class="modal-msg-tag stale-tag">STALE</span>'
        : '<span class="modal-msg-tag active-tag">ACTIVE</span>';
      const maxSize = 8000; // for bar scaling
      const barPct = Math.min(100, (msg.size / maxSize) * 100);

      msgHtml += `<div class="modal-msg">
        <div class="modal-msg-header">
          <span class="role-badge ${roleClass}">${roleLabel}</span>
          ${staleTag}
          <div class="modal-msg-size-bar"><div class="modal-msg-size-fill" style="width:${barPct}%"></div></div>
          <span class="modal-msg-size-text">${fmtTokens(msg.size)}</span>
        </div>
        <div class="modal-msg-body">${escapeHtml(msg.content)}</div>
      </div>`;
    });
    $modalMessages.innerHTML = msgHtml;

    // Stale blocks section
    if (staleBlocks.length > 0) {
      const showCount = Math.min(3, staleBlocks.length);
      const sorted = [...staleBlocks].sort((a, b) => b.size - a.size);
      let staleHtml = `<div class="modal-stale-header">Stale blocks (${staleBlocks.length})</div><div class="modal-stale-list">`;
      sorted.slice(0, showCount).forEach((b) => {
        staleHtml += `<div class="rec-item">
          <div class="rec-tool">${b.tool}: ${b.resource}</div>
          <div class="rec-meta"><span>${fmtTokens(b.size)}</span> <span>entered T${b.entered}</span></div>
        </div>`;
      });
      if (staleBlocks.length > showCount) {
        staleHtml += `<div class="modal-stale-header">Show ${staleBlocks.length - showCount} more &rarr;</div>`;
      }
      staleHtml += "</div>";
      $modalStale.innerHTML = staleHtml;
    } else {
      $modalStale.innerHTML = "";
    }

    // Nav buttons
    $modalPrev.disabled = turnNumber <= 0;
    $modalNext.disabled = turnNumber >= maxTurn;
  }

  // ---- Composite turn update ----
  function updateTurnView(turnNumber) {
    currentTurn = turnNumber;
    $turnSlider.value = turnNumber;
    $turnLabel.textContent = `Turn ${turnNumber} / ${maxTurn}`;

    updateScorecards(sessionData);
    updateTape(turnNumber);
    updateRecommendations(turnNumber);
    updateTurnDetails(turnNumber);
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
    // Restart if playing
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
    // Don't capture when typing in inputs
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
