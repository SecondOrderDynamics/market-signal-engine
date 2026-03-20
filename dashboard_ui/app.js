async function getJson(path) {
  const resp = await fetch(path, { cache: "no-store" });
  if (!resp.ok) throw new Error(`${path} -> ${resp.status}`);
  return resp.json();
}

function asText(value) {
  if (value === null || value === undefined || value === "") return "-";
  return String(value);
}

function asNum(value, decimals = 2) {
  const n = Number(value);
  if (value === null || value === undefined || value === "" || Number.isNaN(n)) return "-";
  return n.toFixed(decimals);
}

function asDollarVol(value) {
  const n = Number(value);
  if (Number.isNaN(n) || n === 0) return "-";
  if (n >= 1_000_000) return `$${(n / 1_000_000).toFixed(1)}M`;
  if (n >= 1_000) return `$${(n / 1_000).toFixed(0)}K`;
  return `$${n.toFixed(0)}`;
}

function asPercent(value) {
  if (value === null || value === undefined || value === "") return "-";
  const num = Number(value);
  if (Number.isNaN(num)) return asText(value);
  return `${(num * 100).toFixed(2)}%`;
}

// Colour a numeric score cell green→yellow→red across a given max.
function scoreClass(value, max) {
  const n = Number(value);
  if (Number.isNaN(n)) return "";
  const ratio = n / max;
  if (ratio >= 0.65) return "score-high";
  if (ratio >= 0.35) return "score-mid";
  return "score-low";
}

function tierBadge(tier) {
  const t = String(tier || "").toLowerCase();
  if (t === "nano") return `<span class="tier-badge tier-nano">Nano</span>`;
  if (t === "small") return `<span class="tier-badge tier-small">Small+</span>`;
  return "-";
}

function anomalyBadge(label) {
  const t = String(label || "").toLowerCase();
  if (t.includes("extreme")) return `<span class="anomaly-badge anomaly-extreme">${label}</span>`;
  if (t.includes("anomalous")) return `<span class="anomaly-badge anomaly-high">${label}</span>`;
  if (t.includes("elevated")) return `<span class="anomaly-badge anomaly-mid">${label}</span>`;
  if (t.includes("within")) return `<span class="anomaly-badge anomaly-low">${label}</span>`;
  return `<span class="anomaly-badge">${asText(label)}</span>`;
}

// Render a watchlist table (small or nano).
// cols: array of { key, label, render } descriptors.
function renderWatchlist(tbody, rows) {
  tbody.innerHTML = "";

  if (!rows || rows.length === 0) {
    const tr = document.createElement("tr");
    const td = document.createElement("td");
    td.colSpan = 13;
    td.textContent = "No rows";
    tr.appendChild(td);
    tbody.appendChild(tr);
    return;
  }

  for (const row of rows) {
    const tr = document.createElement("tr");

    const cells = [
      { html: `<strong>${asText(row.ticker)}</strong>` },
      { text: `$${asNum(row.price)}`, cls: "" },
      { text: asNum(row.rel_volume, 1) + "x", cls: scoreClass(row.rel_volume, 5) },
      { text: asDollarVol(row.dollar_volume) },
      { text: asNum(row.total_score), cls: scoreClass(row.total_score, 15) },
      { text: asNum(row.event_intel_score), cls: scoreClass(row.event_intel_score, 10) },
      { text: asNum(row.anomaly_score), cls: scoreClass(row.anomaly_score, 10) },
      { html: anomalyBadge(row.anomaly_label) },
      { text: asNum(row.short_interest_score), cls: scoreClass(row.short_interest_score, 3) },
      { text: asNum(row.breakout_score), cls: scoreClass(row.breakout_score, 3) },
      { text: asNum(row.trend_score), cls: scoreClass(row.trend_score, 3) },
      { text: asText(row.repeat_count_30d) },
      { text: asText(row.reason), cls: "reason-cell" },
    ];

    for (const cell of cells) {
      const td = document.createElement("td");
      if (cell.html !== undefined) {
        td.innerHTML = cell.html;
      } else {
        td.textContent = cell.text ?? "-";
      }
      if (cell.cls) td.className = cell.cls;
      tr.appendChild(td);
    }
    tbody.appendChild(tr);
  }
}

function renderTopCandidates(tbody, rows) {
  tbody.innerHTML = "";
  if (!rows || rows.length === 0) {
    const tr = document.createElement("tr");
    const td = document.createElement("td");
    td.colSpan = 6;
    td.textContent = "No rows";
    tr.appendChild(td);
    tbody.appendChild(tr);
    return;
  }
  for (const row of rows) {
    const tr = document.createElement("tr");
    const cells = [
      { text: asText(row.ticker) },
      { html: tierBadge(row.cap_tier) },
      { text: asNum(row.total_score), cls: scoreClass(row.total_score, 15) },
      { text: asNum(row.event_intel_score), cls: scoreClass(row.event_intel_score, 10) },
      { text: asNum(row.anomaly_score), cls: scoreClass(row.anomaly_score, 10) },
      { html: anomalyBadge(row.anomaly_label) },
    ];
    for (const cell of cells) {
      const td = document.createElement("td");
      if (cell.html !== undefined) td.innerHTML = cell.html;
      else td.textContent = cell.text ?? "-";
      if (cell.cls) td.className = cell.cls;
      tr.appendChild(td);
    }
    tbody.appendChild(tr);
  }
}

function renderRecentRuns(tbody, runs) {
  tbody.innerHTML = "";
  if (!runs || runs.length === 0) {
    const tr = document.createElement("tr");
    const td = document.createElement("td");
    td.colSpan = 5;
    td.textContent = "No runs";
    tr.appendChild(td);
    tbody.appendChild(tr);
    return;
  }
  for (const x of runs) {
    const tr = document.createElement("tr");
    const statusCls = x.success ? "status-ok" : "status-fail";
    [
      { text: asText(x.run_id) },
      { html: `<span class="${statusCls}">${asText(x.status)}</span>` },
      { text: asText(x.timestamp_utc) },
      { text: asText(x.explosive_found) },
      { text: asText(x.eligible) },
    ].forEach(({ text, html }) => {
      const td = document.createElement("td");
      if (html !== undefined) td.innerHTML = html;
      else td.textContent = text ?? "-";
      tr.appendChild(td);
    });
    tbody.appendChild(tr);
  }
}

function setStatusChip(status, success) {
  const chip = document.getElementById("statusChip");
  chip.className = "chip";
  if (success === true || status === "success") {
    chip.classList.add("chip-success");
    chip.textContent = "Success";
    return;
  }
  if (status === "failed" || success === false) {
    chip.classList.add("chip-failed");
    chip.textContent = "Failed";
    return;
  }
  chip.classList.add("chip-neutral");
  chip.textContent = asText(status || "Unknown");
}

function renderEvalStats(payload) {
  const root = document.getElementById("evalStats");
  const rows = [
    ["Rows in evaluation output", asText(payload.rows)],
    ["Eligible rows", asText(payload.eligible_rows)],
    ["Skipped rows", asText(payload.skipped_rows)],
    ["Mean fwd 5d close", asPercent(payload.mean_fwd5d_close)],
    ["Median fwd 5d close", asPercent(payload.median_fwd5d_close)],
  ];
  root.innerHTML = "";
  for (const [k, v] of rows) {
    const dt = document.createElement("dt");
    dt.textContent = k;
    const dd = document.createElement("dd");
    dd.textContent = v;
    root.appendChild(dt);
    root.appendChild(dd);
  }
}

async function loadDashboard() {
  try {
    const [latest, runsRes, evalRes, mergedRes] = await Promise.all([
      getJson("/api/latest-run"),
      getJson("/api/runs?limit=12"),
      getJson("/api/evaluation"),
      getJson("/api/merged-watchlist?limit=100"),
    ]);

    const counts = latest.candidate_counts || {};
    const evalCounts = latest.evaluation_counts || {};
    const allRows = mergedRes.rows || [];

    const smallRows = allRows.filter(r => !r.cap_tier || r.cap_tier === "small");
    const nanoRows  = allRows.filter(r => r.cap_tier === "nano");

    document.getElementById("smallFound").textContent = String(smallRows.length);
    document.getElementById("nanoFound").textContent  = String(nanoRows.length);
    document.getElementById("mergedTotal").textContent = asText(counts.merged_total || allRows.length);
    document.getElementById("eligibleRows").textContent = asText(evalCounts.eligible);
    document.getElementById("skippedRows").textContent  = asText(evalCounts.skipped_insufficient_forward);

    const ts    = asText(latest.timestamp_utc);
    const runId = asText(latest.run_id);
    document.getElementById("subhead").textContent = `Latest run ${runId} at ${ts}`;
    setStatusChip(latest.status, latest.success);

    renderWatchlist(document.querySelector("#smallTable tbody"), smallRows);
    renderWatchlist(document.querySelector("#nanoTable tbody"), nanoRows);

    renderTopCandidates(
      document.querySelector("#topCandidatesTable tbody"),
      latest.top_candidates || []
    );
    renderRecentRuns(
      document.querySelector("#recentRunsTable tbody"),
      (runsRes.runs || []).map(x => ({
        run_id: x.run_id,
        status: x.status,
        success: x.success,
        timestamp_utc: x.timestamp_utc,
        explosive_found: x.explosive_found,
        eligible: x.eligible,
      }))
    );
    renderEvalStats(evalRes || {});

  } catch (err) {
    document.getElementById("subhead").textContent = `Failed to load dashboard data: ${err.message}`;
    setStatusChip("failed", false);
  }
}

document.getElementById("refreshBtn").addEventListener("click", loadDashboard);
loadDashboard();
setInterval(loadDashboard, 30000);
