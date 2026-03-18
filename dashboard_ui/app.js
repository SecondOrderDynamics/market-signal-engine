async function getJson(path) {
  const resp = await fetch(path, { cache: "no-store" });
  if (!resp.ok) {
    throw new Error(`${path} -> ${resp.status}`);
  }
  return resp.json();
}

function asText(value) {
  if (value === null || value === undefined || value === "") return "-";
  return String(value);
}

function asPercent(value) {
  if (value === null || value === undefined || value === "") return "-";
  const num = Number(value);
  if (Number.isNaN(num)) return asText(value);
  return `${(num * 100).toFixed(2)}%`;
}

function renderRows(tbody, rows, columns) {
  tbody.innerHTML = "";
  if (!rows || rows.length === 0) {
    const tr = document.createElement("tr");
    const td = document.createElement("td");
    td.colSpan = columns.length;
    td.textContent = "No rows";
    tr.appendChild(td);
    tbody.appendChild(tr);
    return;
  }
  for (const row of rows) {
    const tr = document.createElement("tr");
    for (const col of columns) {
      const td = document.createElement("td");
      const value = row[col];
      td.textContent = asText(value);
      tr.appendChild(td);
    }
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
      getJson("/api/merged-watchlist?limit=25"),
    ]);

    const counts = latest.candidate_counts || {};
    const evalCounts = latest.evaluation_counts || {};

    document.getElementById("explosiveFound").textContent = asText(counts.explosive_found);
    document.getElementById("mergedTotal").textContent = asText(counts.merged_total);
    document.getElementById("eligibleRows").textContent = asText(evalCounts.eligible);
    document.getElementById("skippedRows").textContent = asText(evalCounts.skipped_insufficient_forward);

    const ts = asText(latest.timestamp_utc);
    const runId = asText(latest.run_id);
    document.getElementById("subhead").textContent = `Latest run ${runId} at ${ts}`;
    setStatusChip(latest.status, latest.success);

    renderRows(
      document.querySelector("#topCandidatesTable tbody"),
      latest.top_candidates || [],
      ["ticker", "total_score", "event_intel_score", "anomaly_score", "anomaly_label"]
    );
    renderRows(
      document.querySelector("#recentRunsTable tbody"),
      (runsRes.runs || []).map((x) => ({
        run_id: x.run_id,
        status: x.status,
        timestamp_utc: x.timestamp_utc,
        explosive_found: x.explosive_found,
        eligible: x.eligible,
      })),
      ["run_id", "status", "timestamp_utc", "explosive_found", "eligible"]
    );
    renderEvalStats(evalRes || {});
    renderRows(
      document.querySelector("#mergedTable tbody"),
      mergedRes.rows || [],
      ["ticker", "total_score", "event_intel_score", "anomaly_score", "anomaly_label"]
    );
  } catch (err) {
    document.getElementById("subhead").textContent = `Failed to load dashboard data: ${err.message}`;
    setStatusChip("failed", false);
  }
}

document.getElementById("refreshBtn").addEventListener("click", loadDashboard);
loadDashboard();
setInterval(loadDashboard, 30000);
