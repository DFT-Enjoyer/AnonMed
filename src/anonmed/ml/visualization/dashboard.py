# ruff: noqa: E501
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


def build_dashboard_manifest(
    instance_root: str | Path,
    *,
    include_samples: bool = False,
    sample_limit: int = 200,
) -> dict[str, Any]:
    root = Path(instance_root)
    warnings: list[str] = []
    runs: list[dict[str, Any]] = []
    for report_path in sorted(root.glob("*/*/report.json")):
        run_dir = report_path.parent
        run_name = run_dir.parent.name
        timestamp = run_dir.name
        report = _load_json(report_path, warnings)
        if report is None:
            continue
        runs.append(
            _run_record(
                root=root,
                run_name=run_name,
                timestamp=timestamp,
                run_dir=run_dir,
                report_path=report_path,
                report=report,
                include_samples=include_samples,
                sample_limit=sample_limit,
                warnings=warnings,
            )
        )

    runs.sort(key=lambda run: (run["timestamp"], run["run_name"]), reverse=True)
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "instance_root": str(root),
        "include_samples": include_samples,
        "sample_limit": sample_limit,
        "warnings": warnings,
        "runs": runs,
    }


def write_dashboard(
    manifest: Mapping[str, Any],
    output_path: str | Path,
) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest_json = json.dumps(manifest, ensure_ascii=False).replace("</", "<\\/")
    html_text = _HTML_TEMPLATE.replace("__ANONMED_DASHBOARD_MANIFEST__", manifest_json)
    output.write_text(html_text, encoding="utf-8")
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a static AnonMed ML runs dashboard.")
    parser.add_argument("--instance-root", default="instance", help="Root directory with run outputs.")
    parser.add_argument(
        "--output",
        default="instance/dashboard.html",
        help="Path to the generated self-contained HTML dashboard.",
    )
    parser.add_argument(
        "--include-samples",
        action="store_true",
        help="Embed raw sample text from snapshots into the dashboard.",
    )
    parser.add_argument(
        "--sample-limit",
        type=int,
        default=200,
        help="Maximum samples embedded per run when --include-samples is used.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    manifest = build_dashboard_manifest(
        args.instance_root,
        include_samples=args.include_samples,
        sample_limit=args.sample_limit,
    )
    output = write_dashboard(manifest, args.output)
    print(f"dashboard: {output}")
    print(f"runs: {len(manifest['runs'])}")
    if manifest["warnings"]:
        print(f"warnings: {len(manifest['warnings'])}")
    return 0


def _load_json(path: Path, warnings: list[str]) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        warnings.append(f"Cannot read {path}: {error}")
        return None
    if not isinstance(payload, dict):
        warnings.append(f"Skipping {path}: JSON root must be an object")
        return None
    return payload


def _run_record(
    *,
    root: Path,
    run_name: str,
    timestamp: str,
    run_dir: Path,
    report_path: Path,
    report: Mapping[str, Any],
    include_samples: bool,
    sample_limit: int,
    warnings: list[str],
) -> dict[str, Any]:
    metrics = _normalize_metrics(report)
    artifacts = _artifact_paths(run_dir, report)
    sample_summary = _snapshot_summary(artifacts.get("dataset_snapshot_json"), warnings)
    samples = []
    if include_samples:
        samples = _snapshot_samples(
            artifacts.get("evaluation_snapshot_json") or artifacts.get("dataset_snapshot_json"),
            sample_limit=sample_limit,
            warnings=warnings,
        )
    return {
        "id": f"{run_name}/{timestamp}",
        "run_name": run_name,
        "timestamp": timestamp,
        "run_dir": _relative_path(root, run_dir),
        "report_path": _relative_path(root, report_path),
        "samples_count": int(report.get("samples_count", sample_summary.get("samples_count", 0))),
        "run": _mapping(report.get("run")),
        "dataset": _mapping(report.get("dataset")),
        "model": _mapping(report.get("model")),
        "training": _mapping(report.get("training")),
        "evaluation": _mapping(report.get("evaluation")),
        "metric_configs": _list(report.get("metrics")),
        "metrics": metrics,
        "artifacts": artifacts,
        "sample_summary": sample_summary,
        "samples": samples,
        "samples_truncated": include_samples and len(samples) >= sample_limit,
    }


def _normalize_metrics(report: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw_metrics = report.get("metric_results", report.get("metrics", {}))
    if not isinstance(raw_metrics, Mapping):
        return []
    normalized = []
    for name, raw_value in sorted(raw_metrics.items()):
        payload = _mapping(raw_value)
        value = payload.get("value")
        if value is None and name == "coverage_percent":
            value = payload.get("coverage_percent")
        metric = {
            "name": str(name),
            "value": _number_or_none(value),
            "tp": _int_or_none(payload.get("tp")),
            "fp": _int_or_none(payload.get("fp")),
            "fn": _int_or_none(payload.get("fn")),
            "threshold": _number_or_none(payload.get("threshold")),
            "fields": payload,
        }
        normalized.append(metric)
    return normalized


def _artifact_paths(run_dir: Path, report: Mapping[str, Any]) -> dict[str, str]:
    artifacts = {}
    instance_payload = _mapping(report.get("instance"))
    for key, value in instance_payload.items():
        if isinstance(value, str):
            artifacts[key] = value
    for key, filename in (
        ("dataset_snapshot_json", "dataset_snapshot.json"),
        ("evaluation_snapshot_json", "evaluation_snapshot.json"),
        ("dataset_snapshot_parquet", "dataset_snapshot.parquet"),
    ):
        path = run_dir / filename
        if path.exists():
            artifacts[key] = str(path)
    return artifacts


def _snapshot_summary(path_text: str | None, warnings: list[str]) -> dict[str, Any]:
    if not path_text:
        return {"available": False, "samples_count": 0, "target_span_counts": {}}
    path = Path(path_text)
    payload = _load_json(path, warnings)
    if payload is None:
        return {"available": False, "samples_count": 0, "target_span_counts": {}}
    counts: dict[str, int] = {}
    cases = payload.get("cases", [])
    if isinstance(cases, list):
        for case in cases:
            for span in _target_spans(case):
                label = str(span.get("label", "unknown"))
                counts[label] = counts.get(label, 0) + 1
    return {
        "available": True,
        "samples_count": int(payload.get("samples_count", len(cases) if isinstance(cases, list) else 0)),
        "target_span_counts": counts,
    }


def _snapshot_samples(
    path_text: str | None,
    *,
    sample_limit: int,
    warnings: list[str],
) -> list[dict[str, Any]]:
    if not path_text or sample_limit <= 0:
        return []
    payload = _load_json(Path(path_text), warnings)
    if payload is None:
        return []
    cases = payload.get("cases", [])
    if not isinstance(cases, list):
        return []
    samples = []
    for case in cases[:sample_limit]:
        document = _mapping(case.get("document")) if isinstance(case, Mapping) else {}
        lines = _list(document.get("lines"))
        text = "\n".join(str(_mapping(line).get("text", "")) for line in lines)
        samples.append(
            {
                "sample_id": _mapping(case).get("sample_id") or document.get("sample_id"),
                "text": text,
                "target_spans": _target_spans(case),
                "prediction_spans": _prediction_spans(case),
            }
        )
    return samples


def _target_spans(case: object) -> list[dict[str, Any]]:
    return _annotation_spans(_mapping(case).get("target"))


def _prediction_spans(case: object) -> list[dict[str, Any]]:
    return _annotation_spans(_mapping(case).get("prediction"))


def _annotation_spans(annotation: object) -> list[dict[str, Any]]:
    spans = []
    for line in _list(_mapping(annotation).get("lines")):
        for span in _list(_mapping(line).get("spans")):
            if isinstance(span, Mapping):
                spans.append(dict(span))
    return spans


def _mapping(value: object) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _list(value: object) -> list[Any]:
    return list(value) if isinstance(value, list) else []


def _number_or_none(value: object) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _int_or_none(value: object) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _relative_path(root: Path, path: Path) -> str:
    try:
        return str(path.relative_to(root.parent))
    except ValueError:
        return str(path)


_HTML_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>AnonMed ML Runs</title>
  <style>
    :root {
      color-scheme: dark;
      --bg:#0b0d10; --panel:#11161d; --panel-2:#151b23; --card:#171e27;
      --ink:#e7ebf2; --muted:#9aa6b7; --faint:#6f7b8b; --line:#2a3442;
      --accent:#7db7ff; --accent-2:#b2d4ff; --tp:#25d095; --fp:#ff5c8a; --fn:#ffb14a;
      --f1:#8ea7ff; --precision:#ff70b8; --recall:#ffca64; --coverage:#a78bfa;
      --grid:#253140; --shadow:0 12px 32px rgba(0,0,0,.28);
    }
    * { box-sizing: border-box; }
    body { margin:0; font:14px/1.45 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; background:var(--bg); color:var(--ink); }
    header { padding:20px 28px; border-bottom:1px solid var(--line); background:#0f1319; }
    h1 { margin:0 0 4px; font-size:24px; letter-spacing:0; }
    h2 { margin:22px 0 10px; font-size:18px; letter-spacing:0; }
    h3 { margin:0; font-size:15px; letter-spacing:0; }
    main { padding:20px 28px 36px; }
    .muted { color:var(--muted); }
    .controls { display:grid; grid-template-columns: repeat(4, minmax(180px, 1fr)); gap:12px; margin:16px 0; }
    input, select, button { width:100%; padding:8px 10px; border:1px solid var(--line); border-radius:6px; background:#111820; color:var(--ink); }
    input::placeholder { color:var(--faint); }
    button { cursor:pointer; }
    .cards { display:grid; grid-template-columns: repeat(5, minmax(120px, 1fr)); gap:12px; }
    .card, .panel, .chart-panel { background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:14px; box-shadow:var(--shadow); }
    .chart-panel { position:relative; padding-bottom:54px; }
    .card .value { font-size:20px; font-weight:700; margin-top:4px; overflow-wrap:anywhere; }
    table { width:100%; border-collapse:collapse; background:var(--panel); border:1px solid var(--line); }
    th, td { padding:8px 10px; border-bottom:1px solid var(--line); text-align:left; vertical-align:top; }
    th { background:#151c25; position:sticky; top:0; z-index:1; color:#c9d4e5; }
    tr.selected { background:#17283a; }
    tbody tr:hover { background:#141b24; }
    .grid { display:grid; grid-template-columns: 1fr 1fr; gap:14px; }
    svg { width:100%; min-height:320px; background:#0f141b; border:1px solid var(--line); border-radius:8px; }
    .chart-fullscreen-button {
      position:absolute; right:22px; bottom:18px; z-index:3; width:auto;
      padding:6px 10px; background:#1a2430; border-color:#3d4d60; color:#e7ebf2;
    }
    .chart-fullscreen-button:hover { background:#253244; }
    .chart-head { display:flex; justify-content:space-between; gap:12px; align-items:flex-start; margin-bottom:8px; }
    .chart-note { color:var(--muted); font-size:12px; max-width:420px; }
    .tabs { display:flex; gap:8px; margin:18px 0 10px; }
    .tabs button { width:auto; }
    .hidden { display:none; }
    .sample { border:1px solid var(--line); border-radius:8px; padding:10px; margin:8px 0; background:var(--panel); }
    .pill { display:inline-block; padding:2px 7px; border-radius:999px; background:#202a36; border:1px solid #334254; margin:1px; color:#d9e3f1; }
    pre { white-space:pre-wrap; color:#dbe4f0; }
    .axis { stroke:#778397; stroke-width:1; }
    .grid-line { stroke:var(--grid); stroke-width:1; }
    .tick, .axis-label { fill:#aeb9ca; font-size:12px; }
    .bar-value { fill:#ecf2fb; font-size:12px; font-weight:700; }
    .legend-text { fill:#c7d0de; font-size:12px; }
    .empty { fill:#8390a3; font-size:13px; }
    .chart-fullscreen {
      position:fixed; inset:0; z-index:30; padding:22px; background:rgba(0,0,0,.78);
      backdrop-filter:blur(8px);
    }
    .fullscreen-card {
      width:100%; height:100%; display:flex; flex-direction:column; gap:12px;
      border:1px solid #3a485a; border-radius:10px; background:#0b0f15; box-shadow:0 22px 60px rgba(0,0,0,.55);
    }
    .fullscreen-top {
      display:flex; align-items:center; justify-content:space-between; gap:12px;
      padding:12px 14px; border-bottom:1px solid var(--line);
    }
    .fullscreen-title { font-size:16px; font-weight:700; }
    .fullscreen-close { width:auto; padding:7px 12px; }
    .fullscreen-stage { flex:1; min-height:0; padding:14px; }
    .fullscreen-stage svg { width:100%; height:100%; min-height:0; display:block; }
    .tooltip {
      position:fixed; z-index:50; max-width:420px; padding:8px 10px;
      border:1px solid #3a485a; border-radius:6px; background:#05070a;
      color:#f1f5fb; box-shadow:0 10px 28px rgba(0,0,0,.45);
      pointer-events:none; white-space:pre-wrap; font-size:12px;
    }
    @media (max-width: 900px) { .controls, .cards, .grid { grid-template-columns:1fr; } }
  </style>
</head>
<body>
<header>
  <h1>AnonMed ML Runs</h1>
  <div class="muted" id="subtitle"></div>
</header>
<main>
  <section class="controls">
    <input id="search" placeholder="Filter run/model/dataset">
    <select id="metric"></select>
    <select id="runName"></select>
    <button id="selectVisible">Select visible runs</button>
  </section>
  <section class="cards" id="overview"></section>
  <section>
    <h2>Runs</h2>
    <div style="max-height:360px; overflow:auto"><table id="runsTable"></table></div>
  </section>
  <section>
    <h2>Metrics Comparison</h2>
    <div class="grid">
      <div class="chart-panel">
        <div class="chart-head"><h3>Quality Profile</h3><div class="chart-note">Grouped F1 / precision / recall for selected or visible runs.</div></div>
        <svg id="profileChart"></svg>
        <button class="chart-fullscreen-button" data-fullscreen-target="profileChart" data-fullscreen-title="Quality Profile">Full screen</button>
      </div>
      <div class="chart-panel">
        <div class="chart-head"><h3>Selected Metric</h3><div class="chart-note">Single metric comparison, sorted by the current table/filter order.</div></div>
        <svg id="metricChart"></svg>
        <button class="chart-fullscreen-button" data-fullscreen-target="metricChart" data-fullscreen-title="Selected Metric">Full screen</button>
      </div>
      <div class="chart-panel">
        <div class="chart-head"><h3>TP / FP / FN</h3><div class="chart-note">Stacked counts for the selected metric. Lower FP/FN is better.</div></div>
        <svg id="countChart"></svg>
        <button class="chart-fullscreen-button" data-fullscreen-target="countChart" data-fullscreen-title="TP / FP / FN">Full screen</button>
      </div>
      <div class="chart-panel">
        <div class="chart-head"><h3>Precision vs Recall</h3><div class="chart-note">Top-right is best. Point numbers follow the selected run order in the table.</div></div>
        <svg id="scatterChart"></svg>
        <button class="chart-fullscreen-button" data-fullscreen-target="scatterChart" data-fullscreen-title="Precision vs Recall">Full screen</button>
      </div>
      <div class="panel"><h3>Run Detail</h3><div id="detail"></div></div>
    </div>
  </section>
  <section id="samplesSection" class="hidden">
    <h2>Samples</h2>
    <div id="samples"></div>
  </section>
</main>
<div id="tooltip" class="tooltip hidden"></div>
<div id="chartFullscreen" class="chart-fullscreen hidden" role="dialog" aria-modal="true" aria-labelledby="chartFullscreenTitle">
  <div class="fullscreen-card">
    <div class="fullscreen-top">
      <div id="chartFullscreenTitle" class="fullscreen-title"></div>
      <button id="chartFullscreenClose" class="fullscreen-close">Close</button>
    </div>
    <div id="chartFullscreenStage" class="fullscreen-stage"></div>
  </div>
</div>
<script>
const manifest = __ANONMED_DASHBOARD_MANIFEST__;
let state = { search: "", metric: "", runName: "", selected: new Set() };
const colors = {
  metric: "#7db7ff",
  f1: "#8ea7ff",
  precision: "#ff70b8",
  recall: "#ffca64",
  coverage: "#a78bfa",
  tp: "#25d095",
  fp: "#ff5c8a",
  fn: "#ffb14a",
  axis: "#778397",
  grid: "#253140",
  text: "#c7d0de",
  muted: "#8f9bad"
};

function metricValue(run, name) {
  const metric = run.metrics.find(m => m.name === name);
  return metric ? metric.value : null;
}
function metricObj(run, name) { return run.metrics.find(m => m.name === name) || {}; }
function fmt(value) { return value === null || value === undefined || Number.isNaN(Number(value)) ? "" : Number(value).toFixed(4); }
function allMetrics() { return [...new Set(manifest.runs.flatMap(r => r.metrics.map(m => m.name)))].sort(); }
function filteredRuns() {
  return manifest.runs.filter(run => {
    const hay = [run.id, run.model.id, run.dataset.id, run.timestamp].join(" ").toLowerCase();
    return (!state.search || hay.includes(state.search.toLowerCase())) &&
      (!state.runName || run.run_name === state.runName);
  });
}
function selectedRuns() {
  const visible = filteredRuns();
  const selected = visible.filter(run => state.selected.has(run.id));
  return selected.length ? selected : visible.slice(0, 8);
}
function keyMetric(run, token) {
  return run.metrics.find(m => m.name.toLowerCase().includes(token) && m.value !== null) || null;
}
function scoreValue(metric) {
  const value = Number(metric?.value);
  if (!Number.isFinite(value)) return null;
  return value > 1 && value <= 100 ? value / 100 : value;
}
function runLabel(run) {
  const stamp = String(run.timestamp || "").replace(/^\d{4}-/, "").replace(/_\d{6}$/, "");
  return `${run.run_name}:${stamp}`;
}
function shortLabel(text, max=18) {
  text = String(text || "");
  return text.length > max ? `${text.slice(0, max - 1)}…` : text;
}
function escapeHtml(s) { return String(s).replace(/[&<>"']/g, ch => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#039;'}[ch])); }
function tooltipAttr(text) { return `data-tooltip="${escapeHtml(text)}"`; }
function showTooltip(text, event) {
  const tooltip = document.getElementById("tooltip");
  tooltip.textContent = text;
  tooltip.classList.remove("hidden");
  moveTooltip(event);
}
function moveTooltip(event) {
  const tooltip = document.getElementById("tooltip");
  const gap = 14;
  const rect = tooltip.getBoundingClientRect();
  let left = event.clientX + gap;
  let top = event.clientY + gap;
  if (left + rect.width > window.innerWidth - 8) left = event.clientX - rect.width - gap;
  if (top + rect.height > window.innerHeight - 8) top = event.clientY - rect.height - gap;
  tooltip.style.left = `${Math.max(8, left)}px`;
  tooltip.style.top = `${Math.max(8, top)}px`;
}
function hideTooltip() {
  document.getElementById("tooltip").classList.add("hidden");
}
function openFullscreenChart(sourceId, title) {
  const source = document.getElementById(sourceId);
  if (!source) return;
  hideTooltip();
  const overlay = document.getElementById("chartFullscreen");
  const stage = document.getElementById("chartFullscreenStage");
  const clone = source.cloneNode(true);
  clone.removeAttribute("id");
  stage.replaceChildren(clone);
  document.getElementById("chartFullscreenTitle").textContent = title;
  overlay.classList.remove("hidden");
  document.body.style.overflow = "hidden";
}
function closeFullscreenChart() {
  hideTooltip();
  document.getElementById("chartFullscreen").classList.add("hidden");
  document.getElementById("chartFullscreenStage").replaceChildren();
  document.body.style.overflow = "";
}
function renderControls() {
  document.getElementById("subtitle").textContent = `${manifest.runs.length} runs from ${manifest.instance_root}; generated ${manifest.generated_at}`;
  const metrics = allMetrics();
  if (!state.metric) state.metric = metrics.find(m => m.includes("f1")) || metrics[0] || "";
  document.getElementById("metric").innerHTML = metrics.map(m => `<option ${m === state.metric ? "selected" : ""}>${m}</option>`).join("");
  const runNames = [...new Set(manifest.runs.map(r => r.run_name))].sort();
  document.getElementById("runName").innerHTML = `<option value="">All run names</option>` + runNames.map(n => `<option ${n === state.runName ? "selected" : ""}>${n}</option>`).join("");
}
function best(metricPart) {
  const pairs = manifest.runs.map(r => [r, keyMetric(r, metricPart)]).filter(x => x[1] && x[1].value !== null);
  pairs.sort((a,b) => b[1].value - a[1].value);
  return pairs[0] || null;
}
function renderOverview() {
  const f1 = best("f1"), precision = best("precision"), recall = best("recall");
  const latest = manifest.runs[0];
  const cards = [
    ["Runs", manifest.runs.length],
    ["Best F1", f1 ? `${fmt(f1[1].value)} · ${f1[0].id}` : ""],
    ["Best Precision", precision ? `${fmt(precision[1].value)} · ${precision[0].id}` : ""],
    ["Best Recall", recall ? `${fmt(recall[1].value)} · ${recall[0].id}` : ""],
    ["Latest", latest ? latest.id : ""],
  ];
  document.getElementById("overview").innerHTML = cards.map(([k,v]) => `<div class="card"><div class="muted">${k}</div><div class="value">${v}</div></div>`).join("");
}
function renderTable() {
  const runs = filteredRuns();
  const metrics = allMetrics().filter(m => /f1|precision|recall|coverage/.test(m));
  const headers = ["select", "run", "timestamp", "model", "dataset", "samples", ...metrics];
  const rows = runs.map(run => `<tr class="${state.selected.has(run.id) ? "selected" : ""}">
    <td><input type="checkbox" data-run="${run.id}" ${state.selected.has(run.id) ? "checked" : ""}></td>
    <td>${run.run_name}</td><td>${run.timestamp}</td><td>${run.model.id || ""}</td><td>${run.dataset.id || ""}</td><td>${run.samples_count}</td>
    ${metrics.map(m => `<td>${fmt(metricValue(run, m))}</td>`).join("")}
  </tr>`).join("");
  document.getElementById("runsTable").innerHTML = `<thead><tr>${headers.map(h => `<th>${h}</th>`).join("")}</tr></thead><tbody>${rows}</tbody>`;
  document.querySelectorAll("[data-run]").forEach(box => box.addEventListener("change", e => {
    const id = e.target.getAttribute("data-run");
    e.target.checked ? state.selected.add(id) : state.selected.delete(id);
    render();
  }));
}

function tickValues(yMax) {
  if (yMax <= 1) return [0, .25, .5, .75, 1];
  const niceMax = Math.ceil(yMax);
  return [0, niceMax * .25, niceMax * .5, niceMax * .75, niceMax];
}
function chartFrame(svgId, rows, yMax=1, yLabel="score") {
  const svg = document.getElementById(svgId), w = 820, h = 340;
  const pad = { left: 74, right: 22, top: 44, bottom: 36 };
  svg.setAttribute("viewBox", `0 0 ${w} ${h}`);
  const plotW = w - pad.left - pad.right, plotH = h - pad.top - pad.bottom;
  const ticks = tickValues(yMax);
  const tickHtml = ticks.map(t => {
    const y = pad.top + plotH - plotH * t / yMax;
    return `<line class="grid-line" x1="${pad.left}" y1="${y}" x2="${w-pad.right}" y2="${y}"/><text class="tick" x="${pad.left-10}" y="${y+4}" text-anchor="end">${fmt(t)}</text>`;
  }).join("");
  const axisX = 18;
  const axes = `${tickHtml}<line class="axis" x1="${pad.left}" y1="${pad.top}" x2="${pad.left}" y2="${h-pad.bottom}"/><line class="axis" x1="${pad.left}" y1="${h-pad.bottom}" x2="${w-pad.right}" y2="${h-pad.bottom}"/><text class="axis-label" x="${axisX}" y="${pad.top + plotH/2}" transform="rotate(-90 ${axisX} ${pad.top + plotH/2})">${yLabel}</text>`;
  return { svg, w, h, pad, plotW, plotH, base: axes };
}
function legend(items, x=58, y=18) {
  return items.map((item, i) => {
    const offset = i * 122;
    return `<rect x="${x+offset}" y="${y}" width="10" height="10" rx="2" fill="${item.color}"/><text class="legend-text" x="${x+offset+16}" y="${y+10}">${item.label}</text>`;
  }).join("");
}
function noData(svg, message) {
  svg.innerHTML = `<text class="empty" x="50%" y="50%" dominant-baseline="middle" text-anchor="middle">${message}</text>`;
}
function barChart(svgId, rows, valueFn, color, legendLabel) {
  const values = rows.map(valueFn).map(v => Number(v)).filter(Number.isFinite);
  const yMax = Math.max(1, ...values);
  const frame = chartFrame(svgId, rows, yMax, "value");
  const { svg, w, h, pad, plotW, plotH, base } = frame;
  if (!rows.length) { noData(svg, "No runs selected"); return; }
  const bw = Math.max(10, plotW / Math.max(1, rows.length) - 16);
  svg.innerHTML = base + legend([{ label: legendLabel, color }]) + rows.map((r,i) => {
    const v = Math.max(0, Number(valueFn(r)) || 0);
    const x = pad.left + i * (plotW / rows.length) + 8;
    const bh = plotH * v / yMax, y = h - pad.bottom - bh;
    const labelX = x + bw / 2;
    return `<rect x="${x}" y="${y}" width="${bw}" height="${bh}" rx="3" fill="${color}" ${tooltipAttr(`${r.id}\n${legendLabel}: ${fmt(v)}`)}></rect>
      <text class="bar-value" x="${labelX}" y="${Math.max(pad.top + 14, y - 6)}" text-anchor="middle">${fmt(v)}</text>`;
  }).join("");
}
function profileChart(svgId, rows) {
  const frame = chartFrame(svgId, rows, 1, "score");
  const { svg, h, pad, plotW, plotH, base } = frame;
  if (!rows.length) { noData(svg, "No runs selected"); return; }
  const series = [
    { key: "f1", label: "F1", color: colors.f1 },
    { key: "precision", label: "Precision", color: colors.precision },
    { key: "recall", label: "Recall", color: colors.recall },
  ];
  const groupW = plotW / rows.length;
  const bw = Math.max(5, Math.min(18, (groupW - 18) / series.length));
  svg.innerHTML = base + legend(series) + rows.map((r,i) => {
    const baseX = pad.left + i * groupW + 8;
    const bars = series.map((s,j) => {
      const metric = keyMetric(r, s.key);
      const v = Math.max(0, Math.min(1, scoreValue(metric) || 0));
      const x = baseX + j * (bw + 3), bh = plotH * v, y = h - pad.bottom - bh;
      return `<rect x="${x}" y="${y}" width="${bw}" height="${bh}" rx="2" fill="${s.color}" ${tooltipAttr(`${r.id}\n${s.label}: ${fmt(v)}`)}></rect>`;
    }).join("");
    return bars;
  }).join("");
}
function stackedCounts(svgId, rows) {
  const svg = document.getElementById(svgId), w = 820, h = 340;
  const pad = { left: 74, right: 22, top: 44, bottom: 36 };
  const plotW = w - pad.left - pad.right, plotH = h - pad.top - pad.bottom;
  const counts = rows.map(r => metricObj(r, state.metric));
  const max = Math.max(1, ...counts.map(c => (c.tp||0)+(c.fp||0)+(c.fn||0)));
  const bw = Math.max(10, plotW / Math.max(1, rows.length) - 16);
  svg.setAttribute("viewBox", `0 0 ${w} ${h}`);
  if (!rows.length) { noData(svg, "No runs selected"); return; }
  const ticks = [0, .25, .5, .75, 1].map(t => Math.round(max * t));
  const base = ticks.map(t => {
    const y = pad.top + plotH - plotH * t / max;
    return `<line class="grid-line" x1="${pad.left}" y1="${y}" x2="${w-pad.right}" y2="${y}"/><text class="tick" x="${pad.left-10}" y="${y+4}" text-anchor="end">${t}</text>`;
  }).join("") + `<line class="axis" x1="${pad.left}" y1="${pad.top}" x2="${pad.left}" y2="${h-pad.bottom}"/><line class="axis" x1="${pad.left}" y1="${h-pad.bottom}" x2="${w-pad.right}" y2="${h-pad.bottom}"/><text class="axis-label" x="18" y="${pad.top + plotH/2}" transform="rotate(-90 18 ${pad.top + plotH/2})">count</text>`;
  svg.innerHTML = base + legend([{label:"TP", color:colors.tp}, {label:"FP", color:colors.fp}, {label:"FN", color:colors.fn}]) + rows.map((r,i) => {
    let y = h - pad.bottom, x = pad.left + i * (plotW / rows.length) + 8;
    const bars = [["tp",colors.tp],["fp",colors.fp],["fn",colors.fn]].map(([k,c]) => {
      const v = metricObj(r, state.metric)[k] || 0;
      const height = plotH * v / max;
      y -= height; return `<rect x="${x}" y="${y}" width="${bw}" height="${height}" fill="${c}" ${tooltipAttr(`${r.id}\n${k.toUpperCase()}: ${v}`)}></rect>`;
    }).join("");
    return bars;
  }).join("");
}
function scatter(svgId, rows) {
  const svg = document.getElementById(svgId), w = 820, h = 340;
  const pad = { left: 76, right: 28, top: 44, bottom: 58 };
  const plotW = w - pad.left - pad.right, plotH = h - pad.top - pad.bottom;
  svg.setAttribute("viewBox", `0 0 ${w} ${h}`);
  if (!rows.length) { noData(svg, "No runs selected"); return; }
  const ticks = [0, .25, .5, .75, 1];
  const grid = ticks.map(t => {
    const x = pad.left + plotW * t, y = pad.top + plotH - plotH * t;
    return `<line class="grid-line" x1="${x}" y1="${pad.top}" x2="${x}" y2="${h-pad.bottom}"/><line class="grid-line" x1="${pad.left}" y1="${y}" x2="${w-pad.right}" y2="${y}"/><text class="tick" x="${x}" y="${h-pad.bottom+18}" text-anchor="middle">${fmt(t)}</text><text class="tick" x="${pad.left-10}" y="${y+4}" text-anchor="end">${fmt(t)}</text>`;
  }).join("");
  svg.innerHTML = grid +
    `<line class="axis" x1="${pad.left}" y1="${h-pad.bottom}" x2="${w-pad.right}" y2="${h-pad.bottom}"/><line class="axis" x1="${pad.left}" y1="${pad.top}" x2="${pad.left}" y2="${h-pad.bottom}"/><text class="axis-label" x="${pad.left + plotW/2}" y="${h-14}" text-anchor="middle">precision</text><text class="axis-label" x="18" y="${pad.top + plotH/2}" transform="rotate(-90 18 ${pad.top + plotH/2})">recall</text>` +
    legend([{ label: "run point", color: colors.metric }, { label: "ideal corner", color: colors.tp }]) +
    `<circle cx="${w-pad.right}" cy="${pad.top}" r="5" fill="${colors.tp}" ${tooltipAttr("Ideal\nprecision: 1.0000\nrecall: 1.0000")}></circle>` +
    rows.map((r, i) => {
      const p = scoreValue(keyMetric(r, "precision"));
      const rec = scoreValue(keyMetric(r, "recall"));
      if (p === null || p === undefined || rec === null || rec === undefined) return "";
      const x = pad.left + p * plotW, y = h - pad.bottom - rec * plotH;
      return `<circle cx="${x}" cy="${y}" r="6" fill="${colors.metric}" stroke="#d7e8ff" stroke-width="1.2" ${tooltipAttr(`${r.id}\nprecision: ${fmt(p)}\nrecall: ${fmt(rec)}`)}></circle>`;
    }).join("");
}
function renderDetail() {
  const run = selectedRuns()[0];
  if (!run) { document.getElementById("detail").innerHTML = ""; return; }
  document.getElementById("detail").innerHTML = `
    <p><b>${run.id}</b></p>
    <p class="muted">${run.run_dir}</p>
    <p>Model: <b>${run.model.id || ""}</b><br>Dataset: <b>${run.dataset.id || ""}</b><br>Samples: <b>${run.samples_count}</b></p>
    <p>Artifacts: ${Object.entries(run.artifacts).map(([k,v]) => `<span class="pill">${k}: ${v}</span>`).join(" ")}</p>
    <table><tbody>${run.metrics.map(m => `<tr><td>${m.name}</td><td>${fmt(m.value)}</td><td>TP ${m.tp ?? ""}</td><td>FP ${m.fp ?? ""}</td><td>FN ${m.fn ?? ""}</td></tr>`).join("")}</tbody></table>
  `;
}
function renderSamples() {
  const section = document.getElementById("samplesSection");
  if (!manifest.include_samples) { section.classList.add("hidden"); return; }
  section.classList.remove("hidden");
  const run = selectedRuns()[0];
  const samples = run ? run.samples || [] : [];
  document.getElementById("samples").innerHTML = samples.map(s => `<div class="sample">
    <div class="muted">${s.sample_id || ""}</div>
    <pre>${escapeHtml(s.text || "")}</pre>
    <div>Target: ${(s.target_spans||[]).map(spanPill).join(" ")}</div>
    <div>Prediction: ${(s.prediction_spans||[]).map(spanPill).join(" ")}</div>
  </div>`).join("");
}
function spanPill(s) { return `<span class="pill">${s.label || ""} ${s.begin ?? ""}-${s.end ?? ""}${s.matched === false ? " · miss" : ""}</span>`; }
function renderCharts() {
  const rows = selectedRuns();
  profileChart("profileChart", rows);
  barChart("metricChart", rows, r => metricValue(r, state.metric) || 0, colors.metric, state.metric || "metric");
  stackedCounts("countChart", rows);
  scatter("scatterChart", rows);
}
function render() {
  renderControls(); renderOverview(); renderTable(); renderCharts(); renderDetail(); renderSamples();
}
document.getElementById("search").addEventListener("input", e => { state.search = e.target.value; render(); });
document.getElementById("metric").addEventListener("change", e => { state.metric = e.target.value; render(); });
document.getElementById("runName").addEventListener("change", e => { state.runName = e.target.value; render(); });
document.getElementById("selectVisible").addEventListener("click", () => { state.selected = new Set(filteredRuns().map(r => r.id)); render(); });
document.querySelectorAll("[data-fullscreen-target]").forEach(button => {
  button.addEventListener("click", () => openFullscreenChart(button.dataset.fullscreenTarget, button.dataset.fullscreenTitle));
});
document.getElementById("chartFullscreenClose").addEventListener("click", closeFullscreenChart);
document.getElementById("chartFullscreen").addEventListener("click", event => {
  if (event.target.id === "chartFullscreen") closeFullscreenChart();
});
document.addEventListener("keydown", event => {
  if (event.key === "Escape" && !document.getElementById("chartFullscreen").classList.contains("hidden")) closeFullscreenChart();
});
document.addEventListener("mouseover", event => {
  const target = event.target.closest?.("[data-tooltip]");
  if (target) showTooltip(target.dataset.tooltip, event);
});
document.addEventListener("mousemove", event => {
  if (!document.getElementById("tooltip").classList.contains("hidden")) moveTooltip(event);
});
document.addEventListener("mouseout", event => {
  const target = event.target.closest?.("[data-tooltip]");
  if (target && !target.contains(event.relatedTarget)) hideTooltip();
});
render();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    raise SystemExit(main())
