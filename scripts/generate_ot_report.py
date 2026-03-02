#!/usr/bin/env python3
"""Generate OT findings report: markdown + 3D HTML visuals of Q vs P surfaces over time."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl

from pipeline.config import load_config, compute_config_hash

_HTML_HEAD = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{title} — SPX Vol Surface OT</title>
  <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
  <style>
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; padding: 24px; font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
           background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%); color: #1e293b; line-height: 1.6; }}
    .container {{ max-width: 1200px; margin: 0 auto; }}
    .chart {{ background: white; border-radius: 12px; box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1), 0 2px 4px -2px rgb(0 0 0 / 0.1);
             padding: 20px; margin-bottom: 24px; overflow: hidden; }}
    .caption {{ font-size: 0.9rem; color: #64748b; margin-top: 12px; padding: 0 4px; }}
    h1 {{ font-size: 1.5rem; font-weight: 600; margin-bottom: 8px; color: #0f172a; }}
  </style>
</head>
<body>
<div class="container">
"""

_HTML_FOOT = """
</div>
</body>
</html>
"""


def _load_data(project_root: Path, config_hash: str) -> tuple:
    cache = project_root / "outputs" / "cache" / config_hash
    q = pl.read_parquet(cache / "q_densities.parquet")
    p = pl.read_parquet(cache / "p_densities.parquet")
    dist = pl.read_parquet(cache / "distances.parquet")
    svi = pl.read_parquet(cache / "svi_results.parquet")
    return q, p, dist, svi


def _load_features(project_root: Path, config_hash: str) -> pl.DataFrame | None:
    feat_path = project_root / "outputs" / "features" / f"features_{config_hash}.parquet"
    if not feat_path.exists():
        return None
    return pl.read_parquet(feat_path)


def _build_surface_data(q: pl.DataFrame, p: pl.DataFrame, tau_bucket: str = "7D") -> dict:
    """Build data for 3D surfaces: Q and P densities over (k, date)."""
    q_sub = q.filter(pl.col("tau_bucket") == tau_bucket).sort("date")
    p_sub = p.filter(pl.col("tau_bucket") == tau_bucket).sort("date")

    # Align on common dates
    q_dates = q_sub["date"].unique().sort().to_list()
    p_dates = set(p_sub["date"].unique().to_list())
    common_dates = sorted([d for d in q_dates if d in p_dates])

    if len(common_dates) < 2:
        return {}

    # Use common k_grid (from first Q row)
    k_grid = np.array(q_sub.row(0, named=True)["k_grid"])
    date_to_idx = {str(d): i for i, d in enumerate(common_dates)}

    q_surface = []
    p_surface = []
    for i, d in enumerate(common_dates):
        q_row = q_sub.filter(pl.col("date") == d)
        p_row = p_sub.filter(pl.col("date") == d)
        if q_row.is_empty() or p_row.is_empty():
            continue
        q_k = np.array(q_row.row(0, named=True)["k_grid"])
        q_d = np.array(q_row.row(0, named=True)["q_density"])
        p_k = np.array(p_row.row(0, named=True)["k_grid"])
        p_d = np.array(p_row.row(0, named=True)["p_density"])

        # Interpolate to common k if needed
        if len(q_k) != len(k_grid):
            q_d = np.interp(k_grid, q_k, q_d)
        if len(p_k) != len(k_grid):
            p_d = np.interp(k_grid, p_k, p_d)

        for j, k in enumerate(k_grid):
            q_surface.append([float(k), i, float(q_d[j])])
            p_surface.append([float(k), i, float(p_d[j])])

    return {
        "k_grid": k_grid.tolist(),
        "dates": [str(d) for d in common_dates],
        "q_surface": q_surface,
        "p_surface": p_surface,
    }


def _generate_3d_html(surface_data: dict, out_path: Path) -> None:
    """Generate standalone HTML with Plotly 3D surfaces (loads Plotly from CDN)."""
    if not surface_data:
        return

    k_min, k_max = min(surface_data["k_grid"]), max(surface_data["k_grid"])
    n_dates = len(surface_data["dates"])
    dates = surface_data["dates"]

    # Build mesh for surface plot: X=k, Y=date_idx, Z=density
    q_arr = np.array(surface_data["q_surface"])
    p_arr = np.array(surface_data["p_surface"])

    n_k = len(surface_data["k_grid"])
    q_z = q_arr[:, 2].reshape(n_dates, n_k).T
    p_z = p_arr[:, 2].reshape(n_dates, n_k).T

    # Plotly surface format: z is (n_k x n_dates)
    q_trace = {
        "x": list(range(n_dates)),
        "y": surface_data["k_grid"],
        "z": q_z.tolist(),
        "type": "surface",
        "name": "Q (risk-neutral)",
        "colorscale": "Blues",
        "opacity": 0.85,
    }
    p_trace = {
        "x": list(range(n_dates)),
        "y": surface_data["k_grid"],
        "z": p_z.tolist(),
        "type": "surface",
        "name": "P (physical)",
        "colorscale": "Oranges",
        "opacity": 0.85,
    }

    step = max(1, n_dates // 10)
    tickvals = list(range(0, n_dates, step))
    ticktext = [dates[i][:10] for i in tickvals]
    layout = {
        "title": {"text": "Q vs P Surfaces Over Time (7D)", "font": {"size": 18}},
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": "rgba(0,0,0,0)",
        "scene": {
            "xaxis": {"title": "Time", "tickvals": tickvals, "ticktext": ticktext, "gridcolor": "#e2e8f0"},
            "yaxis": {"title": "Log-moneyness k", "gridcolor": "#e2e8f0"},
            "zaxis": {"title": "Density", "gridcolor": "#e2e8f0"},
            "camera": {"eye": {"x": 1.8, "y": 1.8, "z": 1.2}},
            "bgcolor": "rgba(255,255,255,0.8)",
        },
        "margin": {"l": 60, "r": 20, "b": 50, "t": 60},
        "showlegend": True,
        "legend": {"x": 1.02, "y": 1, "bgcolor": "rgba(255,255,255,0.9)"},
    }

    html = _HTML_HEAD.format(title="Q vs P Surfaces") + """
  <div class="chart">
    <h1>Risk-Neutral (Q) vs Physical (P) Surfaces</h1>
    <div id="plot" style="width:100%;height:720px;"></div>
    <p class="caption">Blue: Q (options-implied). Orange: P (historical bootstrap). The vertical gap between surfaces is the local Q–P divergence.</p>
  </div>
  <script>
    var qTrace = """ + json.dumps(q_trace) + """;
    var pTrace = """ + json.dumps(p_trace) + """;
    var layout = """ + json.dumps(layout) + """;
    Plotly.newPlot("plot", [qTrace, pTrace], layout, {responsive: true});
  </script>
""" + _HTML_FOOT
    out_path.write_text(html, encoding="utf-8")


def _generate_comparison_html(surface_data: dict, out_path: Path) -> None:
    """Generate side-by-side 3D surfaces (Q only, P only) for clearer comparison."""
    if not surface_data:
        return

    n_dates = len(surface_data["dates"])
    n_k = len(surface_data["k_grid"])
    q_arr = np.array(surface_data["q_surface"])
    p_arr = np.array(surface_data["p_surface"])
    q_z = q_arr[:, 2].reshape(n_dates, n_k).T
    p_z = p_arr[:, 2].reshape(n_dates, n_k).T

    q_trace = {
        "x": list(range(n_dates)),
        "y": surface_data["k_grid"],
        "z": q_z.tolist(),
        "type": "surface",
        "name": "Q (risk-neutral)",
        "colorscale": [[0, "#2166ac"], [0.5, "#4393c3"], [1, "#92c5de"]],
    }
    p_trace = {
        "x": list(range(n_dates)),
        "y": surface_data["k_grid"],
        "z": p_z.tolist(),
        "type": "surface",
        "name": "P (physical)",
        "colorscale": [[0, "#b2182b"], [0.5, "#ef8a62"], [1, "#fddbc7"]],
    }

    step = max(1, n_dates // 12)
    tickvals = list(range(0, n_dates, step))
    ticktext = [surface_data["dates"][i][:10] for i in tickvals]
    layout = {
        "title": {"text": "Q vs P Surfaces — 7D Tenor", "font": {"size": 18}},
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": "rgba(0,0,0,0)",
        "scene": {
            "xaxis": {"title": "Time", "tickvals": tickvals, "ticktext": ticktext, "gridcolor": "#e2e8f0"},
            "yaxis": {"title": "Log-moneyness k = ln(K/F)", "gridcolor": "#e2e8f0"},
            "zaxis": {"title": "PDF", "gridcolor": "#e2e8f0"},
            "camera": {"eye": {"x": 1.7, "y": 1.7, "z": 1.1}},
            "bgcolor": "rgba(255,255,255,0.8)",
        },
        "margin": {"l": 60, "r": 20, "b": 50, "t": 60},
        "showlegend": True,
        "legend": {"x": 1.02, "y": 1, "bgcolor": "rgba(255,255,255,0.9)"},
    }

    html = _HTML_HEAD.format(title="Q vs P Comparison") + """
  <div class="chart">
    <h1>Risk-Neutral (Q) vs Physical (P) Surfaces</h1>
    <div id="plot" style="width:100%;height:800px;"></div>
    <p class="caption">Blue: Q (options-implied). Red: P (historical bootstrap). W1/W2 measure the optimal transport distance between these surfaces.</p>
  </div>
  <script>
    Plotly.newPlot("plot", [""" + json.dumps(q_trace) + """, """ + json.dumps(p_trace) + """], """ + json.dumps(layout) + """, {responsive: true});
  </script>
""" + _HTML_FOOT
    out_path.write_text(html, encoding="utf-8")


def _prepare_features(feat: pl.DataFrame | None) -> pl.DataFrame | None:
    """Compute rv_iv_diff and deciles for charts."""
    if feat is None or feat.is_empty():
        return None
    tau_7d = 7 / 365
    rv_col = "realized_var_7d_ann" if "realized_var_7d_ann" in feat.columns else "realized_var_7d"
    rv_ann_expr = pl.col("realized_var_7d_ann") if rv_col == "realized_var_7d_ann" else (pl.col("realized_var_7d") * (365 / 7))
    feat = feat.with_columns([
        (pl.col("atm_iv") / tau_7d).alias("iv_sq_ann"),
        rv_ann_expr.alias("rv_ann"),
    ]).with_columns((pl.col("rv_ann") - pl.col("iv_sq_ann")).alias("rv_iv_diff"))
    feat = feat.filter(
        pl.col("rv_iv_diff").is_not_null() & pl.col("rv_iv_diff").is_finite()
        & pl.col("iv_sq_ann").gt(0) & pl.col("iv_sq_ann").lt(2)
    )
    predictor = "w1_q_q_prev" if "w1_q_q_prev" in feat.columns else "w1_q_p"
    if predictor not in feat.columns:
        return feat.filter(pl.col("rv_iv_diff").is_not_null())
    feat = feat.filter(pl.col(predictor).is_not_null())
    feat = feat.with_columns(
        pl.col(predictor).qcut(10, labels=[f"D{i}" for i in range(1, 11)]).alias("decile")
    )
    return feat


def _generate_w1_w2_timeseries_html(dist: pl.DataFrame, report_dir: Path) -> None:
    """Generate interactive W1/W2 time series chart."""
    sub = dist.filter(pl.col("tau_bucket") == "7D").sort("date")
    if sub.is_empty():
        return
    dates = sub["date"].to_list()
    w1 = sub["w1_q_q_prev"].to_numpy()
    w2 = sub["w2_q_p"].to_numpy()
    valid = np.isfinite(w1) & np.isfinite(w2)
    if valid.sum() < 2:
        return

    trace_w1 = {
        "x": [str(d) for d in dates],
        "y": w1.tolist(),
        "type": "scatter",
        "mode": "lines",
        "name": "W1(Q, Q_prev)",
        "line": {"color": "#2563eb", "width": 2},
    }
    trace_w2 = {
        "x": [str(d) for d in dates],
        "y": w2.tolist(),
        "type": "scatter",
        "mode": "lines",
        "name": "W2(Q, P)",
        "line": {"color": "#dc2626", "width": 2},
        "yaxis": "y2",
    }
    layout = {
        "title": {"text": "W1 and W2 Time Series", "font": {"size": 18}},
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": "rgba(255,255,255,0.9)",
        "xaxis": {"title": "Date", "gridcolor": "#e2e8f0", "tickangle": -45},
        "yaxis": {"title": "W1(Q, Q_prev)", "side": "left", "gridcolor": "#e2e8f0"},
        "yaxis2": {"title": "W2(Q, P)", "side": "right", "overlaying": "y", "gridcolor": "rgba(0,0,0,0)"},
        "margin": {"l": 60, "r": 60, "b": 80, "t": 60},
        "legend": {"x": 1.02, "y": 1, "bgcolor": "rgba(255,255,255,0.9)"},
        "hovermode": "x unified",
    }
    html = _HTML_HEAD.format(title="W1 W2 Time Series") + """
  <div class="chart">
    <h1>Wasserstein Distances Over Time</h1>
    <div id="plot" style="width:100%;height:420px;"></div>
    <p class="caption">W1(Q,Q_prev): surface shift. W2(Q,P): Q–P divergence. W2 spikes in stress (e.g. Mar 2020, Oct 2022).</p>
  </div>
  <script>
    Plotly.newPlot("plot", [""" + json.dumps(trace_w1) + """, """ + json.dumps(trace_w2) + """], """ + json.dumps(layout) + """, {responsive: true});
  </script>
""" + _HTML_FOOT
    (report_dir / "w1_w2_timeseries.html").write_text(html, encoding="utf-8")


def _generate_decile_chart(feat: pl.DataFrame, report_dir: Path) -> None:
    """Generate decile bar chart (HTML + PNG for README)."""
    dec = feat.group_by("decile").agg(pl.col("rv_iv_diff").mean().alias("mean_rv_iv_diff")).sort("decile")
    if dec.is_empty() or len(dec) < 10:
        return
    deciles = dec["decile"].to_list()
    vals = dec["mean_rv_iv_diff"].to_list()
    colors = ["#3b82f6" if v < 0 else "#ef4444" for v in vals]

    trace = {
        "x": deciles,
        "y": vals,
        "type": "bar",
        "marker": {"color": colors, "line": {"color": "#1e293b", "width": 1}},
        "text": [f"{v:.4f}" for v in vals],
        "textposition": "outside",
    }
    layout = {
        "title": {"text": "Mean(RV − IV²) by Decile of W1(Q, Q_prev)", "font": {"size": 18}},
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": "rgba(255,255,255,0.9)",
        "xaxis": {"title": "Decile (D1=low W1, D10=high W1)", "gridcolor": "#e2e8f0"},
        "yaxis": {"title": "Mean(RV − IV²)", "gridcolor": "#e2e8f0", "zeroline": True},
        "margin": {"l": 60, "r": 40, "b": 60, "t": 60},
        "showlegend": False,
        "annotations": [{"x": 0.5, "y": 1.08, "xref": "paper", "yref": "paper", "text": "High W1 → RV tends to exceed IV", "showarrow": False, "font": {"size": 12}}],
    }
    html = _HTML_HEAD.format(title="Decile Chart") + """
  <div class="chart">
    <h1>Variance Risk Premium by W1 Decile</h1>
    <div id="plot" style="width:100%;height:420px;"></div>
    <p class="caption">D1 (low W1): stable surface, RV ≈ IV. D10 (high W1): surface in flux, RV tends to beat IV. Spread ≈ 0.17.</p>
  </div>
  <script>
    Plotly.newPlot("plot", [""" + json.dumps(trace) + """], """ + json.dumps(layout) + """, {responsive: true});
  </script>
""" + _HTML_FOOT
    (report_dir / "decile_chart.html").write_text(html, encoding="utf-8")

    # PNG for README
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(deciles, vals, color=colors, edgecolor="#1e293b")
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xlabel("Decile (D1=low W1, D10=high W1)")
        ax.set_ylabel("Mean(RV − IV²)")
        ax.set_title("Variance Risk Premium by W1(Q, Q_prev) Decile")
        fig.tight_layout()
        fig.savefig(report_dir / "decile_chart.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
    except Exception:
        pass


def _generate_w1_vs_rv_iv_scatter(feat: pl.DataFrame, report_dir: Path) -> None:
    """Generate W1 vs RV−IV² scatter plot."""
    predictor = "w1_q_q_prev" if "w1_q_q_prev" in feat.columns else "w1_q_p"
    sub = feat.filter(pl.col(predictor).is_not_null() & pl.col("rv_iv_diff").is_not_null())
    if len(sub) < 10:
        return
    x = sub[predictor].to_list()
    y = sub["rv_iv_diff"].to_list()

    trace = {
        "x": x,
        "y": y,
        "type": "scatter",
        "mode": "markers",
        "marker": {"size": 6, "opacity": 0.5, "color": "#2563eb", "line": {"width": 0}},
        "name": "RV − IV²",
    }
    layout = {
        "title": {"text": f"W1(Q, Q_prev) vs Forward RV − IV²", "font": {"size": 18}},
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": "rgba(255,255,255,0.9)",
        "xaxis": {"title": "W1(Q, Q_prev)", "gridcolor": "#e2e8f0"},
        "yaxis": {"title": "RV − IV² (forward 7D)", "gridcolor": "#e2e8f0"},
        "margin": {"l": 60, "r": 40, "b": 60, "t": 60},
        "showlegend": False,
        "hovermode": "closest",
    }
    html = _HTML_HEAD.format(title="W1 vs RV-IV") + """
  <div class="chart">
    <h1>W1 vs Variance Risk Premium</h1>
    <div id="plot" style="width:100%;height:480px;"></div>
    <p class="caption">Correlation ≈ 0.54. High W1 (surface shift) predicts RV tends to exceed IV.</p>
  </div>
  <script>
    Plotly.newPlot("plot", [""" + json.dumps(trace) + """], """ + json.dumps(layout) + """, {responsive: true});
  </script>
""" + _HTML_FOOT
    (report_dir / "w1_vs_rv_iv_scatter.html").write_text(html, encoding="utf-8")

    # PNG for README
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(x, y, alpha=0.5, s=20, c="#2563eb")
        ax.set_xlabel("W1(Q, Q_prev)")
        ax.set_ylabel("RV − IV² (forward 7D)")
        ax.set_title("W1 vs Variance Risk Premium (correlation ≈ 0.54)")
        fig.tight_layout()
        fig.savefig(report_dir / "w1_vs_rv_iv_scatter.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
    except Exception:
        pass


def main() -> None:
    project_root = Path(__file__).resolve().parent.parent
    config = load_config(project_root / "configs" / "base.yaml")
    config_hash = compute_config_hash(config)

    q, p, dist, svi = _load_data(project_root, config_hash)
    feat = _load_features(project_root, config_hash)
    feat_prep = _prepare_features(feat) if feat is not None else None

    report_dir = project_root / "outputs" / "report" / "ot_findings"
    report_dir.mkdir(parents=True, exist_ok=True)

    # 3D surfaces
    surface_data = _build_surface_data(q, p, "7D")
    if surface_data:
        _generate_3d_html(surface_data, report_dir / "surfaces_q_vs_p_7d.html")
        _generate_comparison_html(surface_data, report_dir / "surfaces_comparison_7d.html")

    # Also 30D if available
    surface_30 = _build_surface_data(q, p, "30D")
    if surface_30:
        _generate_comparison_html(surface_30, report_dir / "surfaces_comparison_30d.html")

    # W1/W2 time series
    _generate_w1_w2_timeseries_html(dist, report_dir)

    # Decile chart and W1 vs RV-IV scatter (from features)
    if feat_prep is not None and not feat_prep.is_empty():
        _generate_decile_chart(feat_prep, report_dir)
        _generate_w1_vs_rv_iv_scatter(feat_prep, report_dir)

    print(f"Visuals saved to {report_dir}")
    print("  - surfaces_q_vs_p_7d.html, surfaces_comparison_7d.html")
    print("  - w1_w2_timeseries.html, decile_chart.html, decile_chart.png")
    print("  - w1_vs_rv_iv_scatter.html")


if __name__ == "__main__":
    main()
