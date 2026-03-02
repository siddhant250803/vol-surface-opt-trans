# SPX Vol Surface — Optimal Transport Analysis

Optimal transport (W1, W2) applied to SPX options: risk-neutral vs physical distributions, variance risk premium, and regime detection.

---

## Overview

We compare **risk-neutral (Q)** and **physical (P)** distributions from SPX options using Wasserstein distances. Main outputs:

- **W1(Q, Q_prev)** — surface shift proxy, correlates with forward RV − IV² (≈0.54)
- **W2(Q, P)** — regime proxy, spikes in stress (Mar 2020, Oct 2022)

Q recovered via Breeden–Litzenberger; P via bootstrap of historical returns.

---

## Main Discoveries

| Metric | Meaning | Finding |
|--------|---------|---------|
| **W1(Q, Q_prev)** | Risk-neutral density change day-to-day | High W1 → RV tends to exceed IV; low W1 → RV ≈ IV |
| **W2(Q, P)** | Q–P divergence | Higher in stress than calm periods |
| **Decile spread** | D10 − D1 mean(RV − IV²) | ≈ 0.17 (D1 ≈ −0.004, D10 ≈ 0.17) |

---

## Visualizations

![W1 time series](outputs/report/factor/w1_q_vs_q_ts.png) ![W2 time series](outputs/report/factor/w2_q_vs_p_ts.png)

![Decile chart](outputs/report/ot_findings/decile_chart.png) ![W1 vs RV-IV](outputs/report/ot_findings/w1_vs_rv_iv_scatter.png)

**Interactive:** [w1_w2_timeseries](outputs/report/ot_findings/w1_w2_timeseries.html) · [decile_chart](outputs/report/ot_findings/decile_chart.html) · [w1_vs_rv_iv](outputs/report/ot_findings/w1_vs_rv_iv_scatter.html) · [3D surfaces](outputs/report/ot_findings/surfaces_q_vs_p_7d.html)

---

## Methodology

1. **SVI fit** → implied vol surface  
2. **Call prices** → from SVI  
3. **Q recovery** → Breeden–Litzenberger (∂²C/∂K²)  
4. **P estimation** → bootstrap of historical returns  
5. **Distances** → W1, W2 via quantile-based formulas  

**Formulas:** W1(μ,ν) = ∫₀¹ |F_μ⁻¹(u) − F_ν⁻¹(u)| du · W2(μ,ν) = √∫₀¹ (F_μ⁻¹(u) − F_ν⁻¹(u))² du

---

## Setup & Run

```bash
pip install -e .   # or: uv sync
```

**Data:** Place options + yield CSVs per `configs/data.yaml`.

```bash
python -m pipeline.run --config configs/base.yaml
PYTHONPATH=. python scripts/generate_ot_report.py
PYTHONPATH=. python scripts/rv_iv_analysis.py
```

**Outputs:** `outputs/report/ot_findings/` (HTML, PNG) · `outputs/report/factor/` (PNG) · `outputs/cache/` · `outputs/features/`


