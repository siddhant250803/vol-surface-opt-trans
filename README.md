# SPX Vol Surface — Optimal Transport Analysis

Optimal transport ($W_1$, $W_2$) applied to SPX options: risk-neutral vs physical distributions, variance risk premium, and regime detection.

---

## Overview

We compare **risk-neutral ($Q$)** and **physical ($P$)** distributions from SPX options using Wasserstein distances. Main outputs:

- **$W_1(Q, Q_{\text{prev}})$** — surface shift proxy, correlates with forward $\mathrm{RV} - \mathrm{IV}^2$ ($\approx 0.54$)
- **$W_2(Q, P)$** — regime proxy, spikes in stress (Mar 2020, Oct 2022)

$Q$ recovered via Breeden–Litzenberger; $P$ via bootstrap of historical returns.

---

## Main Discoveries

| Metric | Meaning | Finding |
|--------|---------|---------|
| **W1(Q, Q_prev)** | Risk-neutral density change day-to-day | High W1 → RV tends to exceed IV; low W1 → RV ≈ IV |
| **W2(Q, P)** | Q–P divergence | Higher in stress than calm periods |
| **Decile spread** | $D_{10} - D_1$ mean($\mathrm{RV} - \mathrm{IV}^2$) | $\approx 0.17$ ($D_1 \approx -0.004$, $D_{10} \approx 0.17$) |

---

## Visualizations

![W1 time series](outputs/report/factor/w1_q_vs_q_ts.png) ![W2 time series](outputs/report/factor/w2_q_vs_p_ts.png)

![Decile chart](outputs/report/ot_findings/decile_chart.png) ![W1 vs RV-IV](outputs/report/ot_findings/w1_vs_rv_iv_scatter.png)

**Interactive:** [w1_w2_timeseries](outputs/report/ot_findings/w1_w2_timeseries.html) · [decile_chart](outputs/report/ot_findings/decile_chart.html) · [w1_vs_rv_iv](outputs/report/ot_findings/w1_vs_rv_iv_scatter.html) · [3D surfaces](outputs/report/ot_findings/surfaces_q_vs_p_7d.html)

---

## Methodology

1. **SVI fit** → implied vol surface [Gatheral & Jacquier (2014)]
2. **Call prices** → from SVI  
3. **Q recovery** → Breeden–Litzenberger ($\partial^2 C/\partial K^2$) [Breeden & Litzenberger (1978)]
4. **P estimation** → bootstrap of historical returns  
5. **Distances** → W1, W2 via quantile-based formulas [Villani (2003)]

**Formulas (1D):**

$W_1(\mu, \nu) = \int_0^1 |F_\mu^{-1}(u) - F_\nu^{-1}(u)|\, du$

$W_2(\mu, \nu) = \sqrt{\int_0^1 (F_\mu^{-1}(u) - F_\nu^{-1}(u))^2\, du}$

---

## References

- Breeden, D.T. & Litzenberger, R.H. (1978). Prices of state-contingent claims implicit in option prices. *Journal of Business*, 51(4), 621–651.
- Gatheral, J. & Jacquier, A. (2014). Arbitrage-free SVI volatility surfaces. *Quantitative Finance*, 14(1), 59–71.
- Villani, C. (2003). *Topics in Optimal Transportation*. AMS.

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


