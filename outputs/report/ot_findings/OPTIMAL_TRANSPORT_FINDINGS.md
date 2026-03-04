# Optimal Transport on the Implied Volatility Surface

*Wasserstein distances to extract structure from SPX options.*

---

## Definitions

See [../../docs/VRP_DEFINITIONS.md](../../docs/VRP_DEFINITIONS.md) for full VRP specification (units, horizons, SVI total variance vs ATM IV). Summary: $`VRP_{t,H} = RV_{t,\text{ann}} - IV^2_{t,\text{ann}}`$ with horizon-matched annualization. Stress = top decile of forward $`RV_{t,\text{ann}}`$.

---

## Setup

Options prices encode the **risk-neutral density** $`Q`$. We estimate the **physical distribution** $`P`$ from historical returns. The gap $`Q`$–$`P`$ is the variance risk premium. We use **optimal transport** ($`W_1`$, $`W_2`$) to measure it.

---

## Pipeline

### 1. Fit the volatility surface

We fit an **SVI** model [3] to the implied volatility smile for each expiry. SVI gives a smooth, arbitrage-free representation of the vol surface.

### 2. Recover the risk-neutral density Q

From the fitted SVI, we compute call prices on a fine grid of log-moneyness $k = \ln(K/F)$. Then we apply **Breeden–Litzenberger** [1]: the risk-neutral PDF is the second derivative of the call price with respect to strike. In log-moneyness space:

$$
q(k) = \frac{e^{rT}\left(\frac{\partial^2 c}{\partial k^2} - \frac{\partial c}{\partial k}\right)}{e^k}
$$

where $c = C/F$. We use **central finite differences** for the derivatives; clip negative density and renormalize. No analytic SVI derivatives.

### 3. Estimate the physical distribution P

**$`P`$ is built via FHS-GJR-GARCH** (Filtered Historical Simulation with GJR-GARCH(1,1)): 252-day rolling window of daily log returns, fit GJR-GARCH with Student-t innovations, residual bootstrap of standardized residuals, forward simulate to $`H`$-day cumulative returns, KDE over log-moneyness. Same grid as $`Q`$. The leverage term captures asymmetric volatility; residual bootstrap preserves volatility clustering.

### 4. Compute Wasserstein distances

For 1D distributions, the $p$-Wasserstein distance has a closed form using quantile functions [2]:

- $`W_1(\mu, \nu) = \int_0^1 |F_\mu^{-1}(u) - F_\nu^{-1}(u)|\, du`$
- $`W_2(\mu, \nu) = \sqrt{\int_0^1 (F_\mu^{-1}(u) - F_\nu^{-1}(u))^2\, du}`$

**$`Q_{\mathrm{prev}}`$**: Constant maturity: interpolate previous day's fitted surface to $`\tau`$ days to expiry, recover $`Q_{\mathrm{prev}}`$ on the same grid.

---

## Findings

### $`W_1(Q, Q_{\text{prev}})`$ and the variance risk premium

We observe a **positive association in-sample** between $`W_1(Q, Q_{\text{prev}})`$ and forward VRP (correlation $`\approx 0.54`$). High $`W_1`$ → surface in flux → RV tends to beat IV. Low $`W_1`$ → stable surface → RV tends to be at or below IV. Subperiod robustness not yet validated.

### $`W_2(Q, P)`$ as regime proxy

$`W_2(Q, P)`$ measures Q–P divergence. Under our FHS-GARCH $`P`$:

| Regime | Mean $`W_2(Q,P)`$ |
|--------|-------------------|
| Calm (bottom RV decile) | 0.047 |
| Stress (top RV decile)  | 0.035 |

$`W_2`$ is *lower* in stress than calm (diff = −0.012). See [DIAGNOSTICS.md](DIAGNOSTICS.md).

### Decile evidence

Sorting dates by $`W_1(Q, Q_{\mathrm{prev}})`$ and computing mean(VRP) by decile:

| Decile | Mean(VRP) |
|--------|------------|
| D1 (low $`W_1`$)  | −0.004 |
| D10 (high $`W_1`$) | 0.17 |

Spread $`D_{10} - D_1 \approx 0.17`$. High-$`W_1`$ dates earn higher VRP on average.

---

## 3D Visualizations

$`Q`$ (blue) and $`P`$ (orange) over time and log-moneyness. Vertical gap = local Q–P divergence.

`surfaces_q_vs_p_7d.html` · `surfaces_comparison_7d.html`

---

## Limitations

- **Data:** Weekly options → coarse series; daily would sharpen.
- **$`P`$:** FHS-GJR-GARCH; jump-diffusion could further refine.
- **Tenor:** 7D only here; multi-tenor would add term structure.
- **$`W_1`$–VRP:** In-sample association; subperiod robustness pending.
- **No transaction cost modeling** yet; strategy backtests are next.

---

## References

[1] Breeden, D.T. & Litzenberger, R.H. (1978). Prices of state-contingent claims implicit in option prices. *Journal of Business*, 51(4), 621–651.

[2] Villani, C. (2003). *Topics in Optimal Transportation*. AMS. (1D quantile formula: §2.1.)

[3] Gatheral, J. & Jacquier, A. (2014). Arbitrage-free SVI volatility surfaces. *Quantitative Finance*, 14(1), 59–71.
