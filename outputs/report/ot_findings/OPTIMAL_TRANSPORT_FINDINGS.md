# Optimal Transport on the Implied Volatility Surface

*Wasserstein distances to extract structure from SPX options.*

---

## Setup

Options prices encode a probability distribution: the risk-neutral density Q. It’s the market’s implied view of where the underlying might end up at expiry. Separately, we can estimate the physical distribution P from historical returns—what actually tends to happen. The gap between Q and P is the core of the variance risk premium.

The question: can we measure that gap in a principled way, and does it tell us anything useful?

We turned to **optimal transport**—specifically the Wasserstein (earth-mover) distances W1 and W2—to compare these distributions. Here’s how we did it and what we learned.

---

## Pipeline

### 1. Fit the volatility surface

We start with SPX options data: strikes, expiries, bid/ask. We fit an **SVI (stochastic volatility inspired)** model [3] to the implied volatility smile for each expiry. SVI gives a smooth, arbitrage-free representation of the vol surface.

### 2. Recover the risk-neutral density Q

From the fitted SVI, we compute call prices on a fine grid of log-moneyness $k = \ln(K/F)$. Then we apply **Breeden–Litzenberger** [1]: the risk-neutral PDF is the second derivative of the call price with respect to strike. In log-moneyness space:

$q(k) = \frac{e^{rT}\left(\frac{\partial^2 c}{\partial k^2} - \frac{\partial c}{\partial k}\right)}{e^k}$

where $c = C/F$. We use central finite differences for the derivatives. No CDF clipping—we go straight to the PDF to avoid numerical artifacts.

### 3. Estimate the physical distribution P

We bootstrap historical log returns: sample 7-day (or whatever tenor) cumulative returns by resampling daily returns with replacement. Histogram over log-moneyness gives P. Same grid as Q for comparability.

### 4. Compute Wasserstein distances

For 1D distributions, the p-Wasserstein distance has a closed form using quantile functions [2]:

- $W_1(\mu, \nu) = \int_0^1 |F_\mu^{-1}(u) - F_\nu^{-1}(u)|\, du$
- $W_2(\mu, \nu) = \sqrt{\int_0^1 (F_\mu^{-1}(u) - F_\nu^{-1}(u))^2\, du}$  

We compute inverse CDFs from the densities and integrate. W1 is the earth-mover distance in L1; W2 is the same in L2.

We compute three distances:

1. **W1(Q, Q_prev)** — how much the risk-neutral density shifted from yesterday to today (same tenor)
2. **W2(Q, P)** — how far Q is from the historical distribution P
3. **W1(Q, P)** — L1 version of Q–P divergence

---

## Findings

### W1(Q, Q_prev) predicts the variance risk premium

When the implied distribution shifts sharply from one day to the next—high W1(Q, Q_prev)—realized variance over the next 7 days tends to exceed implied variance. Correlation with forward $\mathrm{RV} - \mathrm{IV}^2$ is about 0.54.

- **High W1** → surface in flux → RV tends to beat IV  
- **Low W1** → stable surface → RV tends to be at or below IV  

So W1(Q, Q_prev) acts as a **forward variance risk premium proxy**: large surface moves predict that realized vol will outpace implied.

### W2(Q, P) flags regimes

The distance between Q (options-implied) and P (historical) spikes in stress. March 2020 and October 2022 show $W_2$ around 0.045–0.059; calm periods sit around 0.027–0.041. When the market’s risk-neutral view diverges from history, W2(Q, P) rises.

### Decile evidence

Sorting dates by $W_1(Q, Q_{\mathrm{prev}})$ and computing $\mathrm{mean}(\mathrm{RV} - \mathrm{IV}^2)$ by decile:

| Decile | Mean(RV − IV²) |
|--------|----------------|
| D1 (low W1)  | −0.004 |
| D10 (high W1) | 0.17 |

Spread $D_{10} - D_1 \approx 0.17$. High-$W_1$ dates earn higher $\mathrm{RV} - \mathrm{IV}^2$ on average. The signal is tradable in principle—we leave implementation to the reader.

---

## 3D Visualizations

Q (blue) and P (orange) over time and log-moneyness. Vertical gap = local Q–P divergence.

`surfaces_q_vs_p_7d.html` · `surfaces_comparison_7d.html`

---

## Limitations

- **Data:** Weekly options → coarse time series. Daily data would sharpen the signal.
- **P:** Bootstrap with a 252-day window. GARCH or other models could improve P.
- **Tenor:** 7D only here. Multi-tenor W1/W2 would add term-structure information.

---

## Summary

We used optimal transport (W1, W2) to compare risk-neutral and physical distributions from SPX options. W1(Q, Q_prev) predicts the variance risk premium; W2(Q, P) tracks regime. The pipeline: SVI fit → Breeden–Litzenberger → bootstrap P → Wasserstein distances. Simple, interpretable, and empirically grounded.

---

## References

[1] Breeden, D.T. & Litzenberger, R.H. (1978). Prices of state-contingent claims implicit in option prices. *Journal of Business*, 51(4), 621–651.

[2] Villani, C. (2003). *Topics in Optimal Transportation*. AMS. (1D quantile formula: §2.1.)

[3] Gatheral, J. & Jacquier, A. (2014). Arbitrage-free SVI volatility surfaces. *Quantitative Finance*, 14(1), 59–71.
