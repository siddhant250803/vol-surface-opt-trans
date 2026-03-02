# Variance Risk Premium (VRP): Definitions and Horizon Matching

We work at a fixed option horizon $H$ in **calendar days** (e.g., $H = 7$). For each observation date $t$, let $\tau_t$ denote the option's time-to-expiry in **years** for the target bucket:

$$
\tau_t = \frac{H}{365}
$$

(If the option expiry is not exactly $H$ days, $\tau_t$ is the actual year fraction used in pricing; we treat it as the horizon-matched maturity for that bucket.)

---

## Implied variance from SVI (risk-neutral)

SVI parameterizes the **total implied variance** (a.k.a. Black–Scholes total variance) as a function of log-moneyness $k$ and maturity $\tau$:

$$
w_t(k,\tau) = \sigma_{t,\text{ann}}^2(k,\tau) \cdot \tau
$$

where $`\sigma_{t,\text{ann}}(k,\tau)`$ is the **annualized** implied volatility.

For the ATM point ($k=0$) at the target horizon $\tau_t$, define:

$$
w^{\text{ATM}}_t \equiv w_t(0,\tau_t)
$$

Then the **annualized implied variance** at ATM is:

$$
IV^2_{t,\text{ann}} \equiv \sigma^2_{t,\text{ann}}(0,\tau_t) = \frac{w^{\text{ATM}}_t}{\tau_t}
$$

Equivalently, the ATM implied volatility is:

$$
IV_{t,\text{ann}} = \sqrt{\frac{w^{\text{ATM}}_t}{\tau_t}}
$$

**Important:** In code and reporting, "ATM IV" must be distinguished from "ATM total variance." Ideally, `atm_iv` would represent $`IV_{t,\text{ann}}`$ (volatility). In this codebase, the variable `atm_iv` stores **total variance** $`w^{\text{ATM}}_t`$ (SVI raw); annualized IV² is computed as `atm_iv / tau`.

---

## Realized variance over the same forward horizon (physical)

Let $S_t$ be the index level at the close on date $t$, and define daily close-to-close log returns:

$$
r_{t+i} = \log\left(\frac{S_{t+i}}{S_{t+i-1}}\right)
$$

Let $t \to t+M$ denote the forward window whose calendar span is $`\text{span\_days}_t \geq H`$ (we choose $M$ such that the forward window spans at least $H$ calendar days; if the next $H$ calendar days include weekends/holidays, $M$ is the number of trading-day returns in that span).

Define **forward realized variance** over that window:

$$
RV_{t,H} \equiv \sum_{i=1}^{M} r_{t+i}^2
$$

This quantity is in "variance over the realized window" units (not annualized).

We annualize it by scaling by calendar time:

$$
RV_{t,\text{ann}} \equiv RV_{t,H} \times \frac{365}{\text{span\_days}_t}
$$

This makes $`RV_{t,\text{ann}}`$ directly comparable to $`IV^2_{t,\text{ann}}`$, since both are annualized variances.

---

## Variance Risk Premium (annualized)

We define the **annualized VRP** at horizon $H$ as:

$$
VRP_{t,H} \equiv RV_{t,\text{ann}} - IV^2_{t,\text{ann}}
$$

**Interpretation:**

- $`VRP_{t,H} > 0`$: realized variance over the forward horizon exceeds option-implied variance (implied was "cheap" relative to realized).
- $`VRP_{t,H} < 0`$: realized variance falls below implied variance (implied was "rich" relative to realized).

---

## Horizon consistency (what we enforce)

For each date $t$, the VRP uses:

1. an options slice with maturity $\tau_t \approx H/365$ (bucketed by $\tau$), and
2. a realized window whose calendar span matches the same horizon definition (at least $H$ calendar days, annualized by the actual span).

This alignment is necessary; mixing "5 trading days" RV with "7 calendar day" options maturity without adjustment will bias $`VRP_{t,H}`$. 
