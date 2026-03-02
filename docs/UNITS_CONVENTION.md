# RV and IV² Units Convention

**Non-negotiable:** RV and IV² must be on the same scale for comparison.

## Convention 2: Annualized Variance

We use **annualized variance** for both RV and IV².

### Returns
- **Units:** Decimal (0.01 = 1%)
- **Source:** `log_ret = diff(log(spx_price))`
- **Check:** `mean(|r|)` should be ~0.01. If ~1.0, returns are in percent (bug).

### Realized Variance (RV)

**Horizon variance (h days):**

$RV_h = \sum r_i^2$ for returns spanning $\geq h$ calendar days

- $r_i$ = log returns (decimal)
- Span = calendar days from first to last return in window

**Annualized:**

$RV_{ann} = RV_h \times (365 / \mathrm{span\_days})$

**Typical magnitudes:** $RV_{ann} \approx 0.02\text{--}0.15$ (annualized variance). $\sqrt{RV_{ann}} \approx 15\text{--}40\%$ vol.

### Implied Variance (IV²)

**From SVI:** $\mathrm{atm\_iv}$ = total variance $w = IV^2 \times \tau$ ($\tau$ in years).

**Annualized:**

$IV^2_{ann} = \mathrm{atm\_iv} / \tau$

For 7D: $\tau = 7/365$, so $IV^2_{ann} = \mathrm{atm\_iv} \times (365/7)$.

**Typical magnitudes:** $IV^2_{ann} \approx 0.02\text{--}0.10$. $\sqrt{IV^2_{ann}} \approx 15\text{--}32\%$ vol.

### RV − IV²

**Same scale:** Both annualized variance. Typical range: $-0.05$ to $0.05$.

**Red flags:**
- $\mathrm{Mean}(\mathrm{RV} - \mathrm{IV}^2) > 0.5$ → unit mismatch
- $\sqrt{RV_{ann}} > 2$ (200% vol) → returns likely in percent

## Data Frequency Caveat

With **weekly** options data, returns are ~weekly (7–14 day gaps). For 7-day RV:
- We accumulate returns until span $\geq 7$ calendar days
- Often 1 return spans 7 days → RV from 1 observation (noisy)
- **Recommendation:** Use daily price data for robust RV.

## Verification

Run:
```bash
PYTHONPATH=. python scripts/rv_iv_unit_diagnostic.py
```
