# RV and IV² Units Convention

**Non-negotiable:** RV and IV² must be on the same scale for comparison.

**Canonical VRP specification:** [VRP_DEFINITIONS.md](VRP_DEFINITIONS.md) (horizons, SVI total variance vs ATM IV, annualization).

## Convention: Annualized Variance

We use **annualized variance** for both RV and IV².

### Returns
- **Units:** Decimal (0.01 = 1%)
- **Source:** `log_ret = diff(log(spx_price))`
- **Check:** `mean(|r|)` should be ~0.01. If ~1.0, returns are in percent (bug).

### Realized Variance (RV)

**Horizon variance (h days):**

$`RV_h = \sum r_i^2`$ for returns spanning $`\geq h`$ calendar days

- $`r_i`$ = log returns (decimal)
- Span = calendar days from first to last return in window

**Annualized:**

$`RV_{\text{ann}} = RV_h \times (365 / \text{span\_days})`$

**Typical magnitudes:** $`RV_{\text{ann}} \approx 0.02\text{--}0.15`$ (annualized variance). $`\sqrt{RV_{\text{ann}}} \approx 15\text{--}40\%`$ vol.

### Implied Variance (IV²)

**From SVI:** `atm_iv` = total variance $`w = IV^2 \times \tau`$ ($`\tau`$ in years).

**Annualized:**

$`IV^2_{\text{ann}} = \text{atm\_iv} / \tau`$

For 7D: $`\tau = 7/365`$, so $`IV^2_{\text{ann}} = \text{atm\_iv} \times (365/7)`$.

**Typical magnitudes:** $`IV^2_{\text{ann}} \approx 0.02\text{--}0.10`$. $`\sqrt{IV^2_{\text{ann}}} \approx 15\text{--}32\%`$ vol.

### RV − IV²

**Same scale:** Both annualized variance. Typical range: $-0.05$ to $0.05$.

**Red flags:**
- $`\text{Mean}(\text{RV} - \text{IV}^2) > 0.5`$ → unit mismatch
- $`\sqrt{RV_{\text{ann}}} > 2`$ (200% vol) → returns likely in percent

## Data Frequency Caveat

With **weekly** options data, returns are ~weekly (7–14 day gaps). For 7-day RV:
- We accumulate returns until span $`\geq 7`$ calendar days
- Often 1 return spans 7 days → RV from 1 observation (noisy)
- **Recommendation:** Use daily price data for robust RV.

## Verification

Run:
```bash
PYTHONPATH=. python scripts/rv_iv_unit_diagnostic.py
```
