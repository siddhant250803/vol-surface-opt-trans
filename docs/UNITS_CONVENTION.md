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
```
RV_h = Σ r_i²   for returns spanning ≥ h calendar days
```
- `r_i` = log returns (decimal)
- Span = calendar days from first to last return in window

**Annualized:**
```
RV_ann = RV_h × (365 / span_days)
```

**Typical magnitudes:** RV_ann ≈ 0.02–0.15 (annualized variance). √RV_ann ≈ 15–40% vol.

### Implied Variance (IV²)

**From SVI:** `atm_iv` = total variance w = IV² × τ (τ in years).

**Annualized:**
```
IV²_ann = atm_iv / τ
```
For 7D: τ = 7/365, so IV²_ann = atm_iv × (365/7).

**Typical magnitudes:** IV²_ann ≈ 0.02–0.10. √IV²_ann ≈ 15–32% vol.

### RV − IV²

**Same scale:** Both annualized variance. Typical range: −0.05 to 0.05.

**Red flags:**
- Mean(RV − IV²) > 0.5 → unit mismatch
- √RV_ann > 2 (200% vol) → returns likely in percent

## Data Frequency Caveat

With **weekly** options data, returns are ~weekly (7–14 day gaps). For 7-day RV:
- We accumulate returns until span ≥ 7 calendar days
- Often 1 return spans 7 days → RV from 1 observation (noisy)
- **Recommendation:** Use daily price data for robust RV.

## Verification

Run:
```bash
PYTHONPATH=. python scripts/rv_iv_unit_diagnostic.py
```
