# Implementation: Exact Incomplete Gamma Integration for Radial Push-Pull Transport

## Goal

Replace the 16-point Gauss-Legendre quadrature in `src/gwtransport/radial.py:_erf_mean_volume_radial` with an exact antiderivative based on incomplete gamma series, achieving machine precision for the erf integral in the radial push-pull well model.

## Background

The radial push-pull model computes a coefficient matrix where each entry requires the mean erf over an output bin:

```
mean_erf[i,j] = (1/ΔV) ∫_{V_lo}^{V_hi} erf( x(V) / (2·√(D_L·τ(V))) ) dV
```

## Mathematical Result

The integral has an **exact antiderivative** when the dispersion model uses a constant effective diffusivity per cell. The key insight is to match the linear model's approach (`diffusion.py`):

### Dispersion model (constant D_L per cell)

For each injection edge j, pre-compute:
```
scale_j  = N · π · h · n · R           (radial scale factor for this layer)
r_max_j  = √((V_max_j - V_j) / scale_j)  (max radius parcel reaches)
L_j      = 2 · r_max_j                 (total streamline length: out + back)
RT_j     = residence time for edge j    (time from injection to full extraction)
D_L_j    = D_m + α_L · L_j / RT_j      (effective longitudinal dispersion coefficient)
```

`L_j` is a **property of the streamline** (constant per injection edge), not a function of V.

Then the diffusion-time product is:
```
Σ(V) = D_L_j · τ(V)
```
where `τ(V) = t(V) - t_j` is linear in V within each constant-flow sub-interval.

### Erf argument

With `u = |V_j - V|`:
```
x(V) = sign(V_j - V) · √(u / S)
erf_arg = x / (2·√Σ) = sign · √(u / (4·S·D_L_j·τ))
```

Within a constant-flow extraction bin where `τ = τ₀ ± u/|Q|` (linear in u):
```
erf_arg² = p · u / (q ± u)
```
where `p = |Q| / (4·S·D_L_j)` and `q = τ₀·|Q|` are **constants** per extraction bin.

### Exact antiderivatives

**V < V_j** (erf > 0, τ increasing with u, uses q + u form):
```
∫ erf(√(p·u/(q+u))) du = G(u)
G(u) = (q+u)·erf(√(pu/(q+u))) - 2q·√(p/π)·K(p, √(u/(q+u)))
```

**V > V_j** (erf < 0, τ decreasing with u, uses q - u form):
```
∫ erf(√(p·u/(q-u))) du = F(u)
F(u) = -(q-u)·erf(√(pu/(q-u))) + 2q·√(p/π)·J(p, √(u/(q-u)))
```

where J and K are computed via **incomplete gamma series**:
```
J(p, a) = Σ_{k=0}^{N} (-1)^k · Γ(k+½) · P(k+½, p·a²) / (2·p^{k+½})   [alternating signs]
K(p, a) = Σ_{k=0}^{N}         Γ(k+½) · P(k+½, p·a²) / (2·p^{k+½})   [all positive]
```

Here `P(s, x) = gammainc(s, x)` is scipy's regularized lower incomplete gamma function. Both series converge to machine precision in ~50 terms. All `gammainc` calls are batched into a single vectorized call over both the series index k and the input elements.

### Verified precision

- Incomplete gamma series: errors ~10⁻¹² to 10⁻¹⁵ (machine precision)
- Works for all α_L ≥ 0, all p values, both V < V_j and V > V_j regimes

## Implementation

### Helper functions

1. `_JK_incomplete_gamma(p, a)` — compute both J(p, a) and K(p, a) simultaneously via the incomplete gamma series. All gammainc calls batched into one vectorized call.

2. `_erf_antideriv_approaching(u, p, q)` — antiderivative F(u) for V > V_j regime.

3. `_erf_antideriv_past(u, p, q)` — antiderivative G(u) for V < V_j regime.

### Main function

`_erf_mean_volume_radial` uses fully vectorized operations:
1. Pre-computes D_L_j per injection edge (constant per cell)
2. Flattens all (i, j) pairs into 1D arrays
3. Determines regime (V < V_j, V > V_j, straddle) via masks
4. Evaluates antiderivatives vectorized for each regime

### Edge cases

- `D_m = 0` and `α_L = 0`: erf = sign(x), no integration needed
- `D_L_j = 0`: erf = sign(x) (step function)
- `τ₀ = 0` (q = 0): G antiderivative gives u·erf(√p)
- `τ₀ < 0` (q < 0): edge j not yet injected, mean_erf = -1
- `u = 0`: F(0) = G(0) = 0

## Key Files

- `src/gwtransport/radial.py` — implementation
- `tests/src/test_radial.py` — tests including machine-precision verification against scipy.integrate.quad
