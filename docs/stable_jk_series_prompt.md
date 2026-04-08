# Research Prompt: Numerically Stable Evaluation of J(p, a) and K(p, a)

## Problem

I have two functions defined by incomplete gamma series that arise as antiderivatives of `erf(√(p·u/(q±u)))`:

```
J(p, a) = Σ_{k=0}^{∞} (-1)^k · Γ(k+½) · P(k+½, p·a²) / (2·p^{k+½})

K(p, a) = Σ_{k=0}^{∞}          Γ(k+½) · P(k+½, p·a²) / (2·p^{k+½})
```

where `P(s, x) = gammainc(s, x)` is the regularized lower incomplete gamma function (`scipy.special.gammainc`), `p > 0`, and `a ≥ 0`.

These appear in the exact antiderivatives of two integrals:

```
F(u) = ∫₀ᵘ erf(√(p·t/(q-t))) dt = -(q-u)·erf(√(pu/(q-u))) + 2q·√(p/π)·J(p, √(u/(q-u)))

G(u) = ∫₀ᵘ erf(√(p·t/(q+t))) dt = (q+u)·erf(√(pu/(q+u))) - 2q·√(p/π)·K(p, √(u/(q+u)))
```

## Numerical instability

The k-th term in the series has magnitude:

```
|term_k| = Γ(k+½) · P(k+½, p·a²) / (2·p^{k+½})
```

- For `k < p`: `Γ(k+½)/p^{k+½}` decreases (ratio `(k+½)/p < 1`), so terms shrink. The series converges.
- For `k > p`: `Γ(k+½)/p^{k+½}` grows factorially (ratio `(k+½)/p > 1`). Although `P(k+½, p·a²) → 0` for `k+½ ≫ p·a²`, the coefficient growth dominates when `a` is moderate-to-large (e.g., `a > 1`), causing the terms to grow exponentially.

**Example**: For `p = 8.42`, `a = 2.0` (`p·a² = 33.7`):
- `k=0`: term = 3.1e-1
- `k=10`: term = 1.1e-4 (minimum)
- `k=25`: term = 3.7e+0
- `k=49`: term = 4.3e+13

With 50 terms, the alternating series J suffers catastrophic cancellation: the true value is ~0.29, but the computed value is ~-3.4e13.

This occurs in practice when a large volume of background-concentration water is prepended to initialize the aquifer in a radial push-pull well model. The volume offset makes `a = √(u/(q-u))` large for injection edges far from the extraction zone.

## Parameter ranges in practice

- `p` ranges from ~0.1 to ~1000 (depends on flow rate, layer geometry, diffusivity)
- `a` ranges from 0 to ~100 (depends on volume ratios; large `a` = large volume offset)
- `p·a²` ranges from 0 to ~10⁶

## Requirements

- Machine precision (~1e-12 relative error)
- Vectorized evaluation over arrays of (p, a) pairs (typically 10⁴–10⁶ pairs)
- Implementable in Python with numpy/scipy (no symbolic math at runtime)

## Possible approaches

### Option A: Owen's T function or bivariate normal CDF

The integrals `∫ erf(√(p·u/(q±u))) du` may be expressible in terms of Owen's T function `T(h, a)` or the bivariate normal CDF `Φ₂(x, y; ρ)`. Owen's T is numerically stable and available in some libraries. Scipy has `scipy.special.owens_t` since version 1.7.

Can J(p, a) and K(p, a) — or equivalently the antiderivatives F(u) and G(u) — be rewritten in terms of Owen's T or bivariate normal CDF?

### Option B: Principled series truncation / asymptotic switching

The series converges for `k < p` and diverges for `k > p`. A principled approach:

1. Sum terms for `k = 0, 1, ..., k_max` where `k_max ≈ floor(p)` (before the ratio `(k+½)/p` exceeds 1).
2. For the remainder (`k > k_max`), use an asymptotic expansion or bound.

Alternatively, for large `a` (where `P(k+½, p·a²) ≈ 1` for all relevant `k`), the J and K series reduce to known closed forms:

```
J(p, ∞) = Σ_{k=0}^{∞} (-1)^k · Γ(k+½) / (2·p^{k+½})
K(p, ∞) = Σ_{k=0}^{∞}          Γ(k+½) / (2·p^{k+½})
```

These may have closed-form expressions (related to `erf(√p)` or similar). Can these be evaluated directly, with the incomplete gamma providing a correction for finite `a`?

### Option C: Change of integration variable

Instead of using the J/K series antiderivative, reformulate the original integral:

```
∫ erf(√(p·u/(q-u))) du
```

via a substitution (e.g., `v = u/(q-u)`, `v = √(u/(q-u))`, or `v = p·u/(q-u)`) that yields an integrand amenable to a different exact antiderivative or a rapidly converging series.

## Verification

Any proposed solution must match `scipy.integrate.quad` to relative tolerance 1e-11 for the test cases:

| p | q | u | Expected F(u) | Expected G(u) |
|---|---|---|---------------|---------------|
| 0.5 | 10 | 2.5 | quad(erf(√(p·t/(q-t))), 0, 2.5) | quad(erf(√(p·t/(q+t))), 0, 2.5) |
| 5 | 100 | 25 | ... | ... |
| 50 | 1000 | 250 | ... | ... |
| 500 | 5000 | 1250 | ... | ... |
| 8.42 | 1500 | 1200 | ~1123.6 | N/A |
| 8.42 | 200500 | 200100 | ~189890.7 | N/A |

The last two rows are the failing cases from the radial push-pull model with large volume offsets.
