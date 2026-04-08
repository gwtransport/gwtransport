# Research Prompt: Exact Radial Erf Integral for Push-Pull Well Transport

## Context

I'm building a radial push-pull well transport model for the `gwtransport` Python package. The model computes how injected tracer returns during extraction from a well in a layered aquifer. The key computation is an erf-based coefficient matrix where each entry requires integrating the error function over a time/volume bin.

## The integral I need to solve

For each (output_bin i, input_edge j) pair, I need the **mean erf over the output bin**:

```
mean_erf[i,j] = (1/ΔV) ∫_{V_lo}^{V_hi} erf( x(V) / (2·√(D_eff(V))) ) dV
```

where V is the cumulative volume (integration variable sweeping across the output bin), and:

### Spatial coordinate (radial geometry)
```
x(V) = sign(V_j - V) · √(|V_j - V| / S)
```
- `V_j`: cumulative volume at input edge j (known constant per cell)
- `S = N·π·h·n·R`: radial scale factor (known constant per layer)

### Effective diffusion product (varies with V)
```
D_eff(V) = D_m · τ(V) + α_L · L(V)
```

**Molecular diffusion term**: `D_m · τ(V)` where τ(V) = t(V) - t_j is the elapsed time since injection. Within a constant-flow sub-interval, t is linear in V:
```
t(V) = t_k + (V - V_k) / Q_k
```
so τ(V) = A + B·V (linear in V) where A, B are known constants per sub-interval.

**Dispersivity term**: `α_L · L(V)` where L is the total radial path length of the parcel:
```
L(V) = r_max + (r_max - r_current(V))
r_current(V) = √(max(V - V_j, 0) / S)
r_max = √((V_max - V_j) / S)    [known constant per cell]
```

## What I already know

### Case 1: Constant D_eff (known exact solution)

When D_eff doesn't vary with V (e.g., τ approximately constant across the bin), defining `a = 1/(4·D_eff·S)` and `u = |V_j - V|`, the integral reduces to `∫ erf(√(a·u)) du` which has the exact antiderivative:

```
F(u) = ((2au - 1) / (2a)) · erf(√(au)) + √(au) · exp(-au) / (a·√π)
```

I verified: F'(u) = erf(√(au)). This works perfectly when D_eff is constant.

### Case 2: Variable D_eff (solved via incomplete gamma series)

When D_eff varies with V, the integral becomes:

For molecular diffusion only (α_L = 0):
```
∫ erf( √(u/S) / (2·√(D_m·(A - u/Q))) ) du
= ∫ erf( √(Q·u / (4·S·D_m·(AQ - u))) ) du
```

This is `∫ erf(√(p·u/(q-u))) du` where p and q are constants. The exact antiderivative is:

```
F(u) = -(q-u)·erf(√(pu/(q-u))) + 2q·√(p/π)·J(p, √(u/(q-u)))
```

where J(p, a) is computed via the incomplete gamma series:
```
J(p, a) = Σ_{k=0}^{N} (-1)^k · Γ(k+½) · P(k+½, p·a²) / (2·p^{k+½})
```

Similarly, for `∫ erf(√(p·u/(q+u))) du`:
```
G(u) = (q+u)·erf(√(pu/(q+u))) - 2q·√(p/π)·K(p, √(u/(q+u)))
```

with K having all positive signs in the series.

## Solution

By using a constant D_L per injection edge (matching the linear model's approach), the full integral including dispersivity is solved exactly via the incomplete gamma series. See `docs/implement_owen_t_radial.md` for implementation details.

## Constraints

- The solution must be implementable in Python using numpy/scipy (no symbolic computation at runtime)
- Machine precision is desired (the existing package achieves this for the linear case)
- The integral must be computed for millions of (i,j) pairs, so computational efficiency matters
- The solution should handle edge cases: u=0, q-u→0, D_m=0, α_L=0
