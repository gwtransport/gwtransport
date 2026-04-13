"""
Utility functions for the Radial Push-Pull Well Transport Model.

Internal helper functions used by :mod:`gwtransport.radial`. These functions
build the coefficient matrices for pure advection and diffusion transport
in radial geometry.

This file is part of gwtransport which is released under AGPL-3.0 license.
See the ./LICENSE file or go to https://github.com/gwtransport/gwtransport/blob/main/LICENSE for full license details.
"""

import numpy as np
import numpy.typing as npt
from scipy.special import ive

from gwtransport.utils import partial_isin

# Argument above which ``scipy.special.ive(0, z)`` overflows to NaN.
# For ``z`` above this threshold we switch to the leading asymptotic
# term ``ive(0, z) ~ 1/sqrt(2*pi*z)``, which is accurate to better than
# ``1/(8z)`` (i.e. < 1e-8 relative for z > 1e7).
_IVE0_ASYMPTOTIC_THRESHOLD = 1.0e7


def _ive0_safe(z: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
    r"""Exponentially-scaled ``I_0`` with asymptotic continuation.

    Computes :math:`I_0(z)\,\mathrm{e}^{-z}` via
    :func:`scipy.special.ive`, falling back to the leading asymptotic
    :math:`1/\sqrt{2\pi z}` for arguments above the numerical overflow
    threshold of SciPy's implementation.

    Parameters
    ----------
    z : ndarray
        Non-negative Bessel argument.

    Returns
    -------
    ndarray
        :math:`I_0(z)\,\mathrm{e}^{-z}`, same shape as ``z``.
    """
    z = np.asarray(z, dtype=float)
    out = np.empty_like(z)
    small = z <= _IVE0_ASYMPTOTIC_THRESHOLD
    out[small] = ive(0, z[small])
    large = ~small
    out[large] = 1.0 / np.sqrt(2.0 * np.pi * z[large])
    return out


def _lifo_input_matrix(
    *,
    flow: npt.NDArray[np.floating],
    dt: npt.NDArray[np.floating],
) -> npt.NDArray[np.floating]:
    """Build exact LIFO stack matrix on the input tedges grid.

    Processes bins chronologically: injection bins push volume onto a stack,
    extraction bins pop from the stack top. Returns a per-extraction-bin
    weight matrix whose row ``i`` gives the normalized fraction of
    extraction bin ``i``'s volume attributed to each injection bin ``j``.

    This function has no knowledge of ``cout_tedges`` — it is pure physics
    on the input tedges grid.

    Parameters
    ----------
    flow : ndarray, shape (n,)
        Flow rate per input bin [m³/day]. Positive = injection,
        negative = extraction, zero = rest.
    dt : ndarray, shape (n,)
        Time bin widths [days].

    Returns
    -------
    w_input : ndarray, shape (n, n)
        LIFO weight matrix on the input tedges grid. ``w_input[i, j]`` is
        the fraction of the volume extracted in tedges bin ``i`` that comes
        from injection tedges bin ``j``. Row sums = 1 for extraction bins
        that fully recover injected volume, < 1 for over-extraction, and
        0 for non-extraction bins.
    """
    n = len(flow)
    extraction_volume = np.where(flow < 0, -flow * dt, 0.0)  # (n,) >= 0

    w_raw = np.zeros((n, n))
    stack: list[list[float]] = []

    for i in range(n):
        vol = flow[i] * dt[i]
        if vol > 0:
            stack.append([float(i), vol])
        elif vol < 0:
            vol_needed = -vol
            while vol_needed > 0 and stack:
                j_idx, vol_avail = stack[-1]
                consumed = min(vol_avail, vol_needed)
                w_raw[i, int(j_idx)] += consumed
                vol_needed -= consumed
                stack[-1][1] -= consumed
                if stack[-1][1] <= 0:
                    stack.pop()

    w_input = np.zeros((n, n))
    has_volume = extraction_volume > 0
    w_input[has_volume] = w_raw[has_volume] / extraction_volume[has_volume, np.newaxis]
    return w_input


def _resample_to_cout(
    *,
    w_input: npt.NDArray[np.floating],
    flow: npt.NDArray[np.floating],
    dt: npt.NDArray[np.floating],
    tedges_days: npt.NDArray[np.floating],
    cout_tedges_days: npt.NDArray[np.floating],
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.bool]]:
    """Flow-weighted resample from the input tedges grid to the cout grid.

    Projects an ``(n, n)`` weight matrix defined on the input tedges grid
    onto an ``(n_cout, n)`` weight matrix on the output ``cout_tedges``
    grid using a flow-weighted average. Uses
    :func:`gwtransport.utils.partial_isin` to compute the time-overlap
    fractions, so arbitrary alignment between ``tedges`` and
    ``cout_tedges`` is handled correctly.

    Parameters
    ----------
    w_input : ndarray, shape (n, n)
        Input-grid weight matrix (row-stochastic for extraction rows).
    flow : ndarray, shape (n,)
        Flow rate per input bin [m³/day].
    dt : ndarray, shape (n,)
        Time bin widths [days].
    tedges_days : ndarray, shape (n+1,)
        Input time edges in days from reference.
    cout_tedges_days : ndarray, shape (n_cout+1,)
        Output time edges in days from reference.

    Returns
    -------
    w : ndarray, shape (n_cout, n)
        Weight matrix on the ``cout_tedges`` grid. Row ``k`` equals the
        flow-weighted mean of ``w_input`` rows overlapping cout bin ``k``.
    has_extraction : ndarray, shape (n_cout,)
        Boolean mask indicating which output bins have extraction volume.
    """
    n = len(flow)
    n_cout = len(cout_tedges_days) - 1

    # Temporal overlap: overlap[i, k] = fraction of input bin i in cout bin k.
    overlap = partial_isin(bin_edges_in=tedges_days, bin_edges_out=cout_tedges_days)  # (n, n_cout)

    is_extraction = flow < 0
    extraction_volume = np.abs(flow) * dt * is_extraction  # (n,)
    ext_vol_overlap = extraction_volume[:, np.newaxis] * overlap  # (n, n_cout)
    total_ext = ext_vol_overlap.sum(axis=0)  # (n_cout,)
    has_extraction = total_ext > 0

    w = np.zeros((n_cout, n))
    # w[k, j] = Σ_i (ext_vol_overlap[i, k] * w_input[i, j]) / total_ext[k]
    w[has_extraction] = (ext_vol_overlap[:, has_extraction].T @ w_input) / total_ext[has_extraction, np.newaxis]

    return w, has_extraction


def _push_pull_advection_matrix(
    *,
    flow: npt.NDArray[np.floating],
    dt: npt.NDArray[np.floating],
    tedges_days: npt.NDArray[np.floating],
    cout_tedges_days: npt.NDArray[np.floating],
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.bool]]:
    """Build LIFO stack coefficient matrix for pure advection.

    Thin wrapper: builds the LIFO matrix on the input tedges grid via
    :func:`_lifo_input_matrix`, then flow-weighted resamples onto the
    ``cout_tedges`` grid via :func:`_resample_to_cout`.

    Parameters
    ----------
    flow : ndarray, shape (n,)
        Flow rate per bin [m³/day]. Positive = injection, negative = extraction.
    dt : ndarray, shape (n,)
        Time bin widths [days].
    tedges_days : ndarray, shape (n+1,)
        Time edges in days from reference.
    cout_tedges_days : ndarray, shape (n_cout+1,)
        Output time edges in days from reference.

    Returns
    -------
    weights : ndarray, shape (n_cout, n)
        Normalized weight matrix. Rows sum to 1 for output bins that fully
        recover injected volume, < 1 for over-extraction, and 0 for
        bins without extraction.
    has_extraction : ndarray, shape (n_cout,)
        Boolean mask indicating which output bins have extraction volume.
    """
    w_input = _lifo_input_matrix(flow=flow, dt=dt)
    return _resample_to_cout(
        w_input=w_input,
        flow=flow,
        dt=dt,
        tedges_days=tedges_days,
        cout_tedges_days=cout_tedges_days,
    )


def _push_pull_diffusion_kernel(
    *,
    w_lifo: npt.NDArray[np.floating],
    flow: npt.NDArray[np.floating],
    dt: npt.NDArray[np.floating],
    tedges_days: npt.NDArray[np.floating],
    v_cum: npt.NDArray[np.floating],
    v_max_after: npt.NDArray[np.floating],
    scale: float,
    molecular_diffusivity: float,
    longitudinal_dispersivity: float,
    retardation_factor: float,
) -> npt.NDArray[np.floating]:
    r"""Build a doubly-stochastic push-pull smear kernel on real injection bins.

    Given the exact LIFO attribution matrix ``w_lifo``, return a
    transition kernel ``K[a, b]`` over real injection bins (excluding
    the prepended background bin at index 0) that redistributes
    LIFO-attributed mass into neighbouring injection bins via radial
    diffusion. The kernel is designed so that the composed matrix
    ``w_smear[:, real] = w_lifo[:, real] @ K`` automatically satisfies
    **both** marginal constraints:

    * Row sums: ``Σ_j w_smear[i, j] = Σ_j w_lifo[i, j]`` for every row
      ``i``, so a constant injection concentration extracts unchanged.
    * Column mass: ``Σ_i ext_vol[i] * w_smear[i, j] = inj_vol[j]`` for
      every real injection bin ``j``, so mass is conserved exactly for
      arbitrary ``cin``.

    These two properties hold **by construction** — there is no
    iterative projection, no rescaling, and no band-aid normalization.
    The mechanism is detailed balance against the ``inj_vol`` measure:

    1. Pick a symmetric base density ``B[a, b] = B[b, a]``, derived
       from the axial 2D radial heat kernel expressed in the true
       radial coordinate ``r = √(V / scale)``:

       .. math::

           B(r_a, r_b; \sigma) = \exp\!\left(
               -\tfrac{r_a^2 + r_b^2}{2\sigma^2}
           \right)\,I_0\!\left(\tfrac{r_a\,r_b}{\sigma^2}\right)

       This is the density **per unit V** of axially symmetric 2D
       Brownian motion: the ``2 pi r`` Jacobian from the cylindrical
       area element is absorbed into ``dV/dr ∝ r``, leaving a density
       in V that is symmetric in ``(V_a, V_b)``. Symmetry is
       automatic because ``I_0`` is symmetric in its argument and
       ``r_a^2 + r_b^2`` is symmetric in ``a, b``. No method-of-images
       term is needed: the Bessel kernel is the fundamental solution
       of the radial diffusion equation with an implicit reflecting
       boundary at ``r = 0`` (the regular-at-the-origin Green's
       function for the radial Laplacian).

    2. Form a transition-probability matrix by multiplying with the
       target bin's V-measure ``dV[b] = inj_vol[b]``:

       .. math::

           q[a, b] = B[a, b]\,\mathrm{d}V[b]/Z

       where ``Z`` is chosen so that ``Σ_b q[a, b] ≤ 1`` for every
       ``a`` (``Z = max_a Σ_b B[a, b] dV[b]``).

    3. Absorb the sub-stochastic slack into the diagonal self-loop:

       .. math::

           K[a, b] = q[a, b] \quad (a \neq b)

           K[a, a] = 1 - \sum_{b \neq a} q[a, b]

       This gives ``Σ_b K[a, b] = 1`` exactly (row-stochastic) and,
       because ``B`` is symmetric, also satisfies detailed balance:
       ``K[a, b] \mathrm{d}V[a] = K[b, a] \mathrm{d}V[b]``. Summing
       over ``a`` gives the stationarity identity
       ``Σ_a \mathrm{d}V[a] K[a, b] = \mathrm{d}V[b]``, i.e. the
       ``inj_vol`` marginal is preserved by ``K``.

    The per-source variance ``sigma_sq[j]`` entering ``B`` is computed
    from the standard radial diffusion relation

    .. math::

        \sigma^2 = 2 (D_m/R)\,\tau_{\mathrm{eff}}(j)
                 + 2\,\alpha_L\,L_{\mathrm{path}}(j)

    using ``tau_eff(j)``, the column-mass-weighted elapsed time between
    injection at source ``j`` and its subsequent extraction, and
    ``L_path(j) = 2 r_max(j) - r_front(j)``, the retarded path length
    of the front reaching its maximum radius after the source injection
    and partially returning to the average extraction radius. The pair
    variance used in ``B[a, b]`` is
    ``sigma_sq[a, b] = 0.5 (sigma_sq[a] + sigma_sq[b])``, which is
    symmetric in ``(a, b)`` as required.

    Parameters
    ----------
    w_lifo : ndarray, shape (n, n)
        Exact LIFO attribution matrix on the input tedges grid.
    flow : ndarray, shape (n,)
        Flow rate per input bin [m³/day]. Index 0 is the prepended
        background bin; indices ≥ 1 are real tedges bins.
    dt : ndarray, shape (n,)
        Time bin widths [days].
    tedges_days : ndarray, shape (n+1,)
        Time edges in days from reference.
    v_cum : ndarray, shape (n+1,)
        Cumulative volume at time edges [m³].
    v_max_after : ndarray, shape (n+1,)
        Maximum cumulative volume reached from each edge onward
        [m³]. Used to compute the retarded path length.
    scale : float
        ``n_layers * pi * h * n * R`` [m^3/m] so that ``r = sqrt(V / scale)``.
    molecular_diffusivity : float
        Molecular diffusivity D_m [m²/day].
    longitudinal_dispersivity : float
        Longitudinal dispersivity alpha_L [m].
    retardation_factor : float
        Retardation factor R [-].

    Returns
    -------
    K_full : ndarray, shape (n, n)
        Full transition kernel. Rows and columns corresponding to
        non-injection bins (and the prepend bin at index 0) are the
        identity, so applying ``K_full`` leaves non-injection LIFO
        columns untouched. The real-injection sub-block satisfies
        row-stochasticity and ``inj_vol``-stationarity by construction.
    """
    n = flow.shape[0]
    inj_vol = np.where(flow > 0, flow * dt, 0.0)
    ext_vol = np.where(flow < 0, -flow * dt, 0.0)
    t_mid = 0.5 * (tedges_days[:-1] + tedges_days[1:])

    # Radial coordinates are measured **relative** to the end of the
    # prepended bin at index 0, so that the well screen at r = 0
    # coincides with the start of real injection. Without this
    # relative coordinate, the massive prepend volume would shift all
    # real bins to enormous absolute radii where the reflecting
    # boundary at r = 0 would become numerically invisible.
    v_rel = v_cum - v_cum[1]  # (n+1,), v_rel[1] = 0 at the well screen
    v_rel_mid = 0.5 * (v_rel[:-1] + v_rel[1:])
    v_rel_max_after = v_max_after - v_cum[1]

    # Kernel acts on real injection bins only (exclude prepend bin 0).
    real_mask = inj_vol > 0
    real_mask[0] = False
    real_idx = np.flatnonzero(real_mask)
    n_real = real_idx.size

    k_full = np.eye(n)
    if n_real == 0:
        return k_full

    # Per-source effective variance. Weight by the LIFO-assigned column
    # mass so that sources with short residence times contribute small
    # sigma and sources with long residence times contribute large
    # sigma, matching the physics.
    col_mass = ext_vol[:, np.newaxis] * w_lifo[:, real_idx]  # (n, n_real)
    col_total = col_mass.sum(axis=0)  # (n_real,)

    # Guard against zero-mass columns (over-extraction edge cases):
    # treat them as having tiny effective sigma so K becomes identity
    # on those bins.
    with np.errstate(divide="ignore", invalid="ignore"):
        weight = np.where(col_total > 0, col_mass / col_total, 0.0)  # (n, n_real)

    tau_per_source = t_mid[:, np.newaxis] - t_mid[real_idx][np.newaxis, :]  # (n, n_real)
    tau_eff = np.sum(weight * np.maximum(tau_per_source, 0.0), axis=0)  # (n_real,)

    v_rel_row_eff = np.sum(weight * v_rel_mid[:, np.newaxis], axis=0)  # (n_real,)
    v_source_lower = v_rel[real_idx]  # (n_real,)
    r_max_source = np.sqrt(np.maximum(v_rel_max_after[real_idx] - v_source_lower, 0.0) / scale)
    r_front_eff = np.sqrt(np.maximum(v_rel_row_eff - v_source_lower, 0.0) / scale)
    l_path = np.maximum(2.0 * r_max_source - r_front_eff, 0.0)

    sigma_sq_source = (2.0 * molecular_diffusivity / retardation_factor) * tau_eff + (
        2.0 * longitudinal_dispersivity
    ) * l_path  # (n_real,)

    # Sources with zero LIFO mass or zero elapsed time get the delta
    # (self) kernel; leave K_full as identity for those rows/columns.
    active = (col_total > 0) & (sigma_sq_source > 0)
    if not np.any(active):
        return k_full

    src_idx = real_idx[active]  # global indices
    dv_active = inj_vol[src_idx]  # (n_active,)
    r_source = np.sqrt(np.maximum(v_rel_mid[src_idx], 0.0) / scale)  # (n_active,)
    sigma_sq_active = sigma_sq_source[active]  # (n_active,)

    # Pair variance: s_pair(a, b) = 0.5 (s(a) + s(b)),  s := sigma^2,
    # which is symmetric in a, b. The pair variance enters the Bessel
    # heat kernel below; any pair-dependent prefactor (such as the 2D
    # Gaussian normalization ``1/(2 pi s)``) cancels in the
    # row-normalized transition kernel, so we omit it and rely on the
    # per-row ``Z_max`` normalization to bring the proposal into the
    # unit simplex.
    sigma_sq_pair = 0.5 * (sigma_sq_active[:, np.newaxis] + sigma_sq_active[np.newaxis, :])

    # Axial 2D radial heat kernel in V-coordinate, expressed through
    # the standard radial variable ``r = sqrt(V/scale)``:
    #
    #     f(V_a | V_b) ~ exp(-(r_a**2 + r_b**2)/(2 s)) * I_0(r_a r_b / s)
    #
    # with ``s = sigma_sq_pair``. This is the density **per unit V** of
    # axially symmetric 2D Brownian motion (the ``2 pi r`` Jacobian from
    # the cylindrical area element is absorbed into
    # ``dV/dr = 2 pi r * scale/(N*R)``, leaving the Bessel form above as
    # the V-space density up to a V-independent prefactor). The density
    # is symmetric in ``(V_a, V_b)`` because ``I_0`` is symmetric and
    # ``r_a**2 + r_b**2`` is symmetric, so it automatically enforces
    # detailed balance with respect to the Lebesgue measure in V (i.e.
    # the ``inj_vol`` measure when discretized).
    #
    # For numerical stability we use the identity
    #
    #     exp(-(r_a**2 + r_b**2)/(2 s)) * I_0(r_a r_b / s)
    #         = exp(-(r_a - r_b)**2/(2 s)) * ive(0, r_a r_b / s),
    #
    # where ``ive(0, z) = I_0(z)*exp(-z)`` is the exponentially-scaled
    # modified Bessel function of order zero (``scipy.special.ive``),
    # which stays finite for large arguments whereas ``I_0`` overflows.
    r_a = r_source[:, np.newaxis]
    r_b = r_source[np.newaxis, :]
    arg = r_a * r_b / sigma_sq_pair  # (n_active, n_active)
    exponent = -((r_a - r_b) ** 2) / (2.0 * sigma_sq_pair)
    # Symmetric in (a, b) because both factors are symmetric in a, b.
    b_mat = np.exp(exponent) * _ive0_safe(arg)

    # Build the kernel from the symmetric base density by a single
    # global normalization:
    #
    #     K[a, b] = B[a, b] · dV[b] / Z_max         (a ≠ b)
    #     K[a, a] = 1 - Σ_{b ≠ a} K[a, b]
    #
    # where ``Z_max = max_a (Σ_b B[a, b] · dV[b])`` is the largest
    # unnormalized row sum. The global ``Z_max`` guarantees that
    # every row has total mass ``≤ 1`` before the self-loop, so the
    # slack goes into a non-negative diagonal.
    #
    # Because ``B[a, b]`` is symmetric in ``(a, b)``, the off-diagonal
    # kernel satisfies
    #
    #     K[a, b] · dV[a] = B[a, b] · dV[a] · dV[b] / Z_max
    #                    = K[b, a] · dV[b],
    #
    # which is detailed balance with respect to the ``inj_vol``
    # measure. Summing over ``a`` gives the stationarity identity
    # ``Σ_a dV[a] · K[a, b] = dV[b]``, so ``inj_vol`` is the
    # stationary distribution and column mass is preserved exactly.
    #
    # The global ``Z_max`` scaling (i) is smooth and (ii) reaches the
    # fully-mixed limit ``K[a, a] = dV[a] / V_total`` when the pair
    # variance is large enough to saturate the kernel, so the
    # breakthrough peak is monotone in the diffusion coefficients
    # throughout the relevant parameter range.
    dv_row = dv_active[np.newaxis, :]
    row_norm = (b_mat * dv_row).sum(axis=1)  # (n_active,)
    z_max = row_norm.max()
    k_sub = (b_mat * dv_row) / z_max

    # Overwrite the diagonal with the self-loop probability. The
    # diagonal entries computed above from ``B·dV / Z_max`` are
    # non-zero and must be cleared first.
    np.fill_diagonal(k_sub, 0.0)
    off_diag_sum = k_sub.sum(axis=1)
    np.fill_diagonal(k_sub, 1.0 - off_diag_sum)

    # Insert back into the full-size kernel at the active indices.
    # Clear the identity diagonal at those positions before writing.
    k_full[src_idx, src_idx] = 0.0
    k_full[np.ix_(src_idx, src_idx)] = k_sub

    # Inactive real bins (zero sigma or zero LIFO mass): keep identity
    # diagonal so ``w_lifo @ k_full`` leaves them in place.
    return k_full


def _push_pull_diffusion_matrix(
    *,
    flow: npt.NDArray[np.floating],
    tedges_days: npt.NDArray[np.floating],
    cout_tedges_days: npt.NDArray[np.floating],
    v_cum: npt.NDArray[np.floating],
    layer_height: float,
    porosity: float,
    retardation_factor: float,
    n_layers: int,
    molecular_diffusivity: float,
    longitudinal_dispersivity: float,
    v_max_after: npt.NDArray[np.floating],
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.bool]]:
    """Build coefficient matrix with radial diffusion for one layer.

    Assembles the weight matrix as the composition:

    1. Exact LIFO attribution on the input tedges grid via
       :func:`_lifo_input_matrix`.
    2. Radial diffusion smear via :func:`_push_pull_diffusion_kernel`.
       The smear is a doubly-stochastic transition kernel on real
       injection bins built from the axial 2D radial heat kernel
       (Bessel form) expressed in the true radial coordinate; by
       construction it preserves both row sums (so constant ``cin``
       extracts unchanged) and the ``ext_vol``-weighted column mass
       (so total mass is conserved for arbitrary ``cin``).
    3. Flow-weighted resample onto ``cout_tedges`` via
       :func:`_resample_to_cout`, which handles arbitrary grid alignment.

    Parameters
    ----------
    flow : ndarray, shape (n,)
        Flow rate per bin [m³/day].
    tedges_days : ndarray, shape (n+1,)
        Time edges in days.
    cout_tedges_days : ndarray, shape (n_cout+1,)
        Output time edges in days.
    v_cum : ndarray, shape (n+1,)
        Cumulative volume at time edges [m³].
    layer_height : float
        Height of a single horizontal streamtube flowing to the well
        screen [m].
    porosity : float
        Porosity n [-].
    retardation_factor : float
        Retardation factor R [-].
    n_layers : int
        Number of layers N.
    molecular_diffusivity : float
        Molecular diffusivity D_m [m²/day].
    longitudinal_dispersivity : float
        Longitudinal dispersivity alpha_L [m].
    v_max_after : ndarray, shape (n+1,)
        Maximum cumulative volume from each edge onward.

    Returns
    -------
    weights : ndarray, shape (n_cout, n)
        Normalized weight matrix on the cout_tedges grid.
    has_extraction : ndarray, shape (n_cout,)
        Boolean mask indicating which output bins have extraction volume.
    """
    dt = np.diff(tedges_days)
    scale = n_layers * np.pi * layer_height * porosity * retardation_factor

    # Step 1: LIFO on the input tedges grid
    w_lifo = _lifo_input_matrix(flow=flow, dt=dt)  # (n, n)

    # Pure advection fast path: skip the smear entirely.
    if molecular_diffusivity == 0.0 and longitudinal_dispersivity == 0.0:
        return _resample_to_cout(
            w_input=w_lifo,
            flow=flow,
            dt=dt,
            tedges_days=tedges_days,
            cout_tedges_days=cout_tedges_days,
        )

    # Step 2: build the doubly-stochastic radial smear kernel and
    # compose it with LIFO. Both row sums and column mass are preserved
    # by the kernel's construction (see :func:`_push_pull_diffusion_kernel`).
    k = _push_pull_diffusion_kernel(
        w_lifo=w_lifo,
        flow=flow,
        dt=dt,
        tedges_days=tedges_days,
        v_cum=v_cum,
        v_max_after=v_max_after,
        scale=scale,
        molecular_diffusivity=molecular_diffusivity,
        longitudinal_dispersivity=longitudinal_dispersivity,
        retardation_factor=retardation_factor,
    )
    w_smear = w_lifo @ k

    # Step 3: flow-weighted resample onto the cout grid
    return _resample_to_cout(
        w_input=w_smear,
        flow=flow,
        dt=dt,
        tedges_days=tedges_days,
        cout_tedges_days=cout_tedges_days,
    )
