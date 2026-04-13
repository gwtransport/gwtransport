.. _radial-kernel-derivation:

Radial Push-Pull Kernel Derivation
==================================

This page derives from first principles the transition kernel used
internally by :mod:`gwtransport.radial` to model radial diffusion in a
push-pull well test. The goal is a weight matrix that

1. exactly preserves mass for an arbitrary input concentration, and
2. reduces to the pure advective push-pull attribution when the
   diffusion coefficients vanish,

both as mathematical identities rather than as numerical observations.

Motivation
----------

A push-pull tracer test alternates injection and extraction at a single
well screen. Without diffusion the model is purely advective: each
injected parcel returns in strict last-in-first-out order, and recovery
is perfect regardless of aquifer heterogeneity. With molecular
diffusion or longitudinal dispersivity, parcels sampled at the well
screen during extraction originate from a blurred distribution around
their advective return position. The discrete analogue is a transition
kernel acting on injection bins.

Earlier implementations built this kernel from a 1D Gaussian with
method-of-images reflection at the well screen. That construction
leaked mass at the boundary and required an iterative
column-mass projection (``_project_to_lifo_column_mass``) to restore
conservation. The derivation below replaces it with a physically
motivated kernel that satisfies both invariants by construction.

Radial coordinate
-----------------

Consider one horizontal streamtube of height :math:`h`, porosity
:math:`n`, and retardation :math:`R`, contributing equally with
:math:`N-1` other streamtubes. For a tracer parcel injected at
cumulative volume :math:`V_j`, the radial distance from the well
screen is

.. math::

    r_j = \sqrt{\frac{V_j}{N\,\pi\,h\,n\,R}}
        = \sqrt{V_j / \mathrm{scale}}, \qquad
    \mathrm{scale} := N\,\pi\,h\,n\,R.

The bin-edge cumulative volume :math:`V` is referenced to the start of
real injection, so that :math:`r = 0` coincides with the well screen.

Heat kernel in two dimensions
-----------------------------

A tracer parcel undergoes effectively two-dimensional Brownian motion
in the aquifer layer (the vertical coordinate is folded into the
streamtube average). The axially symmetric heat kernel on the plane,
given a source at radius :math:`r'` and elapsed variance
:math:`\sigma^2`, is

.. math::

    G_{2\mathrm{D}}(r, r'; \sigma^2)
      = \frac{1}{2\pi \sigma^2}
        \exp\!\left(-\frac{r^2 + r'^2}{2\sigma^2}\right)
        I_0\!\left(\frac{r\,r'}{\sigma^2}\right),

where :math:`I_0` is the modified Bessel function of the first kind of
order zero. This is the fundamental solution of the radial diffusion
equation that is regular at the origin: it is the natural replacement
for the 1D Gaussian with method-of-images reflection, and no explicit
image term is needed because the Bessel kernel already satisfies the
reflecting boundary condition :math:`\partial_r G|_{r=0} = 0`.

Pushing :math:`G_{2\mathrm{D}}` forward into volume coordinates
:math:`V = \mathrm{scale} \cdot r^2` uses the area element
:math:`\mathrm{d}A = 2\pi r\,\mathrm{d}r` and the fact that
:math:`\mathrm{d}V = 2\pi r \cdot \mathrm{scale}/(N R)\,\mathrm{d}r`,
which together absorb the :math:`2\pi r` factor into
:math:`\mathrm{d}V`. The density **per unit V** is therefore
proportional to

.. math::

    B(V_a, V_b; \sigma^2)
      \;\propto\; \exp\!\left(
          -\frac{r_a^2 + r_b^2}{2\sigma^2}
      \right) I_0\!\left(\frac{r_a\,r_b}{\sigma^2}\right),

which is **symmetric** in :math:`(V_a, V_b)`:
:math:`B(V_a, V_b) = B(V_b, V_a)`, because both
:math:`r_a^2 + r_b^2` and :math:`r_a\,r_b` are symmetric in
:math:`a, b`. Symmetry is the key ingredient for detailed balance
below.

Variance model
--------------

The variance :math:`\sigma^2(j)` entering the kernel for a source at
bin :math:`j` combines molecular diffusion (proportional to an
effective residence time :math:`\tau_\mathrm{eff}`) and longitudinal
mechanical dispersion (proportional to a retarded path length
:math:`L_\mathrm{path}`):

.. math::

    \sigma^2(j) = 2\,\frac{D_m}{R}\,\tau_\mathrm{eff}(j)
                + 2\,\alpha_L\,L_\mathrm{path}(j).

Here :math:`\tau_\mathrm{eff}(j)` is the push-pull-weighted elapsed
time between injection at bin :math:`j` and its subsequent extraction,
computed from the exact attribution matrix on the input grid, and

.. math::

    L_\mathrm{path}(j) = 2\,r_\mathrm{max}(j) - r_\mathrm{front}(j)

is the out-and-back path length of the front reaching its maximum
radius and partially returning to the average extraction radius. Both
:math:`r_\mathrm{max}` and :math:`r_\mathrm{front}` are measured in the
relative coordinate anchored at the well screen.

The pair variance used in :math:`B` is the arithmetic mean

.. math::

    \sigma^2(a, b) = \tfrac{1}{2}\bigl(\sigma^2(a) + \sigma^2(b)\bigr),

which is manifestly symmetric in :math:`a, b` and preserves the
symmetry of :math:`B`.

Detailed balance construction
-----------------------------

Given the discrete injection volumes :math:`\mathrm{d}V[b] :=
\mathrm{inj\_vol}[b]`, build the transition probability

.. math::

    K[a, b] = B[a, b]\,\mathrm{d}V[b] / Z_\mathrm{max}
      \qquad (a \neq b),

where

.. math::

    Z_\mathrm{max} = \max_a \sum_b B[a, b]\,\mathrm{d}V[b].

Fill the diagonal with the slack

.. math::

    K[a, a] = 1 - \sum_{b \neq a} K[a, b].

By construction the rows of :math:`K` sum to one, and the diagonal is
non-negative because each unnormalized row sum is bounded above by
:math:`Z_\mathrm{max}`.

Because :math:`B[a, b] = B[b, a]`, the off-diagonal kernel satisfies
**detailed balance** against the :math:`\mathrm{inj\_vol}` measure:

.. math::

    K[a, b]\,\mathrm{d}V[a]
      = \frac{B[a, b]\,\mathrm{d}V[a]\,\mathrm{d}V[b]}{Z_\mathrm{max}}
      = K[b, a]\,\mathrm{d}V[b].

Summing this identity over :math:`a` and adding the diagonal term
yields the stationarity relation

.. math::

    \sum_a \mathrm{d}V[a]\,K[a, b] = \mathrm{d}V[b],

i.e. :math:`\mathrm{inj\_vol}` is a stationary distribution of
:math:`K`. Composing :math:`K` with the exact push-pull attribution
matrix :math:`W_\mathrm{PP}` on the input grid therefore preserves
the :math:`\mathrm{ext\_vol}`-weighted column mass:

.. math::

    \sum_i \mathrm{ext\_vol}[i]\,(W_\mathrm{PP} K)[i, b]
      = \sum_i \mathrm{ext\_vol}[i]\,W_\mathrm{PP}[i, b]
      = \mathrm{inj\_vol}[b],

the second equality being the mass-conservation property of the pure
push-pull attribution.

Limiting behaviour
------------------

* **Zero diffusion.** At :math:`\sigma^2 \to 0` the Bessel kernel
  concentrates on the diagonal: :math:`B[a, b] \to 0` for
  :math:`a \neq b`, so :math:`K \to I` and the composition reduces to
  the exact push-pull attribution on the input grid.

* **Large diffusion.** As the pair variance grows, the kernel tends
  towards the fully mixed limit

  .. math::

      K[a, b] \to \mathrm{d}V[a] / V_\mathrm{total},

  which distributes mass across injection bins proportionally to the
  :math:`\mathrm{inj\_vol}` measure. The breakthrough peak is
  monotone in the diffusion coefficients throughout the interior of
  the parameter range.

Numerical notes
---------------

For :math:`r_a r_b / \sigma^2 \gtrsim 10^7`, SciPy's
:func:`~scipy.special.ive` (exponentially scaled :math:`I_0`)
overflows. The implementation switches to the leading asymptotic term

.. math::

    I_0(z)\,e^{-z} \;\sim\; \frac{1}{\sqrt{2\pi z}}
      \qquad (z \to \infty),

which is accurate to better than :math:`1/(8 z) < 10^{-8}` in the
affected regime. The transition threshold is defined by
``_IVE0_ASYMPTOTIC_THRESHOLD`` in :mod:`gwtransport.radial_utils`.

See also
--------

* :mod:`gwtransport.radial` for the public forward and inverse API.
* :func:`gwtransport.radial.push_pull` for the forward solver.
* :ref:`concept-residence-time` for background on residence times.
