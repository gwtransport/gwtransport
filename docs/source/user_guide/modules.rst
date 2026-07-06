Choosing a transport model
==========================

``gwtransport`` covers five transport settings. Choose the module by how the
solute enters the aquifer.

- **Infiltrating surface water**, flowing through to a well (bank filtration)

  - :mod:`~gwtransport.advection` --- advection and macrodispersion
  - :mod:`~gwtransport.diffusion` --- also microdispersion and molecular diffusion

- **Areal rainfall recharge** to a pumping well (optionally with upstream surface water)

  - :mod:`~gwtransport.recharge` --- advection; rainfall mixed vertically, giving an
    exponential residence-time distribution

- **Areal deposition** from the surface

  - :mod:`~gwtransport.deposition` --- advection; areal source mixed vertically over
    the height of the aquifer

- **Water injected at a well**, then recovered (push-pull / ASR)

  - :mod:`~gwtransport.radial_asr` --- radial advection with microdispersion,
    molecular diffusion, and optional steady regional background flow (drift)

Shared building blocks: :mod:`~gwtransport.residence_time`,
:mod:`~gwtransport.logremoval`, and :mod:`~gwtransport.gamma`.


Capability matrix
-----------------

.. list-table::
   :header-rows: 1
   :widths: 16 10 9 13 12 12 8 8

   * - Module
     - Geometry
     - Advection
     - Molecular diffusion
     - Microdispersion
     - Macrodispersion
     - Forward
     - Inverse
   * - :mod:`~gwtransport.advection`
     - any
     - ✓
     - –
     - –
     - ✓
     - ✓
     - ✓
   * - :mod:`~gwtransport.diffusion`
     - orthogonal
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
   * - :mod:`~gwtransport.deposition`
     - any
     - ✓
     - –
     - –
     - –
     - ✓
     - ✓
   * - :mod:`~gwtransport.recharge`
     - any
     - ✓
     - –
     - –
     - –
     - ✓
     - –
   * - :mod:`~gwtransport.radial_asr`
     - radial
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓

Geometry is fixed only where within-streamtube dispersion (microdispersion or
molecular diffusion) is modeled --- :mod:`~gwtransport.diffusion`
(Cartesian Kreft–Zuber) and :mod:`~gwtransport.radial_asr` (radial Airy/Whittaker).
Macrodispersion (the spread across pore volumes) lives in the volume coordinate and
needs no geometry, so the advective modules are geometry-agnostic. See
:ref:`concept-dispersion-scales` for macrodispersion versus microdispersion. Only
:mod:`~gwtransport.radial_asr` supports a reversing (signed) flow schedule; the other modules
assume one-directional flow.


Conceptual models
-----------------

advection
~~~~~~~~~

Water infiltrates and is transported in parallel along multiple aquifer pore
volumes to extraction. For each aquifer pore volume, transport is 1D advection
with linear or non-linear sorption; there is no microdispersion or molecular
diffusion, while the spread across aquifer pore volumes provides macrodispersion.
Forward and backward modeling are supported. No assumption is made about whether
the flow is radial or orthogonal.

:Reactions: linear retardation (:ref:`concept-retardation-factor`) and non-linear
   Freundlich/Langmuir sorption (:ref:`concept-nonlinear-sorption`).
:Limitations: non-linear sorption is forward-only.
:Examples: :doc:`/examples/01_Aquifer_Characterization_Temperature`,
   :doc:`/examples/08_bank_filtration_timflow`,
   :doc:`/examples/10_Advection_with_non_linear_sorption`.

diffusion
~~~~~~~~~

Water infiltrates and is transported in parallel along multiple aquifer pore
volumes to extraction. For each aquifer pore volume, transport is 1D advection
with microdispersion, molecular diffusion, and linear sorption; the spread across
aquifer pore volumes provides macrodispersion. Forward and backward modeling are
supported. The flow is assumed orthogonal.

:Reactions: linear retardation (:ref:`concept-retardation-factor`).
:Limitations: geometry fixed to orthogonal; no non-linear sorption. The
   :mod:`~gwtransport.diffusion_fast` and :mod:`~gwtransport.diffusion_fast_fast`
   variants share this conceptual model with faster (and, for ``_fast_fast``,
   approximate) implementations.
:Examples: :doc:`/examples/01_Aquifer_Characterization_Temperature`.

deposition
~~~~~~~~~~

Areal deposition supplies mass to the groundwater, mixed instantaneously over the
height of the aquifer. The aquifer has a constant thickness with a finite pore
volume; water with zero concentration infiltrates at one end and is extracted at
the other, whether the flow is radial or orthogonal. Transport is 1D advection
with linear sorption; there is no microdispersion, molecular diffusion, or
macrodispersion. Forward and backward modeling are supported.

:Reactions: areal deposition source; linear retardation
   (:ref:`concept-retardation-factor`).
:Limitations: models a source, not removal; a single (finite) pore volume, so no
   macrodispersion.
:Examples: :doc:`/examples/04_Deposition_Analysis_Bank_Filtration`.

recharge
~~~~~~~~

Concentration at extraction has two sources. 1) Water infiltrates and is
transported through an aquifer with constant thickness to extraction. 2) During
transport, rainfall is mixed instantaneously over the height of the aquifer. In an
unbounded aquifer all extracted water originates as recharge. Transport is
advective with linear sorption; there is no microdispersion, molecular diffusion,
or macrodispersion. Only forward modeling is supported. No assumption is made about
whether the flow is radial or orthogonal.

:Behavior: the areal entry and vertical mixing produce an exponential
   residence-time distribution (planform-independent in the unbounded model).
:Reactions: linear retardation (:ref:`concept-retardation-factor`).
:Limitations: forward-only; the bounded model uses a 1D upstream boundary.
:Examples: :doc:`/examples/12_Bank_Filtration_Rainwater_Fraction`.

radial_asr
~~~~~~~~~~

Water is injected in an infinite aquifer at a single fully-penetrating well and
later recovered at the same well under a signed flow schedule (push-pull / ASR).
Transport is radial advection with microdispersion, molecular diffusion, and linear
sorption; the spread of velocities across the well screen provides macrodispersion.
A steady uniform regional background flow (``regional_flux``) may be superimposed,
making the stored water drift between injection and recovery; ``regional_flux = 0``
(the default) is purely radial. Forward and backward modeling are supported.

:Reactions: linear retardation (:ref:`concept-retardation-factor`).
:Limitations: a single well (no two-point / doublet geometry); no non-linear
   sorption; regional drift is limited to the slow-drift envelope (the stored
   plume must stay well inside the stagnation radius; the engine raises a
   ``ValueError`` beyond it).
:Examples: :doc:`/examples/13_Aquifer_Storage_Recovery`.

.. _concept-drift-envelope:

Feasibility envelope under regional drift
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The practical reach of the slow-drift envelope is easiest to judge from a worked scenario: one storage
cycle per year for 1--20 years, each year 90 days injection, 90 days storage, 90 days recovery, and
90 days idle, discretized in 30-day bins. The well pumps ``Q = +-100`` m³/day, so every season stores and
recovers 9 000 m³ in an aquifer of thickness ``b = 10`` m and porosity ``n = 0.3``; the seasonal bubble
radius is ``R_b = sqrt(r_w² + V/(pi b n)) ≈ 31`` m around an ``r_w = 0.5`` m well, with microdispersivity
``alpha_L = 0.5`` m. The well strength ``A_0 = Q/(2 pi b n) ≈ 5.3`` m²/day makes regional drift seepage
velocities of 6 / 12 / 18 m/yr correspond to ``eps = v_d R_b / A_0 ≈ 0.10 / 0.19 / 0.29``. The injected
deviation is 1 during every injection season. Cells report the recovery efficiency of the **final** year
(the flow-weighted mean extracted deviation) and, in parentheses, the drift-induced loss relative to a
zero-drift run of the same length; ``--`` marks combinations the honesty guards refuse (``ValueError``:
the plume, including its accumulated storage-season displacement, approaches the stagnation radius or
outruns the azimuthal truncation).

Conservative solute (``R = 1``, effective pore diffusion ``D_m = 8.6e-5`` m²/day ≈ ``1e-9`` m²/s):

.. list-table::
   :header-rows: 1
   :widths: 10 14 22 22 22

   * - Years
     - No drift
     - 6 m/yr
     - 12 m/yr
     - 18 m/yr
   * - 1
     - 0.883
     - 0.865 (-0.018)
     - 0.822 (-0.061)
     - 0.765 (-0.118)
   * - 2
     - 0.914
     - 0.886 (-0.028)
     - 0.829 (-0.085)
     - --
   * - 3
     - 0.928
     - 0.893 (-0.035)
     - 0.830 (-0.098)
     - --
   * - 5
     - 0.942
     - 0.897 (-0.045)
     - 0.830 (-0.112)
     - --
   * - 10
     - 0.957
     - 0.898 (-0.059)
     - --
     - --
   * - 15
     - 0.963
     - 0.898 (-0.066)
     - --
     - --
   * - 20
     - 0.967
     - 0.898 (-0.070)
     - --
     - --

Temperature (thermal retardation ``R = rho c_b / (n rho_w c_w) ≈ 2.2``; thermal diffusivity
``lambda_b / (n rho_w c_w) = 0.172`` m²/day for a bulk conductivity ``lambda_b = 2.5`` W/m/K):

.. list-table::
   :header-rows: 1
   :widths: 10 14 22 22 22

   * - Years
     - No drift
     - 6 m/yr
     - 12 m/yr
     - 18 m/yr
   * - 1
     - 0.756
     - 0.751 (-0.004)
     - 0.740 (-0.016)
     - --
   * - 2
     - 0.807
     - 0.801 (-0.007)
     - 0.783 (-0.024)
     - --
   * - 3
     - 0.831
     - 0.823 (-0.009)
     - 0.800 (-0.031)
     - --
   * - 5
     - 0.856
     - 0.844 (-0.012)
     - --
     - --
   * - 10
     - 0.883
     - 0.863 (-0.019)
     - --
     - --
   * - 15
     - 0.895
     - 0.870 (-0.024)
     - --
     - --
   * - 20
     - 0.903
     - 0.874 (-0.029)
     - --
     - --

Reading the tables:

- A conservative solute tolerates 6 m/yr of drift over at least 20 years and 12 m/yr up to about
  5 years; 18 m/yr survives only a single cycle. Temperature is one step tighter at the strong end (the
  thermal plume is wider), but loses 2.5--4x less to drift at the same rate: thermal retardation halves
  the plume's drift displacement (``v_d / R``). Seasonal heat storage at modest regional flow sits well
  inside the envelope.
- The drift loss grows with record length at a fixed drift rate: without drift the system re-captures
  its unrecovered carryover year after year, while drift removes that carryover permanently (solute at
  6 m/yr: -0.018 after one cycle, -0.070 after twenty years).
- The refused combinations genuinely need a spatially resolved numerical transport model -- the engine
  raises rather than extrapolating beyond its envelope.
- This schedule is pessimistic for feasibility: the well is idle half of every year, and it is exactly
  the idle-season drift that accumulates. Schedules with shorter storage and idle periods stay inside
  the envelope at correspondingly higher drift rates.

Computed with ``n_quad = 60``, automatic azimuthal truncation (``M = 2--8`` across the table), and the
engine-internal Laplace-inversion settings ``n_terms = 24``, ``tol = 1e-8`` (the public API fixes these
at its defaults; spot checks through the public API reproduce the displayed values at the shown
precision). Absolute values carry ≲ 0.002 discretization at the longest records;
differences within a row are consistent to better than that. The zero-drift references use the drift
engine at negligible ``regional_flux`` so that both columns share the same rest-phase kernel.


Building blocks
---------------

These modules are not transport scenarios but the shared layer underneath.

- :mod:`~gwtransport.residence_time` --- flow-weighted residence time
  (:ref:`concept-residence-time`); see
  :doc:`/examples/02_Residence_Time_Analysis`.
- :mod:`~gwtransport.logremoval` --- first-order decay (e.g. pathogen inactivation)
  applied to a residence time; see
  :doc:`/examples/03_Pathogen_Removal_Bank_Filtration`.
- :mod:`~gwtransport.gamma` --- two-parameter aquifer pore volume distribution
  (:ref:`concept-gamma-distribution`).


Scope and not-yet-available
---------------------------

- No two-point / doublet radial transport: :mod:`~gwtransport.radial_asr` injects
  and recovers at a single well, not between an injection well and a separate
  observation or extraction point.
- No finite-radial bounded recharge: the bounded :mod:`~gwtransport.recharge` model
  uses a straight (1D) upstream boundary, not a circular capture-zone boundary.
- Package-wide: streamlines are independent (no transverse mixing,
  :ref:`assumption-no-transverse-mixing`); apart from the steady uniform background
  flow of :mod:`~gwtransport.radial_asr`, there is no multi-well or 2D/3D regional
  flow, and no kinetic or chained reactions.
