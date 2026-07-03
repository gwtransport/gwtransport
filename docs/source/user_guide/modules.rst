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
