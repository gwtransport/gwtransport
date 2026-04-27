.. _theory:

Theory: Why the APVD Approach Works
====================================

This page deepens :doc:`concepts` by walking from the classical residence-time distribution to the flow-invariant pore volume distribution, listing the misconceptions practitioners encounter, and showing the derivations behind the variance formulas in :ref:`concept-variance-components`.

.. _theory-rtd-to-pvd:

From the Residence Time Distribution to the Pore Volume Distribution
---------------------------------------------------------------------

This section walks from the residence-time distribution (RTD) — what an experimentalist measures in a tracer test — to the pore volume distribution (PVD), the flow-invariant geometric "fingerprint" of the aquifer. Both are introduced briefly in :ref:`concept-pore-volume-distribution` and :ref:`concept-residence-time`; here we connect them.

The Residence Time Distribution
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The residence-time distribution :math:`E(t)` was introduced by Danckwerts (1953) as the distribution of times that fluid elements spend in a flow system. It is what one *measures*: inject a pulse of tracer at the inlet, normalise the breakthrough curve at the outlet, and the result is :math:`E(t)`.

For an aquifer between an infiltration source and an extraction well, the RTD is built from the per-streamline residence times defined in :ref:`concept-residence-time`. Each streamline :math:`i` carries an equal share of the total flow, its tracer arrives after :math:`\tau_i = V_i \cdot R / Q_i`, and the well sees the average over all streamlines.

The problem: the RTD changes with flow
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The RTD depends explicitly on flow. When the total pumping rate :math:`Q` doubles, every :math:`\tau_i` halves and the entire distribution shifts and compresses. Under time-varying :math:`Q(t)`, each streamline integrates a different flow history, so the RTD at any moment depends on the recent flow record — not just the current rate.

The practical consequence is that **a measured RTD is valid only for the flow conditions under which it was measured.** Calibrating an RTD on one season and applying it to another would silently introduce a model error proportional to the change in flow magnitude.

The fix: work in pore-volume space
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The expression :math:`\tau_i = V_i \cdot R / Q_i` separates a streamline-specific *geometric* quantity, :math:`V_i`, from a *forcing* quantity, :math:`Q_i`. The volumes :math:`V_i` are properties of the aquifer; the flow :math:`Q_i` is what the operator imposes.

The pore volume distribution :math:`f(V)` is the flow-invariant analogue of the RTD: it describes how pore volume is distributed across streamlines, and it is unchanged when :math:`Q` rises or falls. Under variable flow, residence times are no longer a simple division — instead, for each streamline they are defined implicitly by

.. math::

   \int_{t-\tau}^{t} Q(s) \, ds \;=\; V_i \cdot R

which says "the residence time at extraction time :math:`t` is whatever :math:`\tau` it took to pump :math:`V_i \cdot R` cubic metres of water." This integral is solved by interpolation on the cumulative-flow curve and is exactly what :py:func:`gwtransport.residence_time.residence_time` computes.

Re-calibration is replaced by cumulative-flow integration. Calibrate the PVD once; predict transport under any flow trajectory.

Why this works: steady streamlines
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The PVD is flow-invariant only when the streamlines themselves are flow-invariant. When total flow scales up or down, each streamline's flow scales by the same factor and the paths do not move — pressure controls the magnitude everywhere proportionally. When boundary conditions redirect flow (a new well activates, recharge shifts asymmetrically), the streamlines rearrange and the PVD changes with them.

The conditions under which this assumption holds, and the way it can be tested by cross-validating across flow regimes, are documented in :ref:`assumption-steady-streamlines`. This page does not restate them.

.. _theory-misconceptions:

Common Misconceptions
---------------------

Practitioners coming from grid-based transport modelling, or from textbook 1D advection–dispersion, often carry intuitions that mislead in the APVD framework. Six recur often enough to deserve direct correction.

1. "Dispersion increases with flow"
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The longitudinal dispersion coefficient :math:`D_L = D_m + \alpha_L \, v` does increase with velocity. *But* the breakthrough curve at a fixed travel distance does **not** spread more in time — the shorter time in the aquifer compensates. In pore-volume units, mechanical-dispersion spreading is flow-independent and molecular-diffusion spreading decreases with flow. See :ref:`concept-dispersion` for the formulas.

2. "The RTD is a property of the aquifer"
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Only under constant flow. Under variable flow the RTD shifts and compresses every time :math:`Q` changes. The **PVD** is the true flow-invariant property; this is the central insight of the APVD approach (see :ref:`theory-rtd-to-pvd`).

3. "More parameters always give a better model"
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Without more *data*, no. A two-parameter gamma APVD calibrated against a temperature time series uses every degree of freedom the measurement constrains. A 3D model with thousands of parameters needs spatially distributed data to be predictive — without it, the extra parameters are underdetermined and the predictions are no more reliable.

4. "Dispersivity is a material constant"
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Apparent :math:`\alpha_L` famously increases with experiment scale (Gelhar et al., 1992): larger experiments integrate more heterogeneity into a single fitted dispersivity. The "true" pore-scale :math:`\alpha_L` is small (mm to cm). What field studies report as "field-scale dispersivity" is largely unresolved heterogeneity — which the APVD captures explicitly as :math:`\sigma_V`.

5. "The APVD captures all dispersion"
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

What :math:`\sigma_V` represents depends on how it was obtained. A breakthrough-fitted :math:`\sigma_V` is an *effective* parameter: it already lumps together macrodispersion, microdispersion, and an average molecular-diffusion contribution from the calibration window, so downstream calculations treat it as the total. A streamline-derived :math:`\sigma_V` is purely *geometric* and reflects macrodispersion only; microdispersion and molecular diffusion enter the variance sum on their own terms. See :ref:`concept-dispersion` for the rules.

6. "Higher flow means more mixing"
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Mixing is more vigorous, but acts over a shorter time. Net spreading of the breakthrough curve in time *decreases* at higher flow. The relative importance of macrodispersion versus microdispersion is unchanged by flow magnitude, while molecular diffusion becomes *less* important at higher flow.

.. _theory-variance-derivations:

Appendix: Derivations of :math:`\sigma_{V,\mathrm{disp}}` and :math:`\sigma_{V,\mathrm{diff}}`
-----------------------------------------------------------------------------------------------

This appendix derives the closed-form expressions used in :ref:`concept-variance-components`. All three derivations start from the temporal variance of the 1D advection–dispersion breakthrough curve at distance :math:`x = L`,

.. math::

   \sigma_t^2 \;=\; \frac{2 \, D_L \, L}{v^{3}},

with :math:`D_L = D_m + \alpha_L \, v`, and convert to pore-volume variance via :math:`\sigma_V^2 = \sigma_t^2 \cdot Q^2` and :math:`v = Q / (A \, n)`, where :math:`A` is the streamtube cross-section and :math:`n` the porosity.

Microdispersion contribution :math:`\sigma_{V,\mathrm{disp}}`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. Take the mechanical-dispersion limit :math:`D_L \approx \alpha_L \, v`, giving :math:`\sigma_t^2 = 2 \alpha_L L / v^{2}`.

2. Convert with :math:`\sigma_V^2 = \sigma_t^2 Q^2`:

   .. math::

      \sigma_V^2 \;=\; \frac{2 \, \alpha_L \, L \, Q^2}{v^{2}} \;=\; 2 \, \alpha_L \, L \, (A \, n)^2.

3. Substitute the mean pore volume of one representative streamline, :math:`\bar V = A \, n \, L`:

   .. math::

      \sigma_{V,\mathrm{disp}} \;=\; \bar V \sqrt{\frac{2 \, \alpha_L}{L}}.

This contribution is **flow-independent** — it depends only on the aquifer geometry (:math:`\bar V`, :math:`L`) and the medium property :math:`\alpha_L`. It matches the Mean PV row of :ref:`concept-variance-components`.

Molecular-diffusion contribution :math:`\sigma_{V,\mathrm{diff}}`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. Take the molecular-diffusion limit :math:`D_L \approx D_m`, giving :math:`\sigma_t^2 = 2 D_m L / v^{3}`.

2. Convert with :math:`\sigma_V^2 = \sigma_t^2 Q^2`:

   .. math::

      \sigma_V^2 \;=\; \frac{2 \, D_m \, L \, Q^2}{v^{3}} \;=\; \frac{2 \, D_m \, L \, (A \, n)^3}{Q}.

3. Substitute :math:`\bar V = A \, n \, L` and group:

   .. math::

      \sigma_{V,\mathrm{diff}} \;=\; \frac{\bar V}{L} \sqrt{\frac{2 \, D_m \, R \, \bar V}{Q}}.

This contribution **decreases** with flow (as :math:`Q^{-1/2}`) — at higher flow, less time is available for diffusion to act. The retardation factor :math:`R` appears explicitly; the next subsection explains why. The result matches the Mean PV row of :ref:`concept-variance-components`.

Why retardation cancels for microdispersion but not for molecular diffusion
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The retarded ADE :math:`R \, \partial_t C + v \, \partial_x C = D_L \, \partial_x^2 C` rescales to effective velocity :math:`v_{\mathrm{eff}} = v / R` and effective dispersion :math:`D_{L,\mathrm{eff}} = D_L / R`. The temporal variance becomes

.. math::

   \sigma_t^2 \;=\; \frac{2 \, (D_L / R) \, L}{(v/R)^{3}} \;=\; \frac{2 \, D_L \, R^{2} \, L}{v^{3}}.

For mechanical dispersion :math:`D_L = \alpha_L \, v`, this gives :math:`\sigma_V^2 = 2 \alpha_L R^{2} L (A n)^2`. In terms of the **effective retarded pore volume** :math:`V_{\mathrm{eff}} = V \cdot R`, the formula reads :math:`\sigma_{V,\mathrm{disp}} = V_{\mathrm{eff}} \sqrt{2 \alpha_L / L}` — the same expression as for the unretarded compound. Retardation slows the compound, but it traverses the same physical heterogeneity, so :math:`R` does not appear separately.

For molecular diffusion :math:`D_L = D_m`, the rescaling gives :math:`\sigma_t^2 = 2 D_m R L / v^{3}` — one factor of :math:`R` survives. Diffusion depends on *time*, not distance, and a more retarded compound spends more time in the aquifer, so it accumulates more diffusive spreading. This is why the molecular-diffusion formula carries an explicit :math:`R` while the mechanical-dispersion formula does not.
