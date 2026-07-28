"""Cross-cutting regression tests for the front-tracking engine.

Scenario-level unit tests live in ``tests/src``. This package holds the checks
that span several engine modules at once: the invariant contracts pinned against
analytic and high-precision references, and the concentration-dispatch parity
between ``concentration_at_point`` and ``compute_domain_mass``.
"""
