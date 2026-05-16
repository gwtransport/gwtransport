"""
Unit Tests for Front Tracking with Varying Flow (DELETED IN P1.8 REFACTOR).

The flow_change machinery (handle_flow_change, recreate_*_with_new_flow) was deleted
in the (V, θ) refactor because flow is absorbed entirely into the θ(t) mapping at the
API boundary. Waves no longer need recreation when the flow rate changes.

Tests that verified the wave-recreation internals are obsolete; tests that verified
output behavior under varying flow are still indirectly covered by the integration
tests in tests/src/test_front_tracking_solver.py and tests/src/test_front_tracking_api.py.

This file is part of gwtransport which is released under AGPL-3.0 license.
See the ./LICENSE file or go to https://github.com/gwtransport/gwtransport/blob/main/LICENSE for full license details.
"""

import pytest

pytestmark = pytest.mark.skip(
    reason="flow_change machinery deleted in (V, θ) refactor — flow is absorbed into θ at the API boundary"
)


def test_flow_changes_machinery_removed():
    """Placeholder so pytest collects this module without errors."""
