import pytest
from deribit.surface.pipeline import build_surface_observations
from deribit.surface.observations import SurfaceExclusionCode

def test_canonical_otm_leg_selection_and_independence(sample_snapshot, curve_result):
    res = build_surface_observations(
        snapshot_id=1,
        quotes=list(curve_result.quotes),
        expiry_forwards=list(curve_result.expiry_forwards),
    )
    keys = [(obs.expiration_timestamp, obs.strike) for obs in res.observations]
    assert len(keys) == len(set(keys))

    for obs in res.observations:
        if obs.strike < obs.forward:
            assert obs.option_type == "put"
        elif obs.strike > obs.forward:
            assert obs.option_type == "call"

        assert obs.mid_iv > 0.0
        assert obs.total_variance == pytest.approx(obs.mid_iv ** 2 * obs.tau)