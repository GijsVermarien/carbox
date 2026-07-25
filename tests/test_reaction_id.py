"""
Tests that every parsed reaction carries a stable reaction_id, and that
the id survives Network.get_ode()'s vectorized reordering.

Guards against a real bug found in the CSE work: get_ode groups/reorders
reactions for vectorization, and if reaction_id isn't carried along in
that same order, rate-output columns silently point at the wrong
reaction.
"""

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from carbox.parsers import parse_chemical_network  # noqa: E402


@pytest.mark.parametrize(
    "network_file,format_type",
    [
        ("data/uclchem_gas_phase_only.csv", "uclchem"),
        ("data/umist22_mini.csv", "umist"),
    ],
)
def test_reaction_id_populated_and_survives_reorder(network_file, format_type):
    network = parse_chemical_network(str(PROJECT_ROOT / network_file), format_type)

    ids_before = [r.reaction_id for r in network.reactions]
    assert len(ids_before) > 0
    assert all(rid is not None for rid in ids_before), (
        f"{format_type} parser left reaction_id unset for some reactions"
    )
    assert len(set(ids_before)) == len(ids_before), "reaction_id values must be unique"

    n_reactions_before = len(network.reactions)
    network.get_ode()  # triggers vectorized grouping/reordering

    ids_after = [r.reaction_id for r in network.reactions]
    assert len(network.reactions) == n_reactions_before
    assert all(rid is not None for rid in ids_after)
    assert set(ids_after) == set(ids_before), (
        "reaction_id set changed across get_ode() reordering"
    )
