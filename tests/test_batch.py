import numpy as np

from moneyrepair.batch import BatchState, load_batch_state, save_batch_state
from moneyrepair.solver import CoverageSolution
from moneyrepair.types import Fragment


def test_batch_state_tracks_confirmed_fragments(tmp_path):
    state = BatchState()
    solution = CoverageSolution(fragment_ids=("a", "b"), coverage=0.99, area=99)
    state.add_confirmation("note-00001", solution)
    path = tmp_path / "state.json"

    save_batch_state(path, state)
    loaded = load_batch_state(path)
    fragments = [
        Fragment("a", np.ones((2, 2), dtype=bool)),
        Fragment("b", np.ones((2, 2), dtype=bool)),
        Fragment("c", np.ones((2, 2), dtype=bool)),
    ]

    assert loaded.used_fragment_ids == {"a", "b"}
    assert loaded.active_fragment_ids(fragments) == {"c"}


def test_batch_state_filters_rejected_solutions():
    state = BatchState()
    bad = CoverageSolution(fragment_ids=("b", "a"), coverage=0.9, area=90)
    good = CoverageSolution(fragment_ids=("c",), coverage=0.5, area=50)

    state.reject_solution(bad)

    assert state.filter_rejected([bad, good]) == [good]
