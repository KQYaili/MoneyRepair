from moneyrepair.batch import BatchState, load_batch_state, save_batch_state
from moneyrepair.solver import CoverageSolution


def _solution(ids, coverage=0.99, area=100):
    return CoverageSolution(fragment_ids=tuple(ids), coverage=coverage, area=area)


def test_confirm_and_reject_write_audit_events():
    state = BatchState()
    state.add_confirmation("note-00001", _solution(["a", "b"]), operator="alice", reason="clean overlay")
    state.reject_solution(_solution(["c", "d"]), operator="bob", reason="visible seam")

    assert [event.action for event in state.audit_log] == ["confirm", "reject"]
    confirm_event = state.audit_log[0]
    assert confirm_event.note_id == "note-00001"
    assert confirm_event.operator == "alice"
    assert confirm_event.reason == "clean overlay"
    assert confirm_event.timestamp
    reject_event = state.audit_log[1]
    assert reject_event.operator == "bob"
    assert reject_event.fragment_ids == ("c", "d")


def test_audit_log_round_trips_through_disk(tmp_path):
    state = BatchState()
    state.add_confirmation("note-00001", _solution(["a", "b"]), operator="alice", reason="ok")
    path = tmp_path / "state.json"
    save_batch_state(path, state)

    reloaded = load_batch_state(path)
    assert len(reloaded.audit_log) == 1
    assert reloaded.audit_log[0].operator == "alice"
    assert reloaded.audit_log[0].action == "confirm"
