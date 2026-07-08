import pytest

from llm_labeling_system import repository
from llm_labeling_system.db import get_conn, init_db


def _window(idx):
    return {
        "window_id": f"hash{idx:04d}",
        "vehicle_id": "veh1",
        "start_index": idx,
        "point_count": 5,
        "start_time": "2024-01-01T08:00:00",
        "end_time": "2024-01-01T08:00:04",
        "summary": {"max_gps_speed": 40, "data_quality_flags": []},
        "rows": [],
    }


def _new_session(conn, name, n=3):
    return repository.create_session(
        conn,
        source_name=name,
        source_path=None,
        window_size=5,
        stride=5,
        max_rows=None,
        max_windows=100,
        row_count=n * 5,
        windows=[_window(i) for i in range(n)],
    )


@pytest.fixture(autouse=True)
def _db():
    init_db()


def test_label_scoping_between_sessions():
    with get_conn() as conn:
        a = _new_session(conn, "a.csv")
        b = _new_session(conn, "b.csv")
        pk = repository.get_window_pk(conn, a, 0)
        repository.upsert_label(conn, a, pk, {
            "label": "speeding", "confidence": 0.8, "risk_level": "medium",
            "use_for_training": True, "human_review_needed": False,
            "reason": "", "evidence": [], "data_quality_flags": [],
            "source": "manual", "model": "human",
        })

    with get_conn() as conn:
        assert repository.session_progress(conn, a)["labeled"] == 1
        # Session B must be untouched -> no cross-session contamination.
        assert repository.session_progress(conn, b)["labeled"] == 0
        assert repository.list_labels(conn, b) == []


def test_relabel_is_upsert_not_duplicate():
    with get_conn() as conn:
        s = _new_session(conn, "c.csv")
        pk = repository.get_window_pk(conn, s, 0)
        base = {
            "confidence": 0.9, "risk_level": "medium", "use_for_training": True,
            "human_review_needed": False, "reason": "", "evidence": [],
            "data_quality_flags": [], "source": "manual", "model": "human",
        }
        repository.upsert_label(conn, s, pk, {"label": "speeding", **base})
        repository.upsert_label(conn, s, pk, {"label": "normal", **base})

    with get_conn() as conn:
        assert repository.session_progress(conn, s)["labeled"] == 1
        label = repository.get_label(conn, s, pk)
        assert label["label"] == "normal"  # latest write wins


def test_next_unlabeled_and_delete_cascade():
    with get_conn() as conn:
        s = _new_session(conn, "d.csv", n=2)
        assert repository.next_unlabeled_seq(conn, s) == 0
        pk = repository.get_window_pk(conn, s, 0)
        repository.upsert_label(conn, s, pk, {
            "label": "normal", "confidence": 0.9, "risk_level": "low",
            "use_for_training": True, "human_review_needed": False,
            "reason": "", "evidence": [], "data_quality_flags": [],
            "source": "manual", "model": "human",
        })
        assert repository.next_unlabeled_seq(conn, s) == 1

    with get_conn() as conn:
        repository.delete_session(conn, s)
    with get_conn() as conn:
        assert repository.get_session(conn, s) is None
        # windows + labels cascade-deleted
        assert repository.window_count(conn, s) == 0
