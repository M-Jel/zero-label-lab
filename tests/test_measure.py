import pytest

from zerolabel.measure import measure


def d(i, x, y, r):
    return {"id": i, "x": x, "y": y, "r": r}


def test_measure_counts_radius_speed_and_guard():
    # SAM sees two drops, one of them moves 3 px right then 4 px down (5 px per step in total)
    sam = [[d(0, 0, 0, 4), d(1, 100, 100, 8)],
           [d(0, 3, 0, 4), d(1, 100, 100, 8)],
           [d(0, 3, 4, 4)]]
    # OpenCV sees the same drops plus 3 extra ones in frame 2
    cv = [[d(0, 0, 0, 4), d(1, 100, 100, 8)],
          [d(0, 3, 0, 4), d(1, 100, 100, 8)],
          [d(0, 3, 4, 4), d(1, 300, 300, 3), d(2, 400, 300, 3), d(3, 500, 300, 3)]]
    m = measure(sam, cv, fps=10, max_dist=8, flag_min=3)
    assert (m["count_first"], m["count_last"], m["tracks"]) == (2, 1, 2)
    assert m["count_mean"] == pytest.approx(5 / 3, abs=0.01)
    assert m["radius_px"]["median"] == 6.0
    assert m["speed_px_per_s"]["median"] == pytest.approx(30.0)   # 3 px * 10 fps, 0, 4 px * 10 fps
    g = m["guard"]
    assert g["flagged_frames"] == 1
    assert g["estimated_coverage"] == pytest.approx(5 / 8, abs=1e-3)
    assert m["frames_without_detection"] == 0


def test_measure_handles_frames_without_detection():
    m = measure([[], []], [[d(0, 1, 1, 2)], []], fps=30)
    assert m["frames_without_detection"] == 2
    assert m["radius_px"]["median"] is None
    assert m["guard"]["estimated_coverage"] == 0.0
