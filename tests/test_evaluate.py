import pytest

from zerolabel.evaluate import match, score


def d(i, x, y, r=5.0):
    return {"id": i, "x": x, "y": y, "r": r}


def test_match_respects_threshold_and_is_one_to_one():
    gt = [d(0, 0, 0), d(1, 100, 100)]
    pr = [d(7, 3, 4), d(8, 1, 0), d(9, 100, 130)]
    m = match(gt, pr, max_dist=8)
    assert len(m) == 1                          # gt 1 is 30 px away from any prediction
    i, j, dist = m[0]
    assert (i, j) == (0, 1) and dist == pytest.approx(1.0)


def test_match_empty_inputs():
    assert match([], [d(0, 1, 1)], 8) == []
    assert match([d(0, 1, 1)], [], 8) == []


def test_score_counts_and_radius_bias():
    truth = {"frames": [[d(0, 10, 10, 5), d(1, 50, 50, 5)],
                        [d(0, 11, 10, 5), d(1, 51, 50, 5)]], "events": []}
    pred = [[d(0, 10, 10, 6)], [d(0, 11, 10, 6), d(1, 300, 300, 4)]]
    s = score(truth, pred, max_dist=8)
    assert (s["tp"], s["fn"], s["fp"]) == (2, 2, 1)
    assert s["recall"] == pytest.approx(0.5)
    assert s["precision"] == pytest.approx(2 / 3)
    assert s["radius_bias"] == pytest.approx(1.0)
    assert s["id_switches"] == 0
    assert (s["pred_ids"], s["true_ids"]) == (2, 2)


def test_score_detects_id_switch_and_merge():
    truth = {"frames": [[d(0, 0, 0), d(1, 40, 0)], [d(0, 0, 0), d(1, 40, 0)], [d(2, 20, 0)]],
             "events": [{"frame": 2, "type": "merge", "parents": [0, 1], "child": 2}]}
    pred = [[d(5, 0, 0), d(6, 40, 0)], [d(9, 0, 0), d(6, 40, 0)], [d(6, 20, 0)]]
    s = score(truth, pred, max_dist=8)
    assert s["id_switches"] == 1                # gt 0: pred 5 then pred 9
    assert (s["merges_ok"], s["merges_checked"], s["merges_total"]) == (1, 1, 1)
