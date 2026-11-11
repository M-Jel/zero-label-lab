import json

import pytest

from zerolabel.sam_parse import parse_file


def _delta(frame, boxes):
    parts = [f"{i}<|box;x1={x1};y1={y1};x2={x2};y2={y2};w=640;h=480;c={c}|>"
             f"<|mask;x=0;y=0;c={c};data=ab,c|d|>" for i, (x1, y1, x2, y2, c) in boxes]
    return f"<{frame}f>" + ",".join(parts) + "\n"


def _write(path, deltas):
    with open(path, "w", encoding="utf-8") as f:
        f.write(json.dumps({"type": "response.created"}) + "\n")
        for d in deltas:
            f.write(json.dumps({"type": "response.output_text.delta", "delta": d}) + "\n")
        f.write(json.dumps({"type": "response.completed"}) + "\n")


def test_parse_boxes_and_fill_missing_frames(tmp_path):
    raw = tmp_path / "circle.jsonl"
    _write(raw, [_delta(0, [(0, (10, 20, 30, 40, 0.9)), (1, (100, 100, 110, 120, 0.5))]),
                 _delta(2, [(0, (12, 20, 32, 40, 0.9))])])
    frames, received = parse_file(raw, n_expected=5)
    assert received == 2
    assert len(frames) == 5                      # padded to the expected length
    assert frames[1] == [] and frames[3] == [] and frames[4] == []
    a, b = frames[0]
    assert (a["id"], a["x"], a["y"], a["r"], a["conf"]) == (0, 20, 30, 10, 0.9)
    assert b["id"] == 1 and b["r"] == pytest.approx((10 + 20) / 4)
    assert frames[2][0]["x"] == 22


def test_mask_payload_is_not_read_as_a_box(tmp_path):
    raw = tmp_path / "circle.jsonl"
    _write(raw, [_delta(0, [(3, (0, 0, 4, 4, 0.7))])])
    frames, _ = parse_file(raw)
    assert len(frames) == 1 and len(frames[0]) == 1 and frames[0][0]["id"] == 3


def test_delta_without_frame_number_is_an_error(tmp_path):
    raw = tmp_path / "bad.jsonl"
    _write(raw, ["0<|box;x1=0;y1=0;x2=1;y2=1|>"])
    with pytest.raises(ValueError):
        parse_file(raw)
