from pathlib import Path

import pytest

from zerolabel.verifier import SIZE_BIAS_SENTENCE, check, glued_words


def measures(coverage):
    return {"frames": 150, "fps": 30.0, "count_first": 0, "count_last": 15, "count_mean": 12.26,
            "tracks": 16, "frames_without_detection": 14,
            "radius_px": {"mean": 8.66, "p10": 7.08, "median": 8.43, "p90": 10.31},
            "speed_px_per_s": {"mean": 107.42, "median": 106.07, "p90": 175.57},
            "guard": {"cv_unmatched_per_frame": 15.41, "flag_min": 3, "flagged_frames": 150,
                      "flagged_fraction": 1.0, "estimated_coverage": coverage}}


GOOD_LOW = ("WARNING: the model probably missed many drops (estimated coverage 44.3%), so the counts "
            "are too low. " + SIZE_BIAS_SENTENCE + "\nSummary\nThe video has 150 frames. "
            "The median radius is 8.43 pixels and the median speed is 106.07 pixels per second.")


def test_good_report_with_warning_passes():
    assert check(GOOD_LOW, measures(0.443)) == []


def test_warning_without_size_sentence_fails():
    text = "WARNING: the model probably missed many drops, so the counts are too low."
    assert any("biased toward larger" in i for i in check(text, measures(0.443)))


def test_false_alarm_warning_fails():
    assert any("starts with 'WARNING:'" in i for i in check(GOOD_LOW, measures(0.9)))


@pytest.mark.parametrize("text,field", [
    ("The estimated_coverage is 0.9.", "estimated_coverage"),
    ("Radius p90 is 10.31 pixels.", "p90"),
    ("Unmatched per frame 15.41.", "unmatched per frame"),
    ("Flagged frames 150 with flag min 3.", "flag min"),
])
def test_raw_field_names_are_flagged(text, field):
    assert any(f"'{field}'" in i for i in check(text, measures(0.9)))


def test_more_than_two_decimals_is_flagged():
    issues = check("Coverage is 0.443 here.", measures(0.443))
    assert any("more than 2 decimals" in i for i in issues)


def test_glued_words():
    assert glued_words("Count 14 frames withoutdetection and good words like percentile") == \
        ["withoutdetection"]
    assert glued_words("however, nevertheless, throughout, overall, measurement") == []


def test_previous_spark_reports_now_fail():
    """The level-1 Spark report (VERIFIED by the old verifier) copied raw field names."""
    old = ("Unmatched per frame 1.22, flagged frames 15 with flagged fraction 0.1 and flag min 3. "
           "Estimated coverage 0.88.")
    m = measures(0.88)
    m["guard"].update(cv_unmatched_per_frame=1.22, flagged_frames=15, flagged_fraction=0.1)
    issues = check(old, m)
    assert any("unmatched per frame" in i for i in issues) and any("flag min" in i for i in issues)


def test_percentile_levels_are_allowed():
    m = measures(0.443)
    m["guard"]["flagged_fraction"] = 1.0
    assert not any("10%" in i for i in check("10% of drops were smaller than 7.08 pixels.", m))
