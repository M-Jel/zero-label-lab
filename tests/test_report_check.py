from zerolabel.report_agent import check


def measures(coverage=0.9):
    return {"frames": 150, "fps": 30.0, "count_first": 12, "count_last": 5, "count_mean": 8.95,
            "tracks": 12, "frames_without_detection": 0,
            "radius_px": {"mean": 14.52, "p10": 8.63, "median": 14.59, "p90": 20.13},
            "speed_px_per_s": {"mean": 108.19, "median": 100.62, "p90": 185.54},
            "guard": {"cv_unmatched_per_frame": 1.22, "flag_min": 3, "flagged_frames": 15,
                      "flagged_fraction": 0.1, "estimated_coverage": coverage}}


def test_numbers_from_measurements_pass():
    assert check("We saw 12 drops at the start and 5 at the end. Median radius 14.59 px.",
                 measures()) == []


def test_invented_number_is_flagged():
    issues = check("We saw 13 drops at the start.", measures())
    assert any("13" in i for i in issues)


def test_percent_of_a_fraction_is_accepted():
    assert check("Estimated coverage is 90%.", measures(0.9)) == []


def test_missing_warning_is_flagged_when_coverage_is_low():
    issues = check("Everything looks fine with 12 drops.", measures(0.44))
    assert any("WARNING" in i for i in issues)
