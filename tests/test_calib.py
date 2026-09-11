"""The calibration format changed once and broke two different readers.

Run #3 died in the engine and run #4 died in the renderer, both on a direct index into
`selectivity_shuffled` after it became `selectivity_shuffled_mean` plus replicates.
These tests pin the shared reader against the old format, the new format, and nothing.
"""
import calib

NEW = {
    "gain": 0.0012,
    "achieved_mean_hz": 9.93,
    "chance_level": 0.125,
    "selectivity_real": 0.41,
    "selectivity_shuffled_mean": 0.67,
    "selectivity_shuffled_sd": 0.04,
    "selectivity_shuffled_replicates": [0.63, 0.66, 0.67, 0.70, 0.69],
    "n_shuffle_replicates": 5,
    "wiring_matters": False,
    "verdict": "The shuffles BEAT the real wiring.",
}

OLD = {
    "gain": 0.0012,
    "chance_level": 0.125,
    "selectivity_real": 0.41,
    "selectivity_shuffled": 0.67,
    "wiring_matters": False,
}


def test_new_format():
    assert calib.shuffled_mean(NEW) == 0.67
    assert calib.n_replicates(NEW) == 5
    assert len(calib.replicates(NEW)) == 5


def test_old_single_shuffle_format_still_reads():
    assert calib.shuffled_mean(OLD) == 0.67
    assert calib.n_replicates(OLD) == 1
    assert calib.replicates(OLD) == [0.67]


def test_no_calibration_is_not_an_error():
    for empty in (None, {}):
        assert calib.shuffled_mean(empty) is None
        assert calib.replicates(empty) == []
        assert calib.n_replicates(empty) == 0
        assert calib.summary(empty) == {}
        assert calib.summary_line(empty) == ""


def test_summary_never_raises_on_missing_keys():
    s = calib.summary({"selectivity_real": 0.41})
    assert s["selectivity_real"] == 0.41
    assert s["verdict"] is None
    assert set(s) == set(calib.KEYS)


def test_summary_line_mentions_replicate_count():
    line = calib.summary_line(NEW)
    assert "41%" in line and "67%" in line and "mean of 5" in line and "12%" in line


def test_summary_line_omits_replicate_count_for_single_shuffle():
    line = calib.summary_line(OLD)
    assert "mean of" not in line
    assert "67%" in line
