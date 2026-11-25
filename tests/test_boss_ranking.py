import pytest

from training.compare_boss import official_cases, pair_results


def test_pair_order_and_map_reversal():
    scores = {
        11: {"A": (15, 17), "B": (3, 5), "C": (3, 7)},
        12: {"A": (3, 18), "B": (9, 5), "C": (9, 7)},
    }
    games = [{"candidate": key, "seed": seed, "official": remote, "local": local}
             for seed, entries in scores.items() for key, (remote, local) in entries.items()]
    versions = {"A": {"official": 9, "local": 17.5}, "B": {"official": 6, "local": 5},
                "C": {"official": 6, "local": 7}}
    ab, _, bc = pair_results(versions, games, list(scores))
    assert ab["order"] == "same" and not ab["official_order_stable_across_maps"]
    assert [p["official"] for p in ab["by_map"]] == [12, -6]
    assert bc["order"] == "tied"


def test_missing_official_games_cannot_be_ranked(tmp_path):
    with pytest.raises(ValueError, match="缺 4 项"):
        official_cases(tmp_path, ["A", "B"], [123])
