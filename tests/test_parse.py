from datetime import date

import pytest

from nldate import parse

TODAY = date(2026, 11, 11)  # a Wednesday


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("today", date(2026, 11, 11)),
        ("tomorrow", date(2026, 11, 12)),
        ("yesterday", date(2026, 11, 10)),
        ("December 1st, 2025", date(2025, 12, 1)),
        ("2025-12-01", date(2025, 12, 1)),
        ("12/1/2025", date(2025, 12, 1)),
        ("1 Dec 2025", date(2025, 12, 1)),
        ("5 days before December 1st, 2025", date(2025, 11, 26)),
        ("1 year and 2 months after yesterday", date(2028, 1, 10)),
        ("two weeks from tomorrow", date(2026, 11, 26)),
        ("in 3 days", date(2026, 11, 14)),
        ("3 days ago", date(2026, 11, 8)),
        ("next Tuesday", date(2026, 11, 17)),
        ("last friday", date(2026, 11, 6)),
    ],
)
def test_parse(text: str, expected: date) -> None:
    assert parse(text, TODAY) == expected


@pytest.mark.parametrize("text", ["", "banana", "February 30, 2025"])
def test_invalid_raises_value_error(text: str) -> None:
    with pytest.raises(ValueError):
        parse(text, TODAY)
