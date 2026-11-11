"""Implementation of the natural-language date parser.

The parser is a small recursive-descent matcher built from regular expressions.
Each ``_try_*`` handler returns a ``date`` if the text matches its pattern and
``None`` otherwise, so handlers can be tried in order.  Any text that no
handler accepts results in a ``ValueError``.
"""

from __future__ import annotations

import calendar
import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import date, datetime, timedelta

# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------

MONTHS: dict[str, int] = {
    "january": 1,
    "jan": 1,
    "february": 2,
    "feb": 2,
    "march": 3,
    "mar": 3,
    "april": 4,
    "apr": 4,
    "may": 5,
    "june": 6,
    "jun": 6,
    "july": 7,
    "jul": 7,
    "august": 8,
    "aug": 8,
    "september": 9,
    "sept": 9,
    "sep": 9,
    "october": 10,
    "oct": 10,
    "november": 11,
    "nov": 11,
    "december": 12,
    "dec": 12,
}

WEEKDAYS: dict[str, int] = {
    "monday": 0,
    "mon": 0,
    "tuesday": 1,
    "tues": 1,
    "tue": 1,
    "wednesday": 2,
    "wed": 2,
    "thursday": 3,
    "thurs": 3,
    "thur": 3,
    "thu": 3,
    "friday": 4,
    "fri": 4,
    "saturday": 5,
    "sat": 5,
    "sunday": 6,
    "sun": 6,
}


def _alternation(words: Iterable[str]) -> str:
    """Regex alternation of the given words, longest first."""
    items = sorted(words, key=len, reverse=True)
    return "(?:" + "|".join(re.escape(w) for w in items) + ")"


MONTH_RE = _alternation(MONTHS)
WEEKDAY_RE = _alternation(WEEKDAYS)

_ONES: dict[str, int] = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
}
_TENS: dict[str, int] = {
    "twenty": 20,
    "thirty": 30,
    "forty": 40,
    "fifty": 50,
    "sixty": 60,
    "seventy": 70,
    "eighty": 80,
    "ninety": 90,
}
_ORD_BASE: dict[str, int] = {
    "first": 1,
    "second": 2,
    "third": 3,
    "fourth": 4,
    "fifth": 5,
    "sixth": 6,
    "seventh": 7,
    "eighth": 8,
    "ninth": 9,
    "tenth": 10,
    "eleventh": 11,
    "twelfth": 12,
    "thirteenth": 13,
    "fourteenth": 14,
    "fifteenth": 15,
    "sixteenth": 16,
    "seventeenth": 17,
    "eighteenth": 18,
    "nineteenth": 19,
    "twentieth": 20,
    "thirtieth": 30,
}


def _build_ordinals() -> dict[str, int]:
    ordinals = dict(_ORD_BASE)
    for tens_word, tens in (("twenty", 20), ("thirty", 30)):
        for word, n in list(_ORD_BASE.items())[:9]:
            ordinals[f"{tens_word} {word}"] = tens + n
            ordinals[f"{tens_word}-{word}"] = tens + n
    return {k: v for k, v in ordinals.items() if v <= 31}


ORDINALS = _build_ordinals()
_ORDINAL_WORD_RE = re.compile(r"\b" + _alternation(ORDINALS) + r"\b")

# unit name -> (kind, multiplier). kind is "months", "days" or "bdays".
UNITS: dict[str, tuple[str, int]] = {
    "day": ("days", 1),
    "days": ("days", 1),
    "week": ("days", 7),
    "weeks": ("days", 7),
    "wk": ("days", 7),
    "wks": ("days", 7),
    "fortnight": ("days", 14),
    "fortnights": ("days", 14),
    "month": ("months", 1),
    "months": ("months", 1),
    "mo": ("months", 1),
    "mos": ("months", 1),
    "year": ("months", 12),
    "years": ("months", 12),
    "yr": ("months", 12),
    "yrs": ("months", 12),
    "decade": ("months", 120),
    "decades": ("months", 120),
    "century": ("months", 1200),
    "centuries": ("months", 1200),
    "business day": ("bdays", 1),
    "business days": ("bdays", 1),
    "working day": ("bdays", 1),
    "working days": ("bdays", 1),
    "workday": ("bdays", 1),
    "workdays": ("bdays", 1),
    "weekday": ("bdays", 1),
    "weekdays": ("bdays", 1),
}
_UNITS_LONGEST_FIRST = sorted(UNITS, key=len, reverse=True)

# Words meaning "later than" (+1) or "earlier than" (-1) a reference date.
_DIRECTIONS: dict[str, int] = {
    "after": 1,
    "from": 1,
    "past": 1,
    "following": 1,
    "since": 1,
    "before": -1,
    "prior to": -1,
    "preceding": -1,
    "until": -1,
    "till": -1,
}
_DIRECTION_RE = re.compile(r"\b(" + "|".join(_DIRECTIONS) + r")\b")

_PAST_MODS = {"last", "previous", "past", "prev"}
_REL_MOD_RE = "(?:next|last|previous|prev|this coming|this|coming|upcoming|past)"


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> date:
    """The n-th (1-based) weekday of a month; n == -1 means the last one."""
    last_day = calendar.monthrange(year, month)[1]
    if n < 0:
        last = date(year, month, last_day)
        return last - timedelta(days=(last.weekday() - weekday) % 7)
    first = date(year, month, 1)
    day = 1 + (weekday - first.weekday()) % 7 + 7 * (n - 1)
    if day > last_day:
        raise ValueError("that month does not have that many such weekdays")
    return date(year, month, day)


HOLIDAYS: dict[str, Callable[[int], date]] = {
    "new years day": lambda y: date(y, 1, 1),
    "new years eve": lambda y: date(y, 12, 31),
    "new years": lambda y: date(y, 1, 1),
    "valentines day": lambda y: date(y, 2, 14),
    "valentines": lambda y: date(y, 2, 14),
    "independence day": lambda y: date(y, 7, 4),
    "fourth of july": lambda y: date(y, 7, 4),
    "halloween": lambda y: date(y, 10, 31),
    "thanksgiving day": lambda y: _nth_weekday(y, 11, 3, 4),
    "thanksgiving": lambda y: _nth_weekday(y, 11, 3, 4),
    "christmas eve": lambda y: date(y, 12, 24),
    "christmas day": lambda y: date(y, 12, 25),
    "christmas": lambda y: date(y, 12, 25),
    "xmas": lambda y: date(y, 12, 25),
}
_HOLIDAY_RE = _alternation(HOLIDAYS)

# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _parse_number(text: str) -> int | None:
    """Parse '5', 'five', 'twenty-one', 'a', 'an', 'a hundred' ..."""
    text = text.strip()
    if text.isdigit():
        return int(text)
    words = text.replace("-", " ").split()
    if not words:
        return None
    total = 0
    for i, word in enumerate(words):
        if word in ("a", "an") and i == 0:
            total += 1
        elif word in _ONES:
            total += _ONES[word]
        elif word in _TENS:
            total += _TENS[word]
        elif word == "hundred":
            total = (total or 1) * 100
        else:
            return None
    return total


def _day_number(token: str) -> int | None:
    """Parse '1', '1st', 'first', 'twenty-first' into a day of the month."""
    token = token.strip()
    m = re.fullmatch(r"(\d{1,2})(?:st|nd|rd|th)?", token)
    if m:
        return int(m.group(1))
    return ORDINALS.get(token)


def _add_months(d: date, months: int) -> date:
    """Add calendar months, clamping the day (Jan 31 + 1 month = Feb 28)."""
    year, month0 = divmod(d.year * 12 + (d.month - 1) + months, 12)
    month = month0 + 1
    day = min(d.day, calendar.monthrange(year, month)[1])
    return date(year, month, day)


def _add_business_days(d: date, n: int) -> date:
    step = 1 if n > 0 else -1
    weeks, rest = divmod(abs(n), 5)
    d += timedelta(days=step * 7 * weeks)
    while rest:
        d += timedelta(days=step)
        if d.weekday() < 5:
            rest -= 1
    return d


@dataclass
class _Delta:
    months: int = 0
    days: int = 0
    business_days: int = 0

    def apply(self, base: date, sign: int) -> date:
        result = _add_months(base, sign * self.months)
        result += timedelta(days=sign * self.days)
        return _add_business_days(result, sign * self.business_days)


def _parse_quantity(part: str) -> tuple[str, int] | None:
    """Parse one 'N unit' chunk into (kind, amount)."""
    part = re.sub(r"^the\s+", "", part.strip())
    count: int | None = None
    unit: str | None = None
    m = re.fullmatch(r"(\d+)\s*([a-z ]+)", part)
    if m:
        count, unit = int(m.group(1)), m.group(2).strip()
    else:
        for candidate in _UNITS_LONGEST_FIRST:
            if part == candidate:
                count, unit = 1, candidate
                break
            if part.endswith(" " + candidate):
                prefix = part[: -len(candidate) - 1]
                number = _parse_number(prefix)
                if number is not None:
                    count, unit = number, candidate
                    break
    if count is None or unit is None or unit not in UNITS:
        return None
    kind, multiplier = UNITS[unit]
    return kind, count * multiplier


def _parse_delta(text: str) -> _Delta | None:
    """Parse '1 year and 2 months', '3 days', 'a week, 2 days' ..."""
    parts = re.split(r"\s*,\s*(?:and\s+)?|\s+and\s+|\s*&\s*", text.strip())
    delta = _Delta()
    for part in parts:
        quantity = _parse_quantity(part)
        if quantity is None:
            return None
        kind, amount = quantity
        if kind == "months":
            delta.months += amount
        elif kind == "days":
            delta.days += amount
        else:
            delta.business_days += amount
    return delta


def _month_ref(text: str, today: date) -> tuple[int, int] | None:
    """Resolve 'december 2025', 'next month', 'month' ... to (year, month)."""
    text = re.sub(r"^the\s+", "", text.strip())
    m = re.fullmatch(r"(?:(next|last|previous|prev|this|coming)\s+)?month", text)
    if m:
        mod = m.group(1)
        shift = 1 if mod in ("next", "coming") else -1 if mod in _PAST_MODS else 0
        first = _add_months(today.replace(day=1), shift)
        return first.year, first.month
    m = re.fullmatch(rf"({MONTH_RE})(?:\s+(\d{{4}}))?", text)
    if m:
        return (int(m.group(2)) if m.group(2) else today.year), MONTHS[m.group(1)]
    return None


# ---------------------------------------------------------------------------
# Handlers: each returns a date or None (no match)
# ---------------------------------------------------------------------------


def _try_keyword(s: str, today: date) -> date | None:
    if s in ("today", "now", "tonight", "this morning", "this afternoon"):
        return today
    if s in ("this evening", "right now", "currently"):
        return today
    if s in ("tomorrow", "tmrw", "tmr", "tomorow", "tommorow"):
        return today + timedelta(days=1)
    if s in ("yesterday", "yday"):
        return today - timedelta(days=1)
    return None


def _try_offset(s: str, today: date) -> date | None:
    """'3 days ago', 'in 2 weeks', '5 days before X', '1 year after X' ..."""
    m = re.fullmatch(r"(.+?)\s+ago", s)
    if m:
        delta = _parse_delta(m.group(1))
        if delta is not None:
            return delta.apply(today, -1)

    m = re.fullmatch(r"in\s+(.+)", s)
    if m:
        delta = _parse_delta(m.group(1))
        if delta is not None:
            return delta.apply(today, 1)

    m = re.fullmatch(r"(.+?)\s+(later|hence|earlier)", s)
    if m:
        delta = _parse_delta(m.group(1))
        if delta is not None:
            return delta.apply(today, -1 if m.group(2) == "earlier" else 1)

    for match in _DIRECTION_RE.finditer(s):
        left, right = s[: match.start()].strip(), s[match.end() :].strip()
        if not left or not right:
            continue
        delta = _parse_delta(left)
        if delta is None:
            continue
        try:
            base = _parse_expr(right, today)
        except ValueError:
            continue
        return delta.apply(base, _DIRECTIONS[match.group(1)])
    return None


def _try_weekday(s: str, today: date) -> date | None:
    """'tuesday', 'next friday', 'last monday', 'this coming sunday' ..."""
    m = re.fullmatch(rf"(?:the\s+)?(?:({_REL_MOD_RE})\s+)?({WEEKDAY_RE})", s)
    if not m:
        return None
    mod, target = m.group(1), WEEKDAYS[m.group(2)]
    ahead = (target - today.weekday()) % 7
    if mod == "next":
        return today + timedelta(days=ahead or 7)
    if mod in _PAST_MODS:
        return today - timedelta(days=(today.weekday() - target) % 7 or 7)
    return today + timedelta(days=ahead)  # bare, "this", "coming", "upcoming"


def _try_weekday_relative(s: str, today: date) -> date | None:
    """'monday after next week', 'tuesday before december 1st', ..."""
    m = re.fullmatch(
        rf"(?:the\s+)?({WEEKDAY_RE})\s+(after|following|from|before|prior to"
        r"|preceding)\s+(.+)",
        s,
    )
    if not m:
        return None
    target = WEEKDAYS[m.group(1)]
    base = _parse_expr(m.group(3), today)
    if m.group(2) in ("before", "prior to", "preceding"):
        return base - timedelta(days=(base.weekday() - target) % 7 or 7)
    return base + timedelta(days=(target - base.weekday()) % 7 or 7)


def _week_shift(word: str) -> int:
    return 1 if word in ("next", "coming") else -1 if word in _PAST_MODS else 0


def _try_weekday_in_week(s: str, today: date) -> date | None:
    """'tuesday next week', 'friday of this week', 'next week friday'."""
    wd = WEEKDAY_RE
    mod = r"(next|last|previous|prev|this|coming|past)"
    m = re.fullmatch(rf"(?:the\s+)?({wd})\s+(?:(?:of|in|on)\s+)?{mod}\s+week", s)
    if m:
        weekday_word, mod_word = m.group(1), m.group(2)
    else:
        m = re.fullmatch(rf"{mod}\s+week(?:'s|s)?\s+(?:on\s+)?({wd})", s)
        if not m:
            return None
        mod_word, weekday_word = m.group(1), m.group(2)
    monday = today - timedelta(days=today.weekday())
    monday += timedelta(days=7 * _week_shift(mod_word))
    return monday + timedelta(days=WEEKDAYS[weekday_word])


def _try_relative_period(s: str, today: date) -> date | None:
    """'next week', 'last month', 'this year', 'next fortnight' ..."""
    m = re.fullmatch(
        r"(?:the\s+)?(next|last|previous|prev|this|coming|past)\s+"
        r"(week|fortnight|month|year)",
        s,
    )
    if not m:
        return None
    shift = _week_shift(m.group(1))
    unit = m.group(2)
    if unit == "week":
        return today + timedelta(days=7 * shift)
    if unit == "fortnight":
        return today + timedelta(days=14 * shift)
    if unit == "month":
        return _add_months(today, shift)
    return _add_months(today, 12 * shift)


def _try_nth_weekday(s: str, today: date) -> date | None:
    """'first monday of december 2025', 'last friday of next month'."""
    m = re.fullmatch(
        rf"(?:the\s+)?(first|second|third|fourth|fifth|last|[1-5](?:st|nd|rd|th)?)"
        rf"\s+({WEEKDAY_RE})\s+(?:of|in)\s+(.+)",
        s,
    )
    if not m:
        return None
    ref = _month_ref(m.group(3), today)
    if ref is None:
        return None
    word = m.group(1)
    n = -1 if word == "last" else _day_number(word)
    if n is None:
        return None
    return _nth_weekday(ref[0], ref[1], WEEKDAYS[m.group(2)], n)


def _try_period_boundary(s: str, today: date) -> date | None:
    """'end of the month', 'first day of next month', 'start of 2026' ..."""
    m = re.fullmatch(
        r"(?:the\s+)?(first day|last day|start|beginning|end)\s+of\s+(.+)", s
    )
    if not m:
        return None
    is_end = m.group(1) in ("last day", "end")
    ref = re.sub(r"^the\s+", "", m.group(2))

    month = _month_ref(ref, today)
    if month is not None:
        year, mon = month
        day = calendar.monthrange(year, mon)[1] if is_end else 1
        return date(year, mon, day)

    year_match = re.fullmatch(
        r"(?:(next|last|previous|prev|this|coming)\s+)?year|(\d{4})", ref
    )
    if year_match:
        if year_match.group(2):
            year = int(year_match.group(2))
        else:
            year = today.year + _week_shift(year_match.group(1) or "this")
        return date(year, 12, 31) if is_end else date(year, 1, 1)

    week_match = re.fullmatch(r"(?:(next|last|previous|prev|this|coming)\s+)?week", ref)
    if week_match:
        monday = today - timedelta(days=today.weekday())
        monday += timedelta(days=7 * _week_shift(week_match.group(1) or "this"))
        return monday + timedelta(days=6 if is_end else 0)
    return None


def _try_day_of_month_ref(s: str, today: date) -> date | None:
    """'the 15th of next month', 'first of next month'."""
    m = re.fullmatch(r"(?:the\s+)?([a-z0-9\- ]+?)\s+of\s+(.+)", s)
    if not m:
        return None
    day = _day_number(m.group(1))
    ref = _month_ref(m.group(2), today)
    if day is None or ref is None:
        return None
    return date(ref[0], ref[1], day)


def _try_holiday(s: str, today: date) -> date | None:
    m = re.fullmatch(rf"(?:({_REL_MOD_RE})\s+)?({_HOLIDAY_RE})(?:\s+(\d{{4}}))?", s)
    if not m:
        return None
    mod, name, year_text = m.group(1), m.group(2), m.group(3)
    make = HOLIDAYS[name]
    if year_text:
        return make(int(year_text))
    this_year = make(today.year)
    if mod == "next":
        return this_year if this_year > today else make(today.year + 1)
    if mod in _PAST_MODS:
        return this_year if this_year < today else make(today.year - 1)
    return this_year


def _expand_year(year: int) -> int:
    return year if year >= 100 else (2000 + year if year < 70 else 1900 + year)


def _try_absolute(s: str, today: date) -> date | None:
    """Concrete calendar dates in many common spellings."""
    text = re.sub(r"\s+", " ", s.replace(",", " ")).strip()

    # --- numeric formats ---------------------------------------------------
    m = re.fullmatch(r"(\d{4})([-/.])(\d{1,2})\2(\d{1,2})", text)
    if m:
        return date(int(m.group(1)), int(m.group(3)), int(m.group(4)))
    m = re.fullmatch(r"(\d{1,2})([-/.])(\d{1,2})\2(\d{4}|\d{2})", text)
    if m:
        a, sep, b, year = int(m.group(1)), m.group(2), int(m.group(3)), m.group(4)
        year_num = _expand_year(int(year))
        if sep == "." or (a > 12 and b <= 12):  # day-first
            return date(year_num, b, a)
        return date(year_num, a, b)
    m = re.fullmatch(r"(\d{1,2})/(\d{1,2})", text)
    if m:
        return date(today.year, int(m.group(1)), int(m.group(2)))
    m = re.fullmatch(r"(\d{4})(\d{2})(\d{2})", text)
    if m:
        return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))

    # --- spelled-out months ------------------------------------------------
    text = text.replace(".", " ")
    text = re.sub(r"\b(\d{1,2})(?:st|nd|rd|th)\b", r"\1", text)
    text = _ORDINAL_WORD_RE.sub(lambda mm: str(ORDINALS[mm.group(0)]), text)
    text = re.sub(r"\b(?:the|of|on)\b", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(rf"^{WEEKDAY_RE}\s+(?=\S)", "", text)

    m = re.fullmatch(rf"({MONTH_RE})\s+(\d{{1,2}})(?:\s+(\d{{4}}))?", text)
    if m:
        year = int(m.group(3)) if m.group(3) else today.year
        return date(year, MONTHS[m.group(1)], int(m.group(2)))
    m = re.fullmatch(rf"(\d{{1,2}})\s+({MONTH_RE})(?:\s+(\d{{4}}))?", text)
    if m:
        year = int(m.group(3)) if m.group(3) else today.year
        return date(year, MONTHS[m.group(2)], int(m.group(1)))
    m = re.fullmatch(rf"(\d{{4}})\s+({MONTH_RE})\s+(\d{{1,2}})", text)
    if m:
        return date(int(m.group(1)), MONTHS[m.group(2)], int(m.group(3)))
    m = re.fullmatch(rf"({MONTH_RE})\s+(\d{{4}})", text)
    if m:
        return date(int(m.group(2)), MONTHS[m.group(1)], 1)
    return None


_HANDLERS: tuple[Callable[[str, date], date | None], ...] = (
    _try_keyword,
    _try_offset,
    _try_weekday,
    _try_weekday_relative,
    _try_weekday_in_week,
    _try_relative_period,
    _try_nth_weekday,
    _try_period_boundary,
    _try_day_of_month_ref,
    _try_holiday,
    _try_absolute,
)


def _parse_expr(s: str, today: date) -> date:
    s = re.sub(r"^on\s+", "", s)
    for handler in _HANDLERS:
        result = handler(s, today)
        if result is not None:
            return result
    raise ValueError(f"unrecognized date: {s!r}")


def _normalize(s: str) -> str:
    s = s.lower().replace("\u2019", "").replace("'", "")
    s = re.sub(r"\s+", " ", s).strip()
    return s.strip(' "!?;:').rstrip(".").strip()


def parse(s: str, today: date | None = None) -> date:
    """Convert a natural-language date such as ``"next Tuesday"`` to a date.

    ``today`` is the reference point for relative expressions and defaults to
    the current date.  Raises ``ValueError`` for anything unrecognized.
    """
    if not isinstance(s, str):
        raise TypeError(f"expected a string, got {type(s).__name__}")
    if today is None:
        today = date.today()
    elif isinstance(today, datetime):
        today = today.date()
    text = _normalize(s)
    if not text:
        raise ValueError("empty date string")
    try:
        return _parse_expr(text, today)
    except OverflowError as exc:
        raise ValueError(f"date out of range: {s!r}") from exc
