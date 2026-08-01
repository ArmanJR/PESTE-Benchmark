"""Versioned Persian text normalization."""

import logging
import re
import unicodedata
from collections.abc import Callable

LOGGER = logging.getLogger(__name__)

_VARIANTS = str.maketrans(
    {
        "ك": "ک",
        "ي": "ی",
        "ى": "ی",
        "ے": "ی",
        "ۍ": "ی",
        "ة": "ه",
        "ۀ": "ه",
        "ە": "ه",
        "ؤ": "و",
        "إ": "ا",
        "أ": "ا",
        "ٱ": "ا",
    }
)
_DIGITS = str.maketrans("۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩", "01234567890123456789")
_DIACRITIC_RANGES = re.compile("[\u0610-\u061a\u064b-\u065f\u0670\u06d6-\u06ed]")
_WHITESPACE = re.compile(r"\s+")
_VULGAR_FRACTION = re.compile(r"(\d+)?([¼½¾])")
_CLOCK_TIME = re.compile(r"(?<!\d)(\d{1,2}):(\d{2})(?!\d)")
_RATIO = re.compile(r"(?<!\d)(\d+):(\d+)(?!\d)")
_SLASH_FRACTION = re.compile(r"(?<!\d)(\d+)/(\d+)(?!\d)")
_NUMERAL = re.compile(r"\d{1,3}(?:[,،٬]\d{3})+(?:[.٫]\d+)?|\d+[.٫]\d+|\d+")
_GROUP_SEPARATOR = re.compile("[,،٬]")
_DECIMAL_SEPARATOR = re.compile("[.٫]")

_ONES = (
    "صفر",
    "یک",
    "دو",
    "سه",
    "چهار",
    "پنج",
    "شش",
    "هفت",
    "هشت",
    "نه",
    "ده",
    "یازده",
    "دوازده",
    "سیزده",
    "چهارده",
    "پانزده",
    "شانزده",
    "هفده",
    "هجده",
    "نوزده",
)
_TENS = ("", "", "بیست", "سی", "چهل", "پنجاه", "شصت", "هفتاد", "هشتاد", "نود")
_HUNDREDS = (
    "",
    "صد",
    "دویست",
    "سیصد",
    "چهارصد",
    "پانصد",
    "ششصد",
    "هفتصد",
    "هشتصد",
    "نهصد",
)
_SCALES = ("", "هزار", "میلیون", "میلیارد", "تریلیون", "کوادریلیون")
_VULGAR_FRACTIONS = {"¼": "یک چهارم", "½": "نیم", "¾": "سه چهارم"}
_FRACTION_DENOMINATORS = {
    "2": "دوم",
    "3": "سوم",
    "4": "چهارم",
    "5": "پنجم",
    "6": "ششم",
    "7": "هفتم",
    "8": "هشتم",
    "9": "نهم",
    "10": "دهم",
}


def _spell_below_thousand(value: int) -> str:
    parts: list[str] = []
    hundreds, remainder = divmod(value, 100)
    if hundreds:
        parts.append(_HUNDREDS[hundreds])
    if remainder:
        if remainder < 20:
            parts.append(_ONES[remainder])
        else:
            tens, ones = divmod(remainder, 10)
            parts.append(_TENS[tens])
            if ones:
                parts.append(_ONES[ones])
    return " و ".join(parts)


def _spell_integer(value: int) -> str:
    if value == 0:
        return _ONES[0]
    groups: list[int] = []
    while value:
        value, group = divmod(value, 1000)
        groups.append(group)
    if len(groups) > len(_SCALES):
        raise ValueError("fa-v2 supports numerals below 10^18")
    parts: list[str] = []
    for scale_index in range(len(groups) - 1, -1, -1):
        group = groups[scale_index]
        if not group:
            continue
        scale = _SCALES[scale_index]
        if scale_index == 1 and group == 1:
            parts.append(scale)
            continue
        spelled_group = _spell_below_thousand(group)
        parts.append(f"{spelled_group} {scale}".rstrip())
    return " و ".join(parts)


def _spell_integer_digits(digits: str) -> str:
    significant_digits = digits.lstrip("0") or "0"
    if len(significant_digits) <= 18:
        return _spell_integer(int(significant_digits))
    LOGGER.warning(
        "Leaving oversized numeral digit-canonicalized",
        extra={"normalization_version": "fa-v2", "digits": len(digits)},
    )
    return digits


def _expand_vulgar_fraction(match: re.Match[str]) -> str:
    whole_number, fraction = match.groups()
    spelled_fraction = _VULGAR_FRACTIONS[fraction]
    if whole_number is None:
        return f" {spelled_fraction} "
    return f" {whole_number} و {spelled_fraction} "


def _expand_clock_time(match: re.Match[str]) -> str:
    hours, minutes = match.groups()
    return f" {hours} و {minutes} "


def _expand_ratio(match: re.Match[str]) -> str:
    left, right = match.groups()
    return f" {left} به {right} "


def _expand_slash_fraction(match: re.Match[str]) -> str:
    numerator, denominator = match.groups()
    denominator_word = _FRACTION_DENOMINATORS.get(denominator)
    if denominator_word is None:
        return match.group()
    return f" {numerator} {denominator_word} "


def _spell_numeral(match: re.Match[str]) -> str:
    numeral = match.group()
    decimal_match = _DECIMAL_SEPARATOR.search(numeral)
    if decimal_match is None:
        integer_digits = _GROUP_SEPARATOR.sub("", numeral)
        return f" {_spell_integer_digits(integer_digits)} "
    integer_text = numeral[: decimal_match.start()]
    fractional_digits = numeral[decimal_match.end() :]
    integer_digits = _GROUP_SEPARATOR.sub("", integer_text)
    if fractional_digits.startswith("0"):
        fractional = " ".join(_ONES[int(digit)] for digit in fractional_digits)
    else:
        fractional = _spell_integer_digits(fractional_digits)
    return f" {_spell_integer_digits(integer_digits)} ممیز {fractional} "


def _normalize_common(text: str, *, spell_numerals: bool) -> str:
    normalized = text.translate(_DIGITS)
    if spell_numerals:
        normalized = _VULGAR_FRACTION.sub(_expand_vulgar_fraction, normalized)
    normalized = unicodedata.normalize("NFKC", normalized).translate(_VARIANTS).translate(_DIGITS)
    if spell_numerals:
        normalized = _CLOCK_TIME.sub(_expand_clock_time, normalized)
        normalized = _RATIO.sub(_expand_ratio, normalized)
        normalized = _SLASH_FRACTION.sub(_expand_slash_fraction, normalized)
        normalized, replacements = _NUMERAL.subn(_spell_numeral, normalized)
        if replacements:
            LOGGER.debug(
                "Expanded numeral expressions during Persian normalization",
                extra={"normalization_version": "fa-v2", "replacements": replacements},
            )
    normalized = normalized.replace("ـ", "")
    normalized = _DIACRITIC_RANGES.sub("", normalized)
    output: list[str] = []
    for character in normalized:
        category = unicodedata.category(character)
        if (
            character in {"\u200c", "\u200d", "\u00a0"}
            or category[0] in {"P", "S"}
            or category in {"Cf", "Cc", "Zl", "Zp", "Zs"}
        ):
            output.append(" ")
        else:
            output.append(character)
    return _WHITESPACE.sub(" ", "".join(output)).strip()


def normalize_fa_v1(text: str) -> str:
    """Apply the immutable ``fa-v1`` normalization policy."""
    return _normalize_common(text, spell_numerals=False)


def normalize_fa_v2(text: str) -> str:
    """Apply ``fa-v1`` plus Persian spell-out normalization for numeric expressions."""
    return _normalize_common(text, spell_numerals=True)


NORMALIZERS: dict[str, Callable[[str], str]] = {
    "fa-v1": normalize_fa_v1,
    "fa-v2": normalize_fa_v2,
}


def normalize(text: str, version: str) -> str:
    """Normalize text with a registered immutable policy."""
    try:
        return NORMALIZERS[version](text)
    except KeyError as error:
        raise ValueError(f"Unknown normalization version: {version}") from error
