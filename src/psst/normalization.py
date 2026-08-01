"""Versioned Persian text normalization."""

import re
import unicodedata
from collections.abc import Callable

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


def normalize_fa_v1(text: str) -> str:
    """Apply the immutable ``fa-v1`` normalization policy."""
    normalized = unicodedata.normalize("NFKC", text).translate(_VARIANTS).translate(_DIGITS)
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


NORMALIZERS: dict[str, Callable[[str], str]] = {"fa-v1": normalize_fa_v1}


def normalize(text: str, version: str) -> str:
    """Normalize text with a registered immutable policy."""
    try:
        return NORMALIZERS[version](text)
    except KeyError as error:
        raise ValueError(f"Unknown normalization version: {version}") from error
