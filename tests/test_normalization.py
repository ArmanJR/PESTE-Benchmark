"""Tests for the immutable Persian normalizer."""

import pytest

from peste.normalization import normalize, normalize_fa_v1


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("ك ي ى ے ة ۀ ە", "ک ی ی ی ه ه ه"),
        ("سَلامـٌ", "سلام"),
        ("۱۲۳ ٤٥٦", "123 456"),
        ("می\u200cروم\u00a0خانه", "می روم خانه"),
        ("سلام، دنیا! ۵۰٪", "سلام دنیا 50"),
        ("  الف\n\tب  ", "الف ب"),
        ("ﻛ", "ک"),
    ],
)
def test_fa_v1(source: str, expected: str) -> None:
    assert normalize_fa_v1(source) == expected


def test_unknown_normalizer_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown normalization version"):
        normalize("متن", "fa-v2")
