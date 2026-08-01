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


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("۱۹۶۷", "هزار و نهصد و شصت و هفت"),
        ("در ۶,۰۰۰ مورد", "در شش هزار مورد"),
        ("شدت ۶.۵ ریشتر", "شدت شش ممیز پنج ریشتر"),
        ("ساعت 06:30", "ساعت شش و سی"),
        ("نسبت 3:2", "نسبت سه به دو"),
        ("ابعاد 29¾ در 24½", "ابعاد بیست و نه و سه چهارم در بیست و چهار و نیم"),
        ("1/5 اینچ", "یک پنجم اینچ"),
        ("۰ و ١٢", "صفر و دوازده"),
    ],
)
def test_fa_v2_spells_out_numerals(source: str, expected: str) -> None:
    assert normalize(source, "fa-v2") == expected


def test_fa_v1_remains_immutable_when_fa_v2_is_added() -> None:
    assert normalize("۱۹۶۷", "fa-v1") == "1967"


def test_fa_v2_canonicalizes_oversized_model_hallucinations_without_failing() -> None:
    assert normalize("۱" * 19, "fa-v2") == "1" * 19


def test_unknown_normalizer_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown normalization version"):
        normalize("متن", "fa-unknown")
