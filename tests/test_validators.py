"""Tests for the output validator."""

from __future__ import annotations

import pytest

from app.errors import InvalidProviderOutputError
from app.validators.output import (
    ValidatedOutput,
    count_words,
    normalize_output,
    validate_generated_text,
)


# A small but representative Vietnamese description used across the tests.
SAMPLE_VIETNAMESE = (
    "Aurora's Echo là một trò chơi khám phá thế giới mở theo phong cách "
    "phiêu lưu, lấy bối cảnh một quần đảo băng giá. Người chơi sẽ vào vai "
    "một nhà thám hiểm trẻ, sử dụng ánh sáng cực quang để mở khoá những "
    "khu vực bí ẩn, gặp gỡ các nhân vật giàu chiều sâu và khám phá câu "
    "chuyện về truyền thuyết bị lãng quên. Trò chơi tập trung vào nhịp "
    "chơi chậm, khám phá không gian và đối thoại sâu sắc thay vì hành "
    "động dồn dập. Đồ họa thủ công kết hợp âm nhạc giao hưởng tạo nên "
    "một bầu không khí yên bình nhưng đầy bí ẩn, phù hợp với người chơi "
    "yêu thích thể loại tường thuật và khám phá thế giới."
)


class TestNormalize:
    """Whitespace and line-ending normalisation."""

    def test_strips_surrounding_whitespace(self) -> None:
        assert normalize_output("  hello world  ") == "hello world"

    def test_collapses_horizontal_whitespace(self) -> None:
        assert normalize_output("hello   world\t\tfriend") == "hello world friend"

    def test_preserves_newlines(self) -> None:
        assert normalize_output("para 1\n\npara 2") == "para 1\n\npara 2"

    def test_normalises_windows_line_endings(self) -> None:
        assert normalize_output("a\r\nb\r\nc") == "a\nb\nc"

    def test_empty_string(self) -> None:
        assert normalize_output("") == ""


class TestCountWords:
    """Word counter behaviour."""

    def test_empty(self) -> None:
        assert count_words("") == 0

    def test_whitespace_only(self) -> None:
        assert count_words("   \n\t  ") == 0

    def test_simple(self) -> None:
        assert count_words("one two three") == 3

    def test_vietnamese(self) -> None:
        # Vietnamese is written with single spaces between syllables,
        # so the same whitespace-splitter that counts English words
        # counts Vietnamese syllables. The number is the deterministic
        # count of whitespace-separated tokens, not the linguistic
        # word count.
        assert count_words("Một trò chơi khám phá thú vị") == 7


class TestValidate:
    """End-to-end validation of a provider response."""

    def test_happy_path(self) -> None:
        result = validate_generated_text(
            SAMPLE_VIETNAMESE,
            max_words=400,
            min_target_words=120,
            max_target_words=200,
        )
        assert isinstance(result, ValidatedOutput)
        # The sample is ~150 words, which is inside the soft range.
        assert result.word_count >= 120
        assert result.within_target_range is True
        # Text is preserved after normalisation.
        assert result.text.startswith("Aurora's Echo là")

    def test_strips_surrounding_whitespace(self) -> None:
        result = validate_generated_text(
            "   " + SAMPLE_VIETNAMESE + "\n\n",
            max_words=400,
            min_target_words=120,
            max_target_words=200,
        )
        assert result.text == SAMPLE_VIETNAMESE

    def test_rejects_none(self) -> None:
        with pytest.raises(InvalidProviderOutputError):
            validate_generated_text(
                None,
                max_words=400,
                min_target_words=120,
                max_target_words=200,
            )

    def test_rejects_empty_string(self) -> None:
        with pytest.raises(InvalidProviderOutputError):
            validate_generated_text(
                "",
                max_words=400,
                min_target_words=120,
                max_target_words=200,
            )

    def test_rejects_whitespace_only(self) -> None:
        with pytest.raises(InvalidProviderOutputError):
            validate_generated_text(
                "   \n\t  ",
                max_words=400,
                min_target_words=120,
                max_target_words=200,
            )

    def test_rejects_non_string(self) -> None:
        with pytest.raises(InvalidProviderOutputError):
            validate_generated_text(
                ["not", "a", "string"],
                max_words=400,
                min_target_words=120,
                max_target_words=200,
            )

    def test_rejects_output_exceeding_max_words(self) -> None:
        with pytest.raises(InvalidProviderOutputError) as exc:
            validate_generated_text(
                SAMPLE_VIETNAMESE,
                max_words=10,
                min_target_words=120,
                max_target_words=200,
            )
        # The error must not contain any provider-specific content.
        assert "Aurora" not in str(exc.value)

    def test_short_output_is_kept(self) -> None:
        # The 120–200-word target is a soft target. Short, readable
        # Vietnamese output must be accepted and flagged as outside
        # the target range.
        result = validate_generated_text(
            "Một trò chơi phiêu lưu ngắn.",
            max_words=400,
            min_target_words=120,
            max_target_words=200,
        )
        assert result.within_target_range is False
        assert result.text == "Một trò chơi phiêu lưu ngắn."

    def test_exact_target_range_lower_bound(self) -> None:
        text = " ".join(["từ"] * 120)
        result = validate_generated_text(
            text,
            max_words=400,
            min_target_words=120,
            max_target_words=200,
        )
        assert result.word_count == 120
        assert result.within_target_range is True

    def test_exact_target_range_upper_bound(self) -> None:
        text = " ".join(["từ"] * 200)
        result = validate_generated_text(
            text,
            max_words=400,
            min_target_words=120,
            max_target_words=200,
        )
        assert result.word_count == 200
        assert result.within_target_range is True

    def test_silent_truncation_is_prevented(self) -> None:
        # 11 words vs max_words=10: the validator must raise, not
        # return the first 10 words.
        text = " ".join(["từ"] * 11)
        with pytest.raises(InvalidProviderOutputError):
            validate_generated_text(
                text,
                max_words=10,
                min_target_words=5,
                max_target_words=8,
            )
