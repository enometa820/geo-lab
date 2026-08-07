"""신뢰구간 계산 검증.

가장 중요한 테스트는 `test_zero_successes_does_not_claim_certainty` 다.
이 저장소의 존재 이유가 "측정을 정직하게 한다"이고, 0/n 에서 폭 0짜리 구간을
내는 것이 그 원칙을 깨는 가장 흔한 방식이다.
"""

import pytest

from geolab.stats import Interval, overlaps, runs_for_margin, wilson_interval


class TestWilsonInterval:
    def test_zero_successes_does_not_claim_certainty(self):
        """0/20 은 "확실히 0%"가 아니다. Wald 였다면 폭 0을 냈을 지점."""
        ci = wilson_interval(0, 20)

        assert ci.point == 0.0
        assert ci.low == 0.0
        assert ci.high > 0.15, "20번 봐서 0번이어도 상한은 15%를 넘어야 한다"
        assert ci.high < 0.20

    def test_all_successes_does_not_claim_certainty(self):
        """20/20 도 마찬가지로 "확실히 100%"가 아니다."""
        ci = wilson_interval(20, 20)

        assert ci.point == 1.0
        assert ci.high == 1.0
        assert ci.low < 0.85

    def test_half_successes(self):
        ci = wilson_interval(10, 20)

        assert ci.point == 0.5
        assert ci.low < 0.5 < ci.high
        # 대칭은 아니지만 n=20, p=0.5 에서는 거의 대칭이다
        assert abs((0.5 - ci.low) - (ci.high - 0.5)) < 0.01

    def test_more_samples_tighten_the_interval(self):
        """표본이 늘면 구간이 좁아진다. 이게 반복 측정을 하는 이유다."""
        few = wilson_interval(5, 10)
        many = wilson_interval(50, 100)

        assert few.point == many.point == 0.5
        assert many.margin < few.margin

    def test_higher_confidence_widens_the_interval(self):
        ci_90 = wilson_interval(10, 20, confidence=0.90)
        ci_95 = wilson_interval(10, 20, confidence=0.95)
        ci_99 = wilson_interval(10, 20, confidence=0.99)

        assert ci_90.margin < ci_95.margin < ci_99.margin

    def test_no_samples_means_we_know_nothing(self):
        ci = wilson_interval(0, 0)

        assert ci.low == 0.0
        assert ci.high == 1.0
        assert ci.n == 0

    def test_interval_stays_within_zero_and_one(self):
        for successes, n in [(0, 1), (1, 1), (1, 3), (2, 3), (0, 5), (5, 5)]:
            ci = wilson_interval(successes, n, confidence=0.99)
            assert 0.0 <= ci.low <= ci.high <= 1.0, f"{successes}/{n} 에서 범위 이탈"

    @pytest.mark.parametrize(
        "successes,n",
        [(-1, 10), (11, 10), (5, -1)],
    )
    def test_invalid_counts_raise(self, successes, n):
        with pytest.raises(ValueError):
            wilson_interval(successes, n)

    @pytest.mark.parametrize("confidence", [0.0, 1.0, -0.5, 1.5])
    def test_invalid_confidence_raises(self, confidence):
        with pytest.raises(ValueError):
            wilson_interval(5, 10, confidence=confidence)


class TestFormatting:
    def test_format_pct_includes_sample_size(self):
        """표본 수 없는 비율은 근거가 아니다. 출력에 항상 n 이 붙어야 한다."""
        text = wilson_interval(5, 20).format_pct()

        assert "25.0%" in text
        assert "n=20" in text
        assert "CI" in text

    def test_margin_is_half_the_width(self):
        ci = Interval(point=0.5, low=0.3, high=0.7, n=20)

        assert ci.margin == pytest.approx(0.2)


class TestRunsForMargin:
    def test_tighter_margin_needs_more_runs(self):
        assert runs_for_margin(0.05) > runs_for_margin(0.10)

    def test_known_value(self):
        """±5pp, 95%, p=0.5 는 표본 385 가 필요하다 (여론조사의 그 숫자)."""
        assert runs_for_margin(0.05) == 385

    def test_expected_rate_near_zero_needs_fewer_runs(self):
        """언급률이 낮을 것으로 예상되면 필요 표본이 준다. 분산이 작아서다."""
        assert runs_for_margin(0.05, p_expected=0.05) < runs_for_margin(0.05, p_expected=0.5)

    @pytest.mark.parametrize("margin", [0.0, 1.0, -0.1])
    def test_invalid_margin_raises(self, margin):
        with pytest.raises(ValueError):
            runs_for_margin(margin)


class TestOverlaps:
    def test_clearly_separated_intervals_do_not_overlap(self):
        a = Interval(point=0.8, low=0.7, high=0.9, n=100)
        b = Interval(point=0.2, low=0.1, high=0.3, n=100)

        assert not overlaps(a, b)

    def test_touching_intervals_overlap(self):
        a = Interval(point=0.5, low=0.4, high=0.6, n=20)
        b = Interval(point=0.7, low=0.6, high=0.8, n=20)

        assert overlaps(a, b)

    def test_small_samples_usually_overlap(self):
        """n=20 에서 40% 대 60% 는 구간이 겹친다.

        점추정치만 보면 "저쪽이 1.5배 많다"고 말하고 싶어지지만, 그렇게 말할 수 없다.
        이 분야에서 순위표가 만들어지는 방식이 정확히 이 실수다.
        """
        a = wilson_interval(8, 20)
        b = wilson_interval(12, 20)

        assert overlaps(a, b)
