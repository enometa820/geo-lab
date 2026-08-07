"""회귀가 **틀린 숫자를 내놓지 않는지** 검증한다.

이 파일이 지키는 것은 계수의 정확도가 아니라 정직성이다. 완전분리에서 숫자를
뱉지 않는가, 군집이 적을 때 그 사실을 말하는가, 추정 못 한 브랜드를 조용히
빼지 않는가.
"""

import pytest

from geolab.modeling import (
    MIN_CLUSTERS_FOR_ROBUST_SE,
    find_empty_cells,
    fit_many,
    fit_stage_language,
)


def row(prompt_id: str, stage: str, language: str, mentioned: int) -> dict:
    return {
        "prompt_id": prompt_id,
        "stage": stage,
        "language": language,
        "mentioned": mentioned,
    }


def balanced_rows(n_clusters: int = 40, per_cluster: int = 10) -> list[dict]:
    """분리가 없고 군집이 넉넉한 데이터.

    단계와 언어가 모두 결과에 영향을 주되, 어떤 수준도 0 또는 전부가 되지 않게
    만든다. 그래야 적합이 실제로 성립한다.
    """
    rows = []
    for c in range(n_clusters):
        stage = ["discovery", "constrained", "problem"][c % 3]
        language = ["ko", "en"][c % 2]
        # 수준마다 언급 비율을 다르게 하되 0과 1은 피한다
        base = {"discovery": 4, "constrained": 7, "problem": 2}[stage]
        hits = base + (1 if language == "ko" else 0)
        for i in range(per_cluster):
            rows.append(row(f"p{c}", stage, language, 1 if i < hits else 0))
    return rows


class TestEmptyCellDetection:
    def test_zero_event_level_is_found(self):
        rows = [
            row("a", "discovery", "ko", 1),
            row("a", "discovery", "ko", 0),
            row("b", "alternative", "ko", 0),
            row("b", "alternative", "ko", 0),
        ]
        found = {(c.variable, c.level) for c in find_empty_cells(rows)}

        assert ("stage", "alternative") in found

    def test_all_event_level_is_also_separation(self):
        """전부 언급된 수준도 분리를 만든다. 0만 보면 절반을 놓친다."""
        rows = [
            row("a", "discovery", "ko", 0),
            row("a", "discovery", "ko", 1),
            row("b", "comparison", "ko", 1),
            row("b", "comparison", "ko", 1),
        ]
        found = {(c.variable, c.level) for c in find_empty_cells(rows)}

        assert ("stage", "comparison") in found

    def test_mixed_level_is_not_flagged(self):
        rows = [
            row("a", "discovery", "ko", 1),
            row("a", "discovery", "ko", 0),
            row("b", "discovery", "en", 1),
            row("b", "discovery", "en", 0),
        ]

        assert find_empty_cells(rows, variables=("stage",)) == ()

    def test_reports_counts_for_explanation(self):
        rows = [row("b", "alternative", "ko", 0) for _ in range(20)]
        cell = find_empty_cells(rows, variables=("stage",))[0]

        assert cell.n == 20
        assert cell.events == 0
        assert "20건" in cell.describe()


class TestSeparationHandling:
    def test_separation_makes_fit_inestimable(self):
        """0회인 단계가 있으면 계수를 내놓지 않는다."""
        rows = (
            [row(f"p{i}", "discovery", "ko", i % 2) for i in range(10)]
            + [row(f"q{i}", "alternative", "ko", 0) for i in range(10)]
        )
        fit = fit_stage_language(rows, "Swit")

        assert fit.estimable is False
        assert fit.terms == ()
        assert any(c.level == "alternative" for c in fit.separation)

    def test_inestimable_fit_still_reports_counts(self):
        """추정을 못 해도 무엇을 봤는지는 남는다."""
        rows = [row(f"p{i}", "alternative", "ko", 0) for i in range(20)]
        fit = fit_stage_language(rows, "Swit")

        assert fit.n_obs == 20
        assert fit.n_events == 0
        assert "추정 불가" in fit.describe()

    def test_empty_rows_raise(self):
        with pytest.raises(ValueError):
            fit_stage_language([], "Swit")

    def test_confounded_design_fails_loudly_not_silently(self):
        """단계와 언어가 완전히 겹치면 두 효과를 분리할 수 없다.

        설계행렬이 특이해져 적합이 깨지는데, 이때 예외를 밖으로 던지면 리포트
        생성이 통째로 멈춘다. 대신 추정 불가로 표시해서 나머지 분석은 살린다.
        """
        rows = (
            [row(f"p{i}", "discovery", "ko", i % 2) for i in range(20)]
            + [row(f"q{i}", "constrained", "en", (i + 1) % 2) for i in range(20)]
        )
        fit = fit_stage_language(rows, "Swit")

        assert fit.estimable is False
        assert fit.terms == ()
        assert fit.note is not None


class TestClusterCaveat:
    def test_few_clusters_produce_caveat(self):
        """프롬프트 18개는 실제 측정과 같은 조건이다."""
        fit = fit_stage_language(balanced_rows(n_clusters=18), "Swit")

        assert fit.n_clusters < MIN_CLUSTERS_FOR_ROBUST_SE
        assert fit.cluster_caveat is not None
        assert "robust SE" in fit.cluster_caveat

    def test_many_clusters_have_no_caveat(self):
        fit = fit_stage_language(balanced_rows(n_clusters=40), "Swit")

        assert fit.n_clusters >= MIN_CLUSTERS_FOR_ROBUST_SE
        assert fit.cluster_caveat is None


class TestSuccessfulFit:
    def test_fit_returns_terms_with_intervals(self):
        fit = fit_stage_language(balanced_rows(), "Swit")

        assert fit.estimable is True
        names = {t.name for t in fit.terms}
        assert "const" in names
        assert "language[ko]" in names
        assert any(n.startswith("stage[") for n in names)

    def test_reference_levels_are_absent_from_terms(self):
        """기준 수준은 항으로 나오지 않는다. 나오면 설계행렬이 특이해진다."""
        fit = fit_stage_language(balanced_rows(), "Swit")
        names = {t.name for t in fit.terms}

        assert "stage[discovery]" not in names
        assert "language[en]" not in names

    def test_odds_ratio_is_exponentiated_coefficient(self):
        fit = fit_stage_language(balanced_rows(), "Swit")
        term = next(t for t in fit.terms if t.name == "language[ko]")
        low, high = term.or_ci

        assert term.odds_ratio == pytest.approx(pow(2.718281828459045, term.coef), rel=1e-6)
        assert low < term.odds_ratio < high

    def test_constrained_stage_has_higher_odds_than_reference(self):
        """합성 데이터에서 constrained 의 언급률을 더 높게 만들었다."""
        fit = fit_stage_language(balanced_rows(), "Swit")
        term = next(t for t in fit.terms if t.name == "stage[constrained]")

        assert term.odds_ratio > 1.0

    def test_custom_reference_changes_term_set(self):
        fit = fit_stage_language(balanced_rows(), "Swit", stage_reference="problem")
        names = {t.name for t in fit.terms}

        assert "stage[discovery]" in names
        assert "stage[problem]" not in names


class TestStageExclusion:
    """무엇을 빼고 적합했는지가 결과에 남아야 한다."""

    def test_excluded_stage_is_recorded_on_the_fit(self):
        rows = balanced_rows(n_clusters=40)
        fit = fit_stage_language(rows, "Swit", exclude_stages=("problem",))

        assert fit.excluded_stages == ("problem",)
        assert "problem" in fit.describe()

    def test_excluded_rows_leave_the_sample(self):
        rows = balanced_rows(n_clusters=40)
        full = fit_stage_language(rows, "Swit")
        trimmed = fit_stage_language(rows, "Swit", exclude_stages=("problem",))

        assert trimmed.n_obs < full.n_obs
        assert not any(t.name == "stage[problem]" for t in trimmed.terms)

    def test_excluding_everything_raises(self):
        rows = balanced_rows(n_clusters=6)
        with pytest.raises(ValueError):
            fit_stage_language(
                rows, "Swit", exclude_stages=("discovery", "constrained", "problem")
            )

    def test_exclusion_can_resolve_separation(self):
        """분리를 일으키던 단계를 빼면 나머지는 추정할 수 있다."""
        rows = balanced_rows(n_clusters=40) + [
            row(f"z{i}", "comparison", "ko", 0) for i in range(20)
        ]

        assert fit_stage_language(rows, "Swit").estimable is False
        assert fit_stage_language(rows, "Swit", exclude_stages=("comparison",)).estimable is True

    def test_fit_many_passes_exclusion_through(self):
        fits = fit_many({"Strong": balanced_rows(n_clusters=40)}, exclude_stages=("problem",))

        assert fits[0].excluded_stages == ("problem",)


class TestFitMany:
    def test_inestimable_brands_are_kept(self):
        """추정 못 한 브랜드를 빼면 약한 브랜드가 조용히 사라진다."""
        fits = fit_many(
            {
                "Strong": balanced_rows(),
                "Weak": [row(f"p{i}", "alternative", "ko", 0) for i in range(20)],
            }
        )
        by_brand = {f.brand: f for f in fits}

        assert set(by_brand) == {"Strong", "Weak"}
        assert by_brand["Weak"].estimable is False

    def test_estimable_fits_come_first(self):
        fits = fit_many(
            {
                "Weak": [row(f"p{i}", "alternative", "ko", 0) for i in range(20)],
                "Strong": balanced_rows(),
            }
        )

        assert fits[0].brand == "Strong"

    def test_empty_row_sets_are_skipped(self):
        fits = fit_many({"Strong": balanced_rows(), "Nothing": []})

        assert {f.brand for f in fits} == {"Strong"}
