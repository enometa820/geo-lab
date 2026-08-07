"""군집화 검증 — API 를 타지 않는 부분만.

이 파일이 지키는 것은 군집의 품질이 아니라 **군집을 과잉 해석하지 않게 만드는
장치들**이다. 약한 구조를 약하다고 말하는가, 후보 점수를 숨기지 않는가, 행과
라벨이 어긋났을 때 조용히 넘어가지 않는가.
"""

import numpy as np
import pytest

from geolab.embedding import (
    ClusterChoice,
    ClusterResult,
    brand_by_cluster,
    choose_clusters,
    has_natural_k,
    label_agreement,
    profile_clusters,
    text_key,
)


def row(prompt_id, stage, language, mentions, length=1000):
    return {
        "prompt_id": prompt_id,
        "run_index": 0,
        "stage": stage,
        "language": language,
        "text": "x" * length,
        "length": length,
        "mentions": mentions,
        "ranks": {b: i + 1 for i, b in enumerate(mentions)},
    }


def separated_vectors(n_per: int = 20, seed: int = 0) -> np.ndarray:
    """뚜렷하게 갈라진 두 덩어리. 실루엣이 높게 나와야 한다."""
    rng = np.random.default_rng(seed)
    a = rng.normal(loc=0.0, scale=0.05, size=(n_per, 8))
    b = rng.normal(loc=5.0, scale=0.05, size=(n_per, 8))
    return np.vstack([a, b])


def blob_vectors(n: int = 40, seed: int = 0) -> np.ndarray:
    """경계가 없는 한 덩어리. 나누라고 하면 나뉘지만 의미는 없다."""
    rng = np.random.default_rng(seed)
    return rng.normal(loc=0.0, scale=1.0, size=(n, 8))


class TestCacheKey:
    def test_same_text_same_model_is_stable(self):
        assert text_key("협업 툴 추천", "m1") == text_key("협업 툴 추천", "m1")

    def test_different_model_gives_different_key(self):
        """모델이 다르면 벡터도 다르다. 캐시를 공유하면 안 된다."""
        assert text_key("협업 툴 추천", "m1") != text_key("협업 툴 추천", "m2")

    def test_different_text_gives_different_key(self):
        assert text_key("a", "m") != text_key("b", "m")


class TestChooseClusters:
    def test_finds_two_groups_in_separated_data(self):
        result = choose_clusters(separated_vectors(), k_range=(2, 3, 4))

        assert result.k == 2
        assert result.silhouette > 0.5
        assert result.weak_structure is False

    def test_labels_align_with_input_rows(self):
        vectors = separated_vectors(n_per=15)
        result = choose_clusters(vectors, k_range=(2, 3))

        assert len(result.labels) == vectors.shape[0]

    def test_all_candidate_scores_are_returned(self):
        """최고점만 보고하면 임의의 선택이 근거처럼 읽힌다."""
        result = choose_clusters(separated_vectors(), k_range=(2, 3, 4))

        assert {c.k for c in result.candidates} == {2, 3, 4}

    def test_weak_structure_is_flagged_on_a_single_blob(self):
        result = choose_clusters(blob_vectors(), k_range=(2, 3, 4))

        assert result.weak_structure is True
        assert "근거로 쓰지 않는다" in result.describe()

    def test_result_is_reproducible(self):
        vectors = blob_vectors()
        first = choose_clusters(vectors, k_range=(2, 3, 4))
        second = choose_clusters(vectors, k_range=(2, 3, 4))

        assert first.labels == second.labels

    def test_k_larger_than_sample_is_skipped(self):
        result = choose_clusters(separated_vectors(n_per=2), k_range=(2, 3, 99))

        assert result.k in (2, 3)

    def test_impossible_k_range_raises(self):
        with pytest.raises(ValueError):
            choose_clusters(blob_vectors(n=3), k_range=(50, 60))

    def test_group_label_is_carried_into_description(self):
        result = choose_clusters(separated_vectors(), k_range=(2,), group="ko")

        assert result.group == "ko"
        assert "[ko]" in result.describe()


class TestProfiles:
    def test_profile_counts_composition(self):
        rows = [
            row("d1-ko", "discovery", "ko", ("Slack", "Notion")),
            row("d1-ko", "discovery", "ko", ("Slack",)),
            row("c1-en", "constrained", "en", ("Swit",)),
        ]
        profiles = {p.cluster: p for p in profile_clusters(rows, [0, 0, 1])}

        assert profiles[0].size == 2
        assert profiles[0].stage_mix == {"discovery": 2}
        assert profiles[0].brand_share["Slack"] == pytest.approx(1.0)
        assert profiles[0].brand_share["Notion"] == pytest.approx(0.5)
        assert profiles[1].language_mix == {"en": 1}

    def test_mean_length_is_reported(self):
        rows = [
            row("d1-ko", "discovery", "ko", (), length=100),
            row("d1-ko", "discovery", "ko", (), length=300),
        ]
        profile = profile_clusters(rows, [0, 0])[0]

        assert profile.mean_length == pytest.approx(200.0)

    def test_length_mismatch_fails_loudly(self):
        """조용히 어긋나면 군집 해석이 통째로 틀린다."""
        with pytest.raises(ValueError):
            profile_clusters([row("d1-ko", "discovery", "ko", ())], [0, 1])


class TestBrandByCluster:
    def test_returns_interval_per_cluster(self):
        rows = [
            row("d1-ko", "discovery", "ko", ("Swit",)),
            row("d1-ko", "discovery", "ko", ()),
            row("c1-ko", "constrained", "ko", ("Swit",)),
        ]
        by_cluster = brand_by_cluster(rows, [0, 0, 1], "Swit")

        assert by_cluster[0].point == pytest.approx(0.5)
        assert by_cluster[0].n == 2
        assert by_cluster[1].point == pytest.approx(1.0)

    def test_zero_cluster_keeps_upper_bound(self):
        rows = [row("d1-ko", "discovery", "ko", ())]
        by_cluster = brand_by_cluster(rows, [0], "Swit")

        assert by_cluster[0].point == 0.0
        assert by_cluster[0].high > 0.0

    def test_length_mismatch_fails_loudly(self):
        with pytest.raises(ValueError):
            brand_by_cluster([row("d1-ko", "discovery", "ko", ())], [0, 1], "Swit")


class TestOverInterpretationGuards:
    """군집을 발견으로 착각하지 않게 막는 장치들."""

    def test_agreement_is_one_when_clusters_reproduce_a_label(self):
        labels = [0, 0, 1, 1]
        agreement = label_agreement(labels, {"언어": ["ko", "ko", "en", "en"]})

        assert agreement["언어"] == pytest.approx(1.0)

    def test_agreement_is_near_zero_for_unrelated_label(self):
        labels = [0, 0, 1, 1]
        agreement = label_agreement(labels, {"무관": ["a", "b", "a", "b"]})

        assert agreement["무관"] < 0.1

    def test_agreement_is_sorted_strongest_first(self):
        labels = [0, 0, 1, 1]
        agreement = label_agreement(
            labels,
            {"무관": ["a", "b", "a", "b"], "언어": ["ko", "ko", "en", "en"]},
        )

        assert list(agreement)[0] == "언어"

    def test_agreement_length_mismatch_fails_loudly(self):
        with pytest.raises(ValueError):
            label_agreement([0, 1], {"언어": ["ko"]})

    def test_monotone_silhouette_means_no_natural_k(self):
        """점수가 계속 오르기만 하면 최고점 k 는 그냥 시도한 최댓값이다."""
        rising = [ClusterChoice(k, 0.1 * k) for k in (2, 3, 4, 5)]

        assert has_natural_k(rising) is False

    def test_peak_in_the_middle_counts_as_natural(self):
        peaked = [
            ClusterChoice(2, 0.20),
            ClusterChoice(3, 0.45),
            ClusterChoice(4, 0.30),
        ]

        assert has_natural_k(peaked) is True

    def test_too_few_candidates_are_not_judged(self):
        assert has_natural_k([ClusterChoice(2, 0.3)]) is True

    def test_a_single_dip_does_not_count_as_a_peak(self):
        """실측에서 잡힌 회귀 사례 — 전체가 우상향인데 한 칸만 내려간다.

        국소 요철을 정점으로 읽으면 그림과 본문이 정반대가 된다.
        """
        rising_with_dip = [
            ClusterChoice(6, 0.255),
            ClusterChoice(7, 0.271),
            ClusterChoice(8, 0.303),
            ClusterChoice(9, 0.293),
            ClusterChoice(10, 0.322),
            ClusterChoice(11, 0.333),
            ClusterChoice(12, 0.353),
        ]

        assert has_natural_k(rising_with_dip) is False

    def test_peak_at_the_smallest_k_is_allowed(self):
        """k=2 가 최고면 그건 '가장 적게 나눈 것이 답'이라는 뜻이라 경계 문제가 아니다."""
        descending = [ClusterChoice(2, 0.5), ClusterChoice(3, 0.3), ClusterChoice(4, 0.2)]

        assert has_natural_k(descending) is True


class TestWeakThreshold:
    def test_threshold_is_explicit_and_documented(self):
        """경계값이 코드 안에 숨어 있으면 리포트가 그것을 설명할 수 없다."""
        assert ClusterResult.WEAK_SILHOUETTE == 0.15
