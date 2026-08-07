"""2차 분석 검증.

여기서 지키는 것은 두 가지다.

1. **조건부 비율의 분모가 맞는가.** P(B|A) 의 분모는 전체가 아니라 A 가 나온
   응답 수다. 이걸 틀리면 공동언급이 조용히 작아진다
2. **언급이 0인 브랜드가 조용히 사라지지 않는가.** 요약에서 빠지면 "언급 없음"이
   리포트에서 안 보이고, 안 보이는 것은 없는 것이 된다
"""

import pytest

from geolab.analysis import (
    cooccurrence,
    cooccurrence_for,
    observation_rows,
    rank_summaries,
    response_texts,
    stage_language_grid,
    summarize_undetectable,
)
from geolab.brands import Brand
from geolab.engines import Answer
from geolab.measure import Observation

MINI = (
    Brand("Slack", ("Slack", "슬랙")),
    Brand("Notion", ("Notion", "노션")),
    Brand("Swit", ("Swit", "스윗")),
)


def answer(prompt_id: str, run_index: int, text: str, ok: bool = True) -> Answer:
    return Answer(
        engine="openai",
        model="test-model",
        prompt_id=prompt_id,
        run_index=run_index,
        text=text,
        ok=ok,
    )


def observation(
    prompt_id: str,
    run_index: int,
    mentions: tuple[str, ...],
    stage: str = "discovery",
    language: str = "ko",
    ok: bool = True,
) -> Observation:
    return Observation(
        engine="openai",
        model="test-model",
        prompt_id=prompt_id,
        stage=stage,
        language=language,
        pair_id=prompt_id[:-3] or "x",
        run_index=run_index,
        mentions=mentions,
        ok=ok,
    )


class TestRankSummaries:
    def test_mean_rank_is_conditional_on_mention(self):
        """언급된 응답만 평균에 들어간다. 미언급을 큰 순위로 채우지 않는다."""
        answers = [
            answer("d1-ko", 0, "노션 다음에 슬랙"),   # Notion 1, Slack 2
            answer("d1-ko", 1, "슬랙만 있다"),        # Slack 1
        ]
        by_brand = {s.brand: s for s in rank_summaries(answers, MINI)}

        assert by_brand["Slack"].n_mentions == 2
        assert by_brand["Slack"].mean_rank == pytest.approx(1.5)
        assert by_brand["Notion"].n_mentions == 1
        assert by_brand["Notion"].mean_rank == pytest.approx(1.0)

    def test_never_mentioned_brand_is_kept_with_none(self):
        """0건도 행으로 남는다. 빠지면 리포트에서 안 보인다."""
        summaries = rank_summaries([answer("d1-ko", 0, "슬랙만")], MINI)
        by_brand = {s.brand: s for s in summaries}

        assert "Swit" in by_brand
        assert by_brand["Swit"].n_mentions == 0
        assert by_brand["Swit"].mean_rank is None
        assert by_brand["Swit"].best_rank is None

    def test_failed_answers_are_excluded_from_denominator(self):
        """실패는 미언급이 아니다. 분모에서 빠져야 한다."""
        answers = [
            answer("d1-ko", 0, "슬랙"),
            answer("d1-ko", 1, "", ok=False),
        ]
        by_brand = {s.brand: s for s in rank_summaries(answers, MINI)}

        assert by_brand["Slack"].n_responses == 1
        assert by_brand["Slack"].mention_share == pytest.approx(1.0)

    def test_top3_denominator_is_mentions_not_responses(self):
        answers = [
            answer("d1-ko", 0, "슬랙이 먼저"),
            answer("d1-ko", 1, "노션만 나온다"),
        ]
        by_brand = {s.brand: s for s in rank_summaries(answers, MINI)}

        assert by_brand["Slack"].top3.n == 1
        assert by_brand["Slack"].top3.point == pytest.approx(1.0)


class TestCooccurrence:
    def test_conditional_denominator_is_a_not_total(self):
        """P(B|A) 의 분모는 A 가 나온 응답 수다."""
        obs = [
            observation("d1-ko", 0, ("Swit", "Notion")),
            observation("d1-ko", 1, ("Notion",)),
            observation("d1-ko", 2, ("Notion",)),
            observation("d1-ko", 3, ("Notion",)),
        ]
        pairs = {(c.a, c.b): c for c in cooccurrence(obs, MINI)}

        swit_notion = pairs[("Swit", "Notion")]
        assert swit_notion.n_a == 1
        assert swit_notion.n_both == 1
        assert swit_notion.conditional.n == 1
        assert swit_notion.conditional.point == pytest.approx(1.0)

    def test_direction_matters(self):
        """P(B|A) 와 P(A|B) 는 다르다."""
        obs = [
            observation("d1-ko", 0, ("Swit", "Notion")),
            observation("d1-ko", 1, ("Notion",)),
        ]
        pairs = {(c.a, c.b): c for c in cooccurrence(obs, MINI)}

        assert pairs[("Swit", "Notion")].conditional.point == pytest.approx(1.0)
        assert pairs[("Notion", "Swit")].conditional.point == pytest.approx(0.5)

    def test_lift_is_none_when_a_brand_never_appears(self):
        obs = [observation("d1-ko", 0, ("Slack",))]
        pairs = {(c.a, c.b): c for c in cooccurrence(obs, MINI)}

        assert pairs[("Slack", "Swit")].lift is None

    def test_lift_above_one_means_more_than_chance(self):
        # Slack 과 Notion 이 항상 같이, Swit 은 따로 나온다
        obs = [
            observation("d1-ko", 0, ("Slack", "Notion")),
            observation("d1-ko", 1, ("Slack", "Notion")),
            observation("d1-ko", 2, ("Swit",)),
            observation("d1-ko", 3, ("Swit",)),
        ]
        pairs = {(c.a, c.b): c for c in cooccurrence(obs, MINI)}

        assert pairs[("Slack", "Notion")].lift > 1.0
        assert pairs[("Slack", "Swit")].lift == pytest.approx(0.0)

    def test_failed_observations_excluded(self):
        obs = [
            observation("d1-ko", 0, ("Slack", "Notion")),
            observation("d1-ko", 1, (), ok=False),
        ]
        pairs = {(c.a, c.b): c for c in cooccurrence(obs, MINI)}

        assert pairs[("Slack", "Notion")].n_total == 1

    def test_mixed_engines_raise_instead_of_silently_merging(self):
        """엔진을 섞은 공동언급은 구성비에 좌우된다. 조용히 합치지 않는다."""
        obs = [
            observation("d1-ko", 0, ("Slack",)),
            Observation(
                engine="anthropic", model="m", prompt_id="d1-ko", stage="discovery",
                language="ko", pair_id="d1", run_index=0, mentions=("Slack",), ok=True,
            ),
        ]
        with pytest.raises(ValueError):
            cooccurrence(obs, MINI)

    def test_engine_filter_allows_mixed_input(self):
        obs = [
            observation("d1-ko", 0, ("Slack",)),
            Observation(
                engine="anthropic", model="m", prompt_id="d1-ko", stage="discovery",
                language="ko", pair_id="d1", run_index=0, mentions=("Notion",), ok=True,
            ),
        ]
        pairs = cooccurrence(obs, MINI, engine="openai")

        assert all(c.n_total == 1 for c in pairs)

    def test_cooccurrence_for_filters_to_one_brand(self):
        obs = [observation("d1-ko", 0, ("Swit", "Notion"))]

        assert {c.a for c in cooccurrence_for(obs, "Swit", MINI)} == {"Swit"}


class TestStageGrid:
    def test_cells_cover_stage_by_language(self):
        obs = [
            observation("d1-ko", 0, ("Swit",), stage="discovery", language="ko"),
            observation("d1-en", 0, (), stage="discovery", language="en"),
            observation("c1-ko", 0, ("Swit",), stage="constrained", language="ko"),
            observation("c1-en", 0, (), stage="constrained", language="en"),
        ]
        cells = {(c.stage, c.language): c for c in stage_language_grid(obs, "Swit", MINI)}

        assert cells[("discovery", "ko")].interval.point == pytest.approx(1.0)
        assert cells[("discovery", "en")].interval.point == pytest.approx(0.0)
        assert len(cells) == 4

    def test_zero_cell_has_nonzero_upper_bound(self):
        """0/1 은 '확실히 0%'가 아니다. Wilson 이 그걸 지킨다."""
        obs = [observation("d1-ko", 0, (), stage="discovery", language="ko")]
        cell = stage_language_grid(obs, "Swit", MINI)[0]

        assert cell.interval.point == 0.0
        assert cell.interval.high > 0.0


class TestModelingTables:
    def test_observation_rows_carry_prompt_id_for_clustering(self):
        """군집 보정을 하려면 어느 프롬프트에서 나온 행인지가 있어야 한다."""
        obs = [observation("d1-ko", 0, ("Swit",))]
        rows = observation_rows(obs, "Swit", prompts=())

        assert rows[0]["prompt_id"] == "d1-ko"
        assert rows[0]["mentioned"] == 1

    def test_observation_rows_drop_failures(self):
        obs = [
            observation("d1-ko", 0, ("Swit",)),
            observation("d1-ko", 1, (), ok=False),
        ]

        assert len(observation_rows(obs, "Swit", prompts=())) == 1

    def test_response_texts_attach_ranks(self):
        answers = [answer("d1-ko", 0, "노션 다음 슬랙")]
        rows = response_texts(answers, prompts=(), brands=MINI)

        assert rows[0]["ranks"] == {"Notion": 1, "Slack": 2}
        assert rows[0]["length"] == len("노션 다음 슬랙")

    def test_response_texts_drop_failures(self):
        answers = [answer("d1-ko", 0, "", ok=False)]

        assert response_texts(answers, prompts=(), brands=MINI) == ()


class TestUndetectableDisclosure:
    def test_ambiguous_brands_are_listed(self):
        """이 목록이 빠지면 미탐이 미언급으로 읽힌다."""
        undetectable = summarize_undetectable()

        assert "Flow" in undetectable
        assert "Linear" in undetectable
        assert "Slack" not in undetectable
