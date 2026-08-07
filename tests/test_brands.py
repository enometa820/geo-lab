"""브랜드 탐지 검증.

이 파일의 절반은 **잡히면 안 되는 것**을 검사한다. 오탐 하나가 측정 전체의
신뢰를 무너뜨리기 때문이다.
"""

import pytest

from geolab.brands import (
    BRANDS,
    Brand,
    detect_mentions,
    excluded_aliases,
    first_positions,
    match_spans,
    mention_ranks,
)


class TestBasicDetection:
    def test_english_name(self):
        assert detect_mentions("I recommend Slack for team chat.") == {"Slack"}

    def test_korean_name(self):
        assert detect_mentions("노션을 추천합니다.") == {"Notion"}

    def test_korean_particle_attached(self):
        """한국어는 조사가 붙는다. 뒤쪽 경계를 요구하면 다 놓친다."""
        for text in ["슬랙은", "슬랙이", "슬랙을", "슬랙과", "슬랙보다"]:
            assert detect_mentions(text) == {"Slack"}, f"{text} 에서 못 잡음"

    def test_case_insensitive(self):
        for text in ["SLACK", "slack", "SlAcK"]:
            assert detect_mentions(text) == {"Slack"}, f"{text} 에서 못 잡음"

    def test_multiple_brands(self):
        text = "슬랙, Notion, 그리고 Trello를 비교하면"

        assert detect_mentions(text) == {"Slack", "Notion", "Trello"}

    def test_repeated_mention_counts_once(self):
        """등장 횟수가 아니라 등장 여부를 잰다."""
        assert detect_mentions("Slack, Slack, 슬랙!") == {"Slack"}

    def test_multiword_alias(self):
        assert detect_mentions("We use Microsoft Teams daily.") == {"Microsoft Teams"}

    def test_empty_text(self):
        assert detect_mentions("") == set()


class TestFalsePositives:
    """여기가 이 모듈의 핵심이다. 하나라도 깨지면 측정값을 못 믿는다."""

    def test_word_boundary_blocks_substring_match(self):
        assert detect_mentions("Slacker") == set()
        assert detect_mentions("slacking off") == set()

    def test_alphanumeric_neighbors_block_match(self):
        assert detect_mentions("myslack") == set()
        assert detect_mentions("slack99") == set()

    def test_ambiguous_korean_nouns_are_not_matched(self):
        """일반 명사와 겹치는 별칭은 의도적으로 안 잡는다."""
        assert detect_mentions("잔디밭에 앉아서") == set()
        assert detect_mentions("미로 같은 구조라 헤맸다") == set()
        assert detect_mentions("화면을 줌 인 해보세요") == set()
        assert detect_mentions("업무 플로우를 정리하면") == set()

    def test_ambiguous_english_nouns_are_not_matched(self):
        assert detect_mentions("a linear regression model") == set()
        assert detect_mentions("the data flow is simple") == set()
        assert detect_mentions("small teams work better") == set()
        assert detect_mentions("on Monday we ship") == set()

    def test_unambiguous_form_is_still_matched(self):
        """모호한 별칭을 뺐다고 그 브랜드를 아예 못 잡는 것은 아니다."""
        assert detect_mentions("JANDI를 도입했다") == {"JANDI"}
        assert detect_mentions("Miro board") == {"Miro"}
        assert detect_mentions("Zoom 회의") == {"Zoom"}
        assert detect_mentions("Linear.app 을 씁니다") == {"Linear"}

    def test_hyphen_and_punctuation_do_not_block(self):
        assert detect_mentions("Slack-based workflow") == {"Slack"}
        assert detect_mentions("(Notion)") == {"Notion"}


class TestTransparency:
    def test_excluded_aliases_are_reported(self):
        """무엇을 안 잡았는지 밖에서 알 수 있어야 한다."""
        excluded = excluded_aliases()

        assert "잔디" in excluded["JANDI"]
        assert "미로" in excluded["Miro"]
        assert "Linear" in excluded["Linear"]

    def test_brands_without_ambiguity_are_absent(self):
        excluded = excluded_aliases()

        assert "Slack" not in excluded
        assert "Notion" not in excluded


class TestCustomBrands:
    def test_caller_can_supply_own_brand_list(self):
        """제품이 되려면 사용자가 자기 브랜드를 넣을 수 있어야 한다."""
        mine = (Brand("Acme", ("Acme", "애크미")),)

        assert detect_mentions("Acme 를 씁니다", brands=mine) == {"Acme"}
        assert detect_mentions("Slack 을 씁니다", brands=mine) == set()

    def test_brand_requires_at_least_one_alias(self):
        with pytest.raises(ValueError):
            Brand("Empty", ())


class TestPositionsAndRanks:
    """순위는 언급률보다 탐지 한계에 민감하다. 그 성질을 고정해 둔다."""

    def test_position_is_first_occurrence(self):
        text = "Trello 를 쓰다가 Slack 으로 옮겼고 다시 Trello 를 봤다"
        positions = first_positions(text)

        assert positions["Trello"] == 0
        assert positions["Slack"] == text.index("Slack")

    def test_unmentioned_brand_is_absent_not_zero(self):
        """0을 돌려주면 '맨 앞에 나왔다'와 구별되지 않는다."""
        positions = first_positions("Slack 만 언급한다")

        assert "Notion" not in positions

    def test_earliest_alias_wins(self):
        """같은 브랜드를 여러 표기로 부르면 가장 앞선 자리가 그 브랜드의 자리다."""
        text = "슬랙 이야기를 먼저 하고 나중에 Slack 을 또 쓴다"
        positions = first_positions(text)

        assert positions["Slack"] == 0

    def test_rank_follows_appearance_order(self):
        text = "먼저 노션, 그 다음 슬랙, 마지막으로 Trello"

        assert mention_ranks(text) == {"Notion": 1, "Slack": 2, "Trello": 3}

    def test_ranks_are_dense_and_start_at_one(self):
        ranks = mention_ranks("Slack 과 Notion 과 Jira")

        assert sorted(ranks.values()) == [1, 2, 3]

    def test_undetected_brand_shifts_ranks_forward(self):
        """일부러 안 잡는 표기가 앞에 있으면 뒤 브랜드의 순위가 당겨진다.

        이건 버그가 아니라 오탐 0 정책의 대가다. 리포트가 이 사실을 밝히지 않으면
        순위가 실제보다 좋아 보인다.
        """
        text = "플로우가 먼저 나오고 그 다음 Slack 이 나온다"

        assert mention_ranks(text) == {"Slack": 1}

    def test_empty_text(self):
        assert first_positions("") == {}
        assert mention_ranks("") == {}


class TestMatchSpans:
    """화면에 표시하는 구간과 실제로 센 것이 어긋나면 근거 제시가 근거를 무너뜨린다."""

    def test_spans_point_at_the_matched_text(self):
        text = "우리는 Slack 을 쓴다"
        spans = match_spans(text, "Slack")

        assert len(spans) == 1
        assert text[spans[0][0] : spans[0][1]] == "Slack"

    def test_every_occurrence_is_returned(self):
        spans = match_spans("Slack 과 슬랙 그리고 Slack", "Slack")

        assert len(spans) == 3

    def test_spans_are_sorted(self):
        spans = match_spans("슬랙 뒤에 Slack", "Slack")

        assert list(spans) == sorted(spans)

    def test_ambiguous_alias_is_not_highlighted(self):
        """세지 않은 표기를 표시하면 '왜 이건 안 셌나'는 질문이 생긴다."""
        assert match_spans("업무 플로우를 정리하면", "Flow") == ()

    def test_unmentioned_brand_returns_empty(self):
        assert match_spans("Slack 만 있다", "Notion") == ()

    def test_unknown_brand_returns_empty(self):
        assert match_spans("아무 텍스트", "존재하지않는브랜드") == ()

    def test_empty_text(self):
        assert match_spans("", "Slack") == ()

    def test_spans_agree_with_detection(self):
        """탐지가 잡은 브랜드는 반드시 구간도 있어야 한다. 둘이 갈라지면 안 된다."""
        text = "슬랙과 Notion 을 비교하면 Jira 도 후보다"
        for canonical in detect_mentions(text):
            assert match_spans(text, canonical), f"{canonical}: 탐지됐는데 구간이 없다"


class TestBrandTable:
    def test_canonical_names_are_unique(self):
        names = [b.canonical for b in BRANDS]

        assert len(names) == len(set(names))

    def test_no_alias_appears_in_both_safe_and_ambiguous(self):
        """같은 표기가 안전하면서 모호할 수는 없다."""
        for brand in BRANDS:
            overlap = set(brand.aliases) & set(brand.ambiguous_aliases)
            assert not overlap, f"{brand.canonical}: {overlap} 가 양쪽에 있다"
