"""브랜드 탐지 검증.

이 파일의 절반은 **잡히면 안 되는 것**을 검사한다. 오탐 하나가 측정 전체의
신뢰를 무너뜨리기 때문이다.
"""

import pytest

from geolab.brands import BRANDS, Brand, detect_mentions, excluded_aliases


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


class TestBrandTable:
    def test_canonical_names_are_unique(self):
        names = [b.canonical for b in BRANDS]

        assert len(names) == len(set(names))

    def test_no_alias_appears_in_both_safe_and_ambiguous(self):
        """같은 표기가 안전하면서 모호할 수는 없다."""
        for brand in BRANDS:
            overlap = set(brand.aliases) & set(brand.ambiguous_aliases)
            assert not overlap, f"{brand.canonical}: {overlap} 가 양쪽에 있다"
