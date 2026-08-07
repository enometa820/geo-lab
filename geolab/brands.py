"""응답 텍스트에서 브랜드 언급을 찾는다.

설계 원칙 — **오탐 0이 커버리지보다 우선한다.**

측정값이 한 번이라도 헛짚으면 이후 숫자 전체가 못 믿을 것이 된다. 그래서 일반
명사와 겹치는 별칭은 아예 잡지 않고, 무엇을 안 잡았는지 기록해 리포트에 표시한다.
미탐은 "이 브랜드는 과소 집계될 수 있다"로 정직하게 보고할 수 있지만, 오탐은
그럴 수 없다 — 틀린 숫자가 맞는 것처럼 보이기 때문이다.

코드로 결정되는 것은 코드로 한다. 문맥 판단이 필요한 모호 별칭은 이 층에서
처리하지 않고 상위(LLM 판정)로 넘길 수 있게 남겨둔다.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

_HANGUL = re.compile(r"[가-힣]")


@dataclass(frozen=True)
class Brand:
    """측정 대상 브랜드.

    canonical           리포트에 쓰는 정식 이름
    aliases             안전하게 매칭 가능한 표기
    ambiguous_aliases   일반 명사와 겹쳐 **의도적으로 매칭하지 않는** 표기.
                        지우지 않고 남기는 이유는 리포트에 한계로 표시하기 위해서다
    """

    canonical: str
    aliases: tuple[str, ...]
    ambiguous_aliases: tuple[str, ...] = field(default=())

    def __post_init__(self) -> None:
        if not self.aliases:
            raise ValueError(f"{self.canonical}: aliases 가 비어 있다")


# 팀 협업 툴. 과제 페르소나("구독형 협업 툴을 서비스하는 B2B SaaS")와 같은 카테고리다.
#
# ambiguous_aliases 에 들어간 것들이 왜 위험한지:
#   잔디 / 미로 / 줌 / 플로우 — 한국어 일반 명사
#   Linear / Flow / Teams / Zoom — 영어 일반 명사
# 이 표기들은 문맥 없이는 브랜드인지 알 수 없다.
BRANDS: tuple[Brand, ...] = (
    Brand("Slack", ("Slack", "슬랙")),
    Brand("Notion", ("Notion", "노션")),
    Brand("Microsoft Teams", ("Microsoft Teams", "MS Teams", "팀즈", "마이크로소프트 팀즈"),
          ambiguous_aliases=("Teams",)),
    Brand("Asana", ("Asana", "아사나")),
    Brand("Trello", ("Trello", "트렐로")),
    Brand("Jira", ("Jira", "지라")),
    Brand("Confluence", ("Confluence", "컨플루언스")),
    Brand("Monday.com", ("Monday.com", "monday.com", "먼데이닷컴"),
          ambiguous_aliases=("Monday", "먼데이")),
    Brand("ClickUp", ("ClickUp", "Click Up", "클릭업")),
    Brand("Linear", ("Linear.app", "리니어"),
          ambiguous_aliases=("Linear",)),
    Brand("Basecamp", ("Basecamp", "베이스캠프")),
    Brand("Miro", ("Miro",),
          ambiguous_aliases=("미로",)),
    Brand("Figma", ("Figma", "피그마")),
    Brand("Zoom", ("Zoom",),
          ambiguous_aliases=("줌",)),
    Brand("Google Workspace", ("Google Workspace", "구글 워크스페이스", "구글워크스페이스")),
    Brand("Dooray", ("Dooray", "두레이")),
    Brand("JANDI", ("JANDI",),
          ambiguous_aliases=("잔디",)),
    Brand("Kakao Work", ("Kakao Work", "KakaoWork", "카카오워크")),
    Brand("Flow", ("플로우 협업", "Flow 협업"),
          ambiguous_aliases=("Flow", "플로우")),
    Brand("Swit", ("Swit", "스윗"),),
)


def _pattern_for(alias: str) -> re.Pattern[str]:
    """별칭 하나를 정규식으로.

    한글은 조사가 붙으므로("슬랙을", "노션은") 뒤쪽 경계를 요구할 수 없다.
    반면 ASCII 는 경계를 요구해야 한다 — 그러지 않으면 "Slacker" 가 Slack 으로
    잡힌다.
    """
    escaped = re.escape(alias)
    if _HANGUL.search(alias):
        return re.compile(escaped)
    return re.compile(rf"(?<![A-Za-z0-9]){escaped}(?![A-Za-z0-9])", re.IGNORECASE)


_COMPILED: dict[str, tuple[re.Pattern[str], ...]] = {
    brand.canonical: tuple(_pattern_for(a) for a in brand.aliases) for brand in BRANDS
}


def detect_mentions(text: str, brands: tuple[Brand, ...] = BRANDS) -> set[str]:
    """텍스트에 언급된 브랜드의 canonical 이름 집합.

    한 응답에서 같은 브랜드가 여러 번 나와도 1회로 센다. 우리가 재는 것은
    "이 응답에 등장했는가"라는 베르누이 시행이지 등장 횟수가 아니다.
    """
    if not text:
        return set()

    compiled = (
        _COMPILED
        if brands is BRANDS
        else {b.canonical: tuple(_pattern_for(a) for a in b.aliases) for b in brands}
    )

    return {
        canonical
        for canonical, patterns in compiled.items()
        if any(p.search(text) for p in patterns)
    }


def excluded_aliases(brands: tuple[Brand, ...] = BRANDS) -> dict[str, tuple[str, ...]]:
    """일부러 매칭하지 않는 표기 목록.

    리포트에 그대로 실어 "이 브랜드는 과소 집계될 수 있다"를 명시하기 위한 것이다.
    측정 한계를 숨기지 않는 것이 이 저장소의 규격이다.
    """
    return {b.canonical: b.ambiguous_aliases for b in brands if b.ambiguous_aliases}
