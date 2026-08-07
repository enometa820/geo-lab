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


def first_positions(text: str, brands: tuple[Brand, ...] = BRANDS) -> dict[str, int]:
    """브랜드별 **첫 등장 문자 위치**. 언급되지 않은 브랜드는 키에 없다.

    `detect_mentions` 가 "나왔는가"만 답하는 데 비해 이쪽은 "어디에 나왔는가"를
    답한다. 둘을 나눠 둔 이유는 쓰임이 다르기 때문이다 — 언급률 집계에는 위치가
    필요 없고, 위치를 섞으면 베르누이 시행이라는 성격이 흐려진다.

    위치가 필요한 이유는 **답변에서 몇 번째로 불리느냐가 노출의 질을 가르기**
    때문이다. 20개 목록의 맨 끝에 붙는 것과 첫 문단에서 추천되는 것은 같은
    "언급 1회"지만 사용자에게 도달하는 정도가 다르다.

    한 브랜드의 별칭이 여러 개면 **가장 앞선 위치**를 쓴다. 어느 표기로 불렸든
    처음 등장한 자리가 그 브랜드의 자리다.
    """
    if not text:
        return {}

    compiled = (
        _COMPILED
        if brands is BRANDS
        else {b.canonical: tuple(_pattern_for(a) for a in b.aliases) for b in brands}
    )

    positions: dict[str, int] = {}
    for canonical, patterns in compiled.items():
        found = [m.start() for p in patterns if (m := p.search(text))]
        if found:
            positions[canonical] = min(found)
    return positions


def match_spans(text: str, canonical: str, brands: tuple[Brand, ...] = BRANDS) -> tuple[tuple[int, int], ...]:
    """한 브랜드가 실제로 매칭된 모든 구간 `(시작, 끝)`.

    리포트에서 원 응답의 근거를 표시할 때 쓴다. **탐지에 쓴 것과 같은 패턴에서
    구간을 뽑는 것이 핵심이다** — 화면에 표시하는 규칙을 따로 만들면 표시된 것과
    집계된 것이 어긋나고, 그 순간 근거 제시가 근거를 무너뜨린다.

    겹치는 구간은 합친다. 별칭이 여러 개라 같은 자리가 두 번 잡힐 수 있다
    (예: "Microsoft Teams" 와 "MS Teams" 는 안 겹치지만, 표기를 늘리다 보면 겹친다).
    """
    if not text:
        return ()

    compiled = (
        _COMPILED
        if brands is BRANDS
        else {b.canonical: tuple(_pattern_for(a) for a in b.aliases) for b in brands}
    )
    patterns = compiled.get(canonical)
    if not patterns:
        return ()

    found = sorted(
        (m.start(), m.end()) for p in patterns for m in p.finditer(text)
    )
    if not found:
        return ()

    merged = [found[0]]
    for start, end in found[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return tuple(merged)


def mention_ranks(text: str, brands: tuple[Brand, ...] = BRANDS) -> dict[str, int]:
    """첫 등장 순서로 매긴 순위. 가장 먼저 나온 브랜드가 1위다.

    위치(문자 인덱스)가 아니라 순위를 따로 두는 이유는 **응답 길이가 제각각**이기
    때문이다. 3000자 응답의 900번째 문자와 600자 응답의 900번째 문자는 같은
    위치지만 전혀 다른 자리다. 순위는 그 길이 차이에 영향받지 않는다.

    동점은 생기지 않는다 — 문자 위치가 같을 수 없기 때문이다.
    """
    positions = first_positions(text, brands)
    ordered = sorted(positions.items(), key=lambda kv: kv[1])
    return {canonical: rank for rank, (canonical, _) in enumerate(ordered, start=1)}


def excluded_aliases(brands: tuple[Brand, ...] = BRANDS) -> dict[str, tuple[str, ...]]:
    """일부러 매칭하지 않는 표기 목록.

    리포트에 그대로 실어 "이 브랜드는 과소 집계될 수 있다"를 명시하기 위한 것이다.
    측정 한계를 숨기지 않는 것이 이 저장소의 규격이다.
    """
    return {b.canonical: b.ambiguous_aliases for b in brands if b.ambiguous_aliases}
