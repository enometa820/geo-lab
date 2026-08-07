"""측정에 쓰는 프롬프트 세트.

과제 페르소나("구독형 협업 툴을 서비스하는 B2B SaaS 스타트업 담당자")와 같은
카테고리로 잡았다. 그래야 측정 결과가 그 담당자의 질문에 그대로 답이 된다.

## 왜 구매 여정 단계로 나누나

조사에 따르면 B2B 구매자는 12개 벤더 고려군을 영업 접촉 전에 3~5개로 압축하고,
승리 벤더의 대부분이 첫날 shortlist 에 있었다. 즉 **어느 단계에서 호명되는지가
결과를 가른다.** "우리 이름을 넣고 물었을 때 나온다"와 "카테고리만 물었을 때
나온다"는 완전히 다른 사건이고, 후자가 훨씬 어렵고 훨씬 중요하다.

## 왜 한국어와 영어를 짝으로 두나

같은 질문이라도 언어에 따라 답이 달라질 수 있다. 특히 국내 서비스는 한국어
질문에서만 등장할 가능성이 높다. 이것이 사실이면 **영어권 도구로 측정하는
한국 브랜드는 자기 실제 노출을 못 본다**는 뜻이 된다. 짝을 맞춰 두면 그 차이를
바로 잴 수 있다.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Stage = Literal["problem", "discovery", "constrained", "comparison", "alternative"]

STAGE_ORDER: tuple[Stage, ...] = (
    "problem",
    "discovery",
    "constrained",
    "comparison",
    "alternative",
)

STAGE_DESCRIPTION: dict[Stage, str] = {
    "problem": "문제만 말하고 카테고리도 특정하지 않는다. 가장 앞단이자 가장 어렵다",
    "discovery": "카테고리를 물어본다. 브랜드는 지정하지 않는다",
    "constrained": "규모·예산·업종 조건을 붙인다. 실제 구매자가 묻는 방식",
    "comparison": "특정 브랜드를 지정해 비교시킨다. 이름이 이미 알려진 뒤의 질문",
    "alternative": "경쟁사 대안을 묻는다. 남의 이름으로 우리가 불려 나오는 자리",
}


@dataclass(frozen=True)
class Prompt:
    id: str
    stage: Stage
    language: Literal["ko", "en"]
    text: str
    pair_id: str
    """한국어·영어 짝을 묶는 키. 언어 간 비교에 쓴다."""


PROMPTS: tuple[Prompt, ...] = (
    # --- problem: 카테고리조차 특정하지 않는다 ---
    Prompt("p1-ko", "problem", "ko", "팀 커뮤니케이션이 여기저기 흩어져서 일이 자꾸 새는데 어떻게 해결하면 좋을까?", "p1"),
    Prompt("p1-en", "problem", "en", "Our team communication is scattered across too many places and things keep falling through the cracks. How should we fix this?", "p1"),
    Prompt("p2-ko", "problem", "ko", "원격 근무 팀에서 업무 진행 상황이 안 보여서 답답한데 뭘 도입해야 할까?", "p2"),
    Prompt("p2-en", "problem", "en", "In our remote team nobody can see what anyone else is working on. What should we adopt?", "p2"),

    # --- discovery: 카테고리를 묻는다 ---
    Prompt("d1-ko", "discovery", "ko", "팀 협업 툴 추천해줘", "d1"),
    Prompt("d1-en", "discovery", "en", "Recommend some team collaboration tools", "d1"),
    Prompt("d2-ko", "discovery", "ko", "업무 관리랑 소통을 한 번에 할 수 있는 협업 소프트웨어에는 뭐가 있어?", "d2"),
    Prompt("d2-en", "discovery", "en", "What collaboration software handles both task management and team communication?", "d2"),

    # --- constrained: 조건을 붙인다 ---
    Prompt("c1-ko", "constrained", "ko", "10명 규모 스타트업에 적합한 협업 툴을 예산까지 고려해서 추천해줘", "c1"),
    Prompt("c1-en", "constrained", "en", "Recommend collaboration tools for a 10-person startup, taking budget into account", "c1"),
    Prompt("c2-ko", "constrained", "ko", "국내 기업이 쓰기 좋은 협업 툴은 뭐가 있어? 한국어 지원이랑 국내 결제가 되면 좋겠어", "c2"),
    Prompt("c2-en", "constrained", "en", "What collaboration tools work well for companies in South Korea, with Korean language support?", "c2"),

    # --- comparison: 이름을 지정한다 ---
    Prompt("m1-ko", "comparison", "ko", "슬랙이랑 마이크로소프트 팀즈 중에 어떤 걸 쓰는 게 나아?", "m1"),
    Prompt("m1-en", "comparison", "en", "Should we use Slack or Microsoft Teams?", "m1"),

    # --- alternative: 남의 이름으로 불려 나온다 ---
    Prompt("a1-ko", "alternative", "ko", "슬랙 말고 다른 협업 툴 대안이 있을까?", "a1"),
    Prompt("a1-en", "alternative", "en", "What are good alternatives to Slack?", "a1"),
    Prompt("a2-ko", "alternative", "ko", "노션 대신 쓸 만한 협업 도구 알려줘", "a2"),
    Prompt("a2-en", "alternative", "en", "What can we use instead of Notion?", "a2"),
)


def by_stage(stage: Stage) -> tuple[Prompt, ...]:
    return tuple(p for p in PROMPTS if p.stage == stage)


def by_language(language: str) -> tuple[Prompt, ...]:
    return tuple(p for p in PROMPTS if p.language == language)


def pairs() -> dict[str, tuple[Prompt, ...]]:
    """pair_id 로 묶은 한국어·영어 짝. 언어 간 차이를 잴 때 쓴다."""
    grouped: dict[str, list[Prompt]] = {}
    for p in PROMPTS:
        grouped.setdefault(p.pair_id, []).append(p)
    return {k: tuple(v) for k, v in grouped.items()}
