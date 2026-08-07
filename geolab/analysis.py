"""저장된 측정을 다시 파고드는 층 — 언급률 다음의 질문들.

`measure.py` 는 **"얼마나 나왔나"**에 답한다. 이 모듈은 그 다음을 묻는다.

    어디쯤에서 불리나      답변 첫머리인가 목록 꼬리인가
    누구와 함께 불리나     AI 가 인식하는 경쟁 구도
    어느 자리에서 빠지나   구매 여정의 어느 단계에서 사라지는가

## 왜 별도 모듈인가

`measure.py` 는 호출과 집계를 하고 그 결과가 이 저장소의 1차 근거다. 이 모듈은
그 근거를 **재료로 쓰는 2차 분석**이다. 섞어 두면 "측정한 것"과 "해석한 것"의
경계가 흐려진다.

## 외부 의존성을 두지 않는다

`stats.py` 와 같은 이유다 — 리뷰어가 설치 없이 읽고 돌려볼 수 있어야 한다.
회귀와 군집화처럼 수치 라이브러리가 꼭 필요한 것은 `modeling.py` 로 분리했다.

## 여기서도 신뢰구간을 뗄 수 없다

"Swit 이 언급된 응답의 60% 에서 Notion 도 같이 언급됐다"는 문장은, 그 60% 가
10건 중 6건이면 거의 아무것도 말하지 않는다. 그래서 이 모듈이 내놓는 비율은
전부 `Interval` 로 나간다. 조건부 비율도 비율이고, 비율에는 구간이 붙는다.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Iterable, Sequence

from .brands import BRANDS, Brand, mention_ranks
from .engines import Answer
from .measure import Observation
from .prompts import PROMPTS, Prompt
from .stats import Interval, wilson_interval


# --------------------------------------------------------------------------
# 원 응답 읽기
# --------------------------------------------------------------------------

def load_answers(path: str | Path) -> tuple[Answer, ...]:
    """`raw.jsonl` 을 다시 읽는다.

    관측(`Observation`)은 탐지 결과만 들고 본문을 버리므로, 위치·순위처럼
    **텍스트가 있어야 답할 수 있는 질문**은 이쪽을 읽어야 한다.

    실패한 응답도 그대로 싣는다. 걸러내는 것은 호출부의 판단이다 — 여기서
    조용히 빼면 "몇 건이 실패였나"를 상위에서 셀 수 없게 된다.
    """
    rows = []
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(Answer(**json.loads(line)))
    return tuple(rows)


# --------------------------------------------------------------------------
# 등장 순위 — "어디쯤에서 불리나"
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class RankSummary:
    """한 브랜드가 답변 안에서 차지한 자리.

    **모든 값이 "언급된 응답"에만 조건부다.** 언급 자체가 드문 브랜드는 평균
    순위가 좋아도 그것이 좋은 노출을 뜻하지 않는다 — 어쩌다 불릴 때 앞에
    불린다는 뜻일 뿐이다. 그래서 `n_mentions` 를 항상 같이 읽어야 한다.

    n_responses   집계 대상이 된 성공 응답 수 (분모)
    n_mentions    그중 이 브랜드가 언급된 응답 수
    top3          언급된 응답 중 3위 이내로 등장한 비율. 분모는 n_mentions
    """

    brand: str
    n_responses: int
    n_mentions: int
    mean_rank: float | None
    median_rank: float | None
    best_rank: int | None
    top3: Interval

    @property
    def mention_share(self) -> float:
        if self.n_responses == 0:
            return 0.0
        return self.n_mentions / self.n_responses

    def describe(self) -> str:
        if self.n_mentions == 0:
            return f"{self.brand}: 언급 0/{self.n_responses}"
        return (
            f"{self.brand}: 언급 {self.n_mentions}/{self.n_responses}, "
            f"평균 {self.mean_rank:.1f}위, 3위 이내 {self.top3.format_pct()}"
        )


def rank_summaries(
    answers: Sequence[Answer],
    brands: tuple[Brand, ...] = BRANDS,
) -> tuple[RankSummary, ...]:
    """브랜드별 등장 순위 요약. 순수 함수.

    ## 순위는 무엇에 대한 순위인가

    **우리가 탐지한 브랜드들 사이의 순위다.** 응답에 등장한 모든 고유명사가
    아니다. 탐지 목록에 없는 도구가 먼저 나왔다면 실제 순위는 이보다 뒤다.

    같은 이유로, 일반 명사와 겹쳐 일부러 안 잡는 표기(`ambiguous_aliases`)가
    있는 브랜드는 순위 계산에서 통째로 빠진다. 그 브랜드가 먼저 등장했더라도
    다른 브랜드의 순위가 그만큼 앞당겨진다. **순위는 언급률보다 이 한계에 더
    민감하다** — 언급률은 그 브랜드 한 줄만 틀리지만 순위는 전체가 밀린다.
    """
    ok = [a for a in answers if a.ok]
    per_brand: dict[str, list[int]] = {b.canonical: [] for b in brands}

    for answer in ok:
        for canonical, rank in mention_ranks(answer.text, brands).items():
            per_brand[canonical].append(rank)

    summaries = []
    for brand in brands:
        ranks = per_brand[brand.canonical]
        n_m = len(ranks)
        top3_hits = sum(1 for r in ranks if r <= 3)
        summaries.append(
            RankSummary(
                brand=brand.canonical,
                n_responses=len(ok),
                n_mentions=n_m,
                mean_rank=(sum(ranks) / n_m) if n_m else None,
                median_rank=float(median(ranks)) if n_m else None,
                best_rank=min(ranks) if n_m else None,
                top3=wilson_interval(top3_hits, n_m),
            )
        )

    # 언급이 많은 순. 같으면 평균 순위가 앞선 순
    summaries.sort(key=lambda s: (-s.n_mentions, s.mean_rank if s.mean_rank else 999))
    return tuple(summaries)


# --------------------------------------------------------------------------
# 공동언급 — "누구와 함께 불리나"
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class CoMention:
    """브랜드 A 가 나온 응답에서 B 도 나왔는가.

    **방향이 있다.** P(B|A) 와 P(A|B) 는 다른 값이고, 다른 것을 뜻한다.
    작은 브랜드가 큰 브랜드와 같이 불리는 것은 흔하지만 그 반대는 드물다.

    lift        P(A,B) / (P(A)P(B)). 1보다 크면 우연보다 자주 함께 나온다는
                뜻이다. 둘 중 하나라도 등장이 0이면 정의되지 않아 None 이다
    conditional P(B|A). 분모가 "A 가 나온 응답 수"이므로 A 가 드물면 구간이
                넓어진다 — 그 사실이 값에 드러나야 해서 Interval 로 둔다
    """

    a: str
    b: str
    n_total: int
    n_a: int
    n_b: int
    n_both: int
    conditional: Interval
    lift: float | None


def cooccurrence(
    observations: Sequence[Observation],
    brands: tuple[Brand, ...] = BRANDS,
    engine: str | None = None,
) -> tuple[CoMention, ...]:
    """브랜드 쌍의 동시 등장. 순수 함수.

    ## 왜 엔진을 섞으면 안 되나

    `measure.aggregate` 가 엔진을 합치지 않는 것과 같은 이유다. 플랫폼마다 한
    응답에 브랜드를 몇 개씩 나열하는지가 다르므로, 섞으면 공동언급이 많은
    플랫폼의 비중이 그대로 값이 된다. 이 저장소의 측정이 아직 단일 엔진이라
    `engine=None` 이면 전부를 쓰지만, 엔진이 둘 이상이면 반드시 지정한다.

    실패한 관측은 제외한다 — 언급이 없는 것이 아니라 잴 수 없었던 것이다.
    """
    rows = [o for o in observations if o.ok]
    if engine is not None:
        rows = [o for o in rows if o.engine == engine]
    elif len({o.engine for o in rows}) > 1:
        raise ValueError(
            "관측에 엔진이 둘 이상 섞여 있다. engine= 으로 지정한다 — "
            "플랫폼마다 나열 개수가 달라 합치면 값이 구성비에 좌우된다"
        )

    total = len(rows)
    names = [b.canonical for b in brands]
    mention_sets = [set(o.mentions) for o in rows]
    counts = {n: sum(1 for s in mention_sets if n in s) for n in names}

    pairs = []
    for a in names:
        for b in names:
            if a == b:
                continue
            n_a, n_b = counts[a], counts[b]
            n_both = sum(1 for s in mention_sets if a in s and b in s)

            if total and n_a and n_b:
                lift = (n_both / total) / ((n_a / total) * (n_b / total))
            else:
                lift = None

            pairs.append(
                CoMention(
                    a=a,
                    b=b,
                    n_total=total,
                    n_a=n_a,
                    n_b=n_b,
                    n_both=n_both,
                    conditional=wilson_interval(n_both, n_a),
                    lift=lift,
                )
            )

    pairs.sort(key=lambda c: (-c.conditional.point, c.a, c.b))
    return tuple(pairs)


def cooccurrence_for(
    observations: Sequence[Observation],
    brand: str,
    brands: tuple[Brand, ...] = BRANDS,
    engine: str | None = None,
) -> tuple[CoMention, ...]:
    """한 브랜드 기준의 공동언급만. "우리가 나올 때 누가 같이 나오나"."""
    return tuple(c for c in cooccurrence(observations, brands, engine) if c.a == brand)


# --------------------------------------------------------------------------
# 단계별 이탈 — "어느 자리에서 사라지나"
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class StageCell:
    """구매 여정 한 칸의 언급률.

    `homogeneous=False` 인 이유 — 한 단계에 프롬프트가 여럿이라 이 값은
    프롬프트를 합친 것이다. 구간 폭이 실제 불확실성을 과소평가할 수 있다.
    """

    brand: str
    stage: str
    language: str
    interval: Interval


def stage_language_grid(
    observations: Sequence[Observation],
    brand: str,
    brands: tuple[Brand, ...] = BRANDS,
) -> tuple[StageCell, ...]:
    """한 브랜드의 (단계 x 언어) 격자.

    담당자가 가장 먼저 볼 그림이다 — **어느 칸이 비어 있는지**가 곧 할 일이다.
    "전체 언급률 3%"는 행동으로 옮길 수 없지만 "조건부 질문에서는 10%, 대안
    질문에서는 0%"는 옮길 수 있다.
    """
    rows = [o for o in observations if o.ok]
    cells = []

    stages = sorted({o.stage for o in rows})
    languages = sorted({o.language for o in rows})

    for stage in stages:
        for language in languages:
            subset = [o for o in rows if o.stage == stage and o.language == language]
            hits = sum(1 for o in subset if brand in o.mentions)
            cells.append(
                StageCell(
                    brand=brand,
                    stage=stage,
                    language=language,
                    interval=wilson_interval(hits, len(subset)),
                )
            )
    return tuple(cells)


# --------------------------------------------------------------------------
# 회귀·군집에 넘길 표
# --------------------------------------------------------------------------

def observation_rows(
    observations: Sequence[Observation],
    brand: str,
    prompts: Sequence[Prompt] = PROMPTS,
) -> tuple[dict[str, object], ...]:
    """관측을 모델링용 평면 표로. 성공 관측만 나간다.

    `modeling.py` 가 이걸 받아 설계행렬을 만든다. 여기서 표를 만들어 두는 이유는
    **어떤 행이 모델에 들어갔는지를 수치 라이브러리 없이 확인**할 수 있게 하기
    위해서다.

    `prompt_id` 를 반드시 실어 보낸다 — 반복 20회가 같은 프롬프트에서 나왔으므로
    회귀의 표준오차를 이 열로 군집 보정해야 한다.
    """
    index = {p.id: p for p in prompts}
    rows = []
    for o in observations:
        if not o.ok:
            continue
        prompt = index.get(o.prompt_id)
        rows.append(
            {
                "prompt_id": o.prompt_id,
                "stage": o.stage,
                "language": o.language,
                "pair_id": o.pair_id,
                "run_index": o.run_index,
                "engine": o.engine,
                "mentioned": int(brand in o.mentions),
                "n_mentions_in_response": len(o.mentions),
                "prompt_text": prompt.text if prompt else "",
            }
        )
    return tuple(rows)


def response_texts(
    answers: Sequence[Answer],
    prompts: Sequence[Prompt] = PROMPTS,
    brands: tuple[Brand, ...] = BRANDS,
) -> tuple[dict[str, object], ...]:
    """군집화에 넘길 응답 표. 성공 응답만.

    텍스트와 함께 그 응답의 메타(단계·언어·탐지된 브랜드)를 싣는다. 군집이
    나온 뒤 **그 군집이 무엇이었는지 설명**하려면 메타가 붙어 있어야 한다.
    군집 번호만으로는 아무 말도 할 수 없다.
    """
    index = {p.id: p for p in prompts}
    rows = []
    for a in answers:
        if not a.ok:
            continue
        prompt = index.get(a.prompt_id)
        ranks = mention_ranks(a.text, brands)
        rows.append(
            {
                "prompt_id": a.prompt_id,
                "run_index": a.run_index,
                "stage": prompt.stage if prompt else "",
                "language": prompt.language if prompt else "",
                "text": a.text,
                "length": len(a.text),
                "mentions": tuple(sorted(ranks)),
                "ranks": ranks,
            }
        )
    return tuple(rows)


def summarize_undetectable(brands: tuple[Brand, ...] = BRANDS) -> tuple[str, ...]:
    """순위·공동언급 해석에서 반드시 같이 읽어야 할 브랜드 목록.

    `ambiguous_aliases` 가 있는 브랜드는 **0% 가 "안 나왔다"를 뜻하지 않는다.**
    리포트가 이 목록을 빠뜨리면 미탐이 미언급으로 읽힌다.
    """
    return tuple(b.canonical for b in brands if b.ambiguous_aliases)
