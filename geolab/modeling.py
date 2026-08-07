"""언급 여부를 구매 단계와 언어로 설명하는 회귀.

`analysis.stage_language_grid` 가 칸마다 비율을 보여준다면, 이 모듈은 **"단계와
언어가 각각 얼마나 기여하는가"**를 하나의 모형으로 분리한다. 격자는 칸이 잘게
쪼개져 표본이 작아지지만, 회귀는 전체를 함께 써서 각 항의 효과를 추정한다.

## 이 모듈은 두 개의 함정 위에 서 있다

이 저장소의 규격이 "한계를 먼저 적는다"이므로, 코드가 그 한계를 **스스로 탐지해서
결과에 실어 보낸다.** 사람이 기억해서 각주를 다는 것에 맡기지 않는다.

### 함정 1 — 완전분리 (complete separation)

어떤 단계에서 브랜드가 **한 번도** 언급되지 않으면, 그 단계의 계수를 음의 무한대로
보낼수록 우도가 계속 커진다. 최대우도추정이 수렴하지 않고, 수렴한 것처럼 보여도
계수와 표준오차가 둘 다 터무니없이 커진다.

문제는 이게 **드문 사고가 아니라 우리 데이터의 기본값**이라는 것이다. 노출이 약한
브랜드일수록 빈 칸이 많고, 그런 브랜드가 바로 이 도구를 필요로 하는 브랜드다.

그래서 `fit_stage_language` 는 적합 전에 빈 칸을 먼저 찾고, 발견하면 그 사실을
`separation` 에 담아 내보낸다. **숫자를 지어내느니 "추정할 수 없다"고 말한다.**

### 함정 2 — 군집이 18개뿐이다

반복 20회는 같은 프롬프트에서 나왔으므로 360행은 독립이 아니다. 보정하지 않으면
표준오차가 실제보다 좁아지고, 그러면 **이 저장소가 비판하는 바로 그 잘못**을
저지르게 된다. 그래서 `prompt_id` 로 군집 보정한 robust standard error 를 쓴다.

다만 군집 보정 자체도 군집 수가 충분해야 잘 작동한다. 통상 30~50개를 권하는데
우리는 프롬프트가 18개다. **적은 군집에서 robust SE 는 여전히 하향 편향된다** —
보정을 했다고 문제가 사라진 것이 아니라 줄어든 것이다. 이 사실을
`cluster_caveat` 로 항상 함께 내보낸다.

## 의존성

`statsmodels` 와 `numpy` 를 쓴다. 측정 코어(`stats.py`·`brands.py`·`measure.py`)와
2차 분석(`analysis.py`)은 여전히 표준 라이브러리만 쓰므로, 이 모듈을 설치하지
않아도 측정과 집계는 그대로 돌아간다.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

# 군집 보정이 신뢰할 만해지는 대략의 군집 수. 문헌에서 흔히 인용되는 하한이다.
MIN_CLUSTERS_FOR_ROBUST_SE = 30


@dataclass(frozen=True)
class Term:
    """회귀 계수 하나.

    로그 오즈 척도의 계수(`coef`)와 오즈비(`odds_ratio`)를 함께 든다. 사람이 읽는
    것은 오즈비지만, 구간을 계산하는 것은 로그 척도에서 해야 대칭이 유지된다.
    """

    name: str
    coef: float
    std_err: float
    z: float
    p_value: float
    ci_low: float
    ci_high: float

    @property
    def odds_ratio(self) -> float:
        return math.exp(self.coef)

    @property
    def or_ci(self) -> tuple[float, float]:
        return math.exp(self.ci_low), math.exp(self.ci_high)

    def describe(self) -> str:
        low, high = self.or_ci
        return f"{self.name}: OR {self.odds_ratio:.2f} (95% CI {low:.2f}~{high:.2f})"


@dataclass(frozen=True)
class EmptyCell:
    """언급이 0이거나 전부인 수준. 완전분리의 원인이 되는 자리."""

    variable: str
    level: str
    n: int
    events: int

    def describe(self) -> str:
        kind = "0회" if self.events == 0 else "전부"
        return f"{self.variable}={self.level}: {self.n}건 중 {kind} 언급"


@dataclass(frozen=True)
class LogitFit:
    """적합 결과와 그 결과를 믿을 수 있는 정도.

    `estimable=False` 면 `terms` 를 읽지 않는다. 완전분리가 있는 상태의 계수는
    숫자로는 나오지만 뜻이 없다.
    """

    brand: str
    n_obs: int
    n_events: int
    n_clusters: int
    reference: dict[str, str]
    terms: tuple[Term, ...]
    converged: bool
    separation: tuple[EmptyCell, ...]
    estimable: bool
    cluster_caveat: str | None
    excluded_stages: tuple[str, ...] = ()
    note: str | None = None

    def describe(self) -> str:
        head = (
            f"{self.brand}: n={self.n_obs}, 언급 {self.n_events}건, "
            f"군집 {self.n_clusters}개"
        )
        if self.excluded_stages:
            head += f" (제외한 단계: {', '.join(self.excluded_stages)})"
        if not self.estimable:
            reasons = "; ".join(c.describe() for c in self.separation)
            return f"{head}\n  추정 불가 — 완전분리: {reasons}"
        lines = [head] + [f"  {t.describe()}" for t in self.terms]
        return "\n".join(lines)


def find_empty_cells(
    rows: Sequence[dict],
    variables: Sequence[str] = ("stage", "language"),
) -> tuple[EmptyCell, ...]:
    """완전분리를 일으킬 수준을 찾는다. 적합 전에 돌린다. 순수 함수.

    수렴 경고에 의존하지 않는 이유 — 경고는 최적화기 사정에 따라 나오기도 하고
    안 나오기도 하는데, **빈 칸은 데이터의 사실이라 항상 같은 답을 준다.**
    그리고 사람에게 설명할 때도 "수렴하지 않았다"보다 "대안 질문에서 20회 중
    0회였다"가 훨씬 쓸모 있다.
    """
    found = []
    for variable in variables:
        levels = sorted({str(r[variable]) for r in rows})
        for level in levels:
            subset = [r for r in rows if str(r[variable]) == level]
            events = sum(int(r["mentioned"]) for r in subset)
            if events == 0 or events == len(subset):
                found.append(
                    EmptyCell(variable=variable, level=level, n=len(subset), events=events)
                )
    return tuple(found)


def fit_stage_language(
    rows: Sequence[dict],
    brand: str,
    stage_reference: str = "discovery",
    language_reference: str = "en",
    exclude_stages: Sequence[str] = (),
) -> LogitFit:
    """언급 여부 ~ 구매 단계 + 언어.

    `rows` 는 `analysis.observation_rows` 의 출력이다.

    ## exclude_stages 를 왜 파라미터로 두나

    호출부가 행을 미리 걸러서 넘겨도 결과는 같다. 그런데 그러면 **무엇을 빼고
    적합했는지가 결과에 남지 않는다.** 리포트를 읽는 사람은 n 이 320인 이유를
    알 수 없고, 다음 사람은 같은 판단을 다시 해야 한다.

    실제로 뺄 만한 단계가 있다. `comparison` 은 질문이 브랜드를 지목하므로
    (예: "슬랙이랑 팀즈 중 뭐가 나아?") 지목된 브랜드는 거의 100%, 나머지는
    거의 0%가 된다. 이건 노출을 잰 것이 아니라 질문을 되읽은 것이고, 계수를
    구할 수 없게 만드는 완전분리의 주된 원인이기도 하다.

    **빼는 것이 기본값은 아니다.** 뺄지 말지는 무엇을 주장하려는지에 달렸고,
    그 판단은 호출부가 하되 흔적은 여기 남는다.

    ## 기준 수준을 왜 이렇게 잡았나

    * 단계 기준 = `discovery` — "협업 툴 추천해줘". 카테고리를 묻는 가장 표준적인
      질문이라 다른 단계를 여기에 견주는 것이 읽기 쉽다
    * 언어 기준 = `en` — 그러면 `language[ko]` 의 계수가 **"한국어로 물으면 얼마나
      더 불리는가"**가 되어 국내 브랜드 관점에서 바로 읽힌다

    기준을 바꿔도 모형의 적합도는 같다. 바뀌는 것은 무엇에 견주어 말하느냐다.

    ## 반환값을 어떻게 읽나

    완전분리가 있으면 `estimable=False` 로 돌아오고 계수는 비어 있다. 그때는
    회귀 대신 `analysis.stage_language_grid` 의 칸별 Wilson 구간을 본다 — 0/80 도
    "0% ~ 4.6%"라는 정직한 답을 준다.
    """
    if not rows:
        raise ValueError("적합할 행이 없다")

    dropped = tuple(sorted(exclude_stages))
    if dropped:
        rows = [r for r in rows if str(r["stage"]) not in dropped]
        if not rows:
            raise ValueError(f"제외 후 남은 행이 없다: exclude_stages={dropped}")

    separation = find_empty_cells(rows)
    n_obs = len(rows)
    n_events = sum(int(r["mentioned"]) for r in rows)
    clusters = sorted({str(r["prompt_id"]) for r in rows})
    caveat = (
        f"군집 {len(clusters)}개는 robust SE 권장 하한({MIN_CLUSTERS_FOR_ROBUST_SE}개)보다"
        " 적다. 보정 후에도 표준오차가 실제보다 좁을 수 있다"
        if len(clusters) < MIN_CLUSTERS_FOR_ROBUST_SE
        else None
    )

    base = dict(
        brand=brand,
        n_obs=n_obs,
        n_events=n_events,
        n_clusters=len(clusters),
        reference={"stage": stage_reference, "language": language_reference},
        cluster_caveat=caveat,
        excluded_stages=dropped,
    )

    if separation:
        return LogitFit(
            terms=(),
            converged=False,
            separation=separation,
            estimable=False,
            note="완전분리가 있어 최대우도추정이 성립하지 않는다. 칸별 Wilson 구간을 쓴다",
            **base,
        )

    import numpy as np
    import statsmodels.api as sm

    stage_levels = sorted({str(r["stage"]) for r in rows} - {stage_reference})
    lang_levels = sorted({str(r["language"]) for r in rows} - {language_reference})

    names = ["const"] + [f"stage[{s}]" for s in stage_levels] + [f"language[{l}]" for l in lang_levels]
    design = np.column_stack(
        [np.ones(n_obs)]
        + [np.array([1.0 if str(r["stage"]) == s else 0.0 for r in rows]) for s in stage_levels]
        + [np.array([1.0 if str(r["language"]) == l else 0.0 for r in rows]) for l in lang_levels]
    )
    outcome = np.array([float(r["mentioned"]) for r in rows])
    groups = np.array([clusters.index(str(r["prompt_id"])) for r in rows])

    model = sm.Logit(outcome, design)
    try:
        result = model.fit(
            disp=False,
            maxiter=200,
            cov_type="cluster",
            cov_kwds={"groups": groups, "use_correction": True},
        )
    except Exception as exc:
        return LogitFit(
            terms=(),
            converged=False,
            separation=separation,
            estimable=False,
            note=f"적합 실패: {type(exc).__name__}: {exc}",
            **base,
        )

    conf = result.conf_int()
    terms = tuple(
        Term(
            name=name,
            coef=float(result.params[i]),
            std_err=float(result.bse[i]),
            z=float(result.tvalues[i]),
            p_value=float(result.pvalues[i]),
            ci_low=float(conf[i][0]),
            ci_high=float(conf[i][1]),
        )
        for i, name in enumerate(names)
    )

    retvals = getattr(result, "mle_retvals", None) or {}
    converged = bool(retvals.get("converged", True))

    return LogitFit(
        terms=terms,
        converged=converged,
        separation=(),
        estimable=converged,
        note=None if converged else "수렴하지 않았다. 계수를 해석하지 않는다",
        **base,
    )


def fit_many(
    rows_by_brand: dict[str, Sequence[dict]],
    stage_reference: str = "discovery",
    language_reference: str = "en",
    exclude_stages: Sequence[str] = (),
) -> tuple[LogitFit, ...]:
    """여러 브랜드를 같은 설정으로 적합한다.

    **추정 불가로 나온 브랜드를 결과에서 빼지 않는다.** 빼면 "회귀를 돌릴 수
    있었던 브랜드"만 남아, 노출이 약한 브랜드가 조용히 사라진다. 그런 브랜드가
    바로 이 분석이 필요한 쪽이다.
    """
    fits = [
        fit_stage_language(rows, brand, stage_reference, language_reference, exclude_stages)
        for brand, rows in rows_by_brand.items()
        if rows
    ]
    fits.sort(key=lambda f: (not f.estimable, -f.n_events))
    return tuple(fits)
