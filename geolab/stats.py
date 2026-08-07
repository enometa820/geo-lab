"""비율 추정의 신뢰구간.

LLM 은 비결정적이라 같은 프롬프트도 매번 다르게 답한다. 그래서 "브랜드가 언급됐다"는
베르누이 시행이고, 우리가 재는 것은 그 성공 확률이다. 관측 1회는 측정이 아니다.

이 모듈은 표준 라이브러리만 쓴다. 리뷰어가 설치 없이 돌려볼 수 있어야 한다.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import NormalDist


@dataclass(frozen=True)
class Interval:
    """비율 추정 결과.

    point   관측된 비율 (successes / n)
    low     신뢰구간 하한
    high    신뢰구간 상한
    n       표본 수
    """

    point: float
    low: float
    high: float
    n: int

    @property
    def margin(self) -> float:
        """구간의 반폭. 점추정치에서 양쪽으로 얼마나 벌어져 있나."""
        return (self.high - self.low) / 2

    def format_pct(self, digits: int = 1) -> str:
        """`23.5% (95% CI 12.1~41.2%, n=20)` 형태로."""
        return (
            f"{self.point * 100:.{digits}f}% "
            f"(CI {self.low * 100:.{digits}f}~{self.high * 100:.{digits}f}%, n={self.n})"
        )


def _z(confidence: float) -> float:
    if not 0 < confidence < 1:
        raise ValueError(f"confidence 는 0과 1 사이여야 한다: {confidence}")
    return NormalDist().inv_cdf(1 - (1 - confidence) / 2)


def wilson_interval(successes: int, n: int, confidence: float = 0.95) -> Interval:
    """Wilson score interval.

    왜 Wald(정규근사)가 아닌가 — 널리 쓰이는 `p ± z*sqrt(p(1-p)/n)` 는 비율이
    0이나 1에 가까울 때 무너진다. 특히 **0/20 이면 구간 폭이 0이 되어 "확실히 0%"
    라고 거짓말한다.** 브랜드 언급률은 자주 0에 가깝기 때문에 이 실패가 예외가
    아니라 기본값이 된다.

    Wilson 은 0/20 에서도 상한을 제대로 낸다 — "20번 봐서 한 번도 안 나왔다"는
    "0%"가 아니라 "0%부터 약 16%까지 가능하다"이고, 그게 사실이다.

    표본이 없으면(n=0) 아무것도 모르는 상태이므로 [0, 1] 을 돌려준다.
    """
    if n < 0:
        raise ValueError(f"n 은 음수일 수 없다: {n}")
    if not 0 <= successes <= n:
        raise ValueError(f"successes 는 0..n 범위여야 한다: {successes}/{n}")
    if n == 0:
        return Interval(point=0.0, low=0.0, high=1.0, n=0)

    z = _z(confidence)
    p = successes / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))

    return Interval(
        point=p,
        low=max(0.0, center - half),
        high=min(1.0, center + half),
        n=n,
    )


def runs_for_margin(margin: float, p_expected: float = 0.5, confidence: float = 0.95) -> int:
    """목표 오차범위를 얻으려면 몇 번 돌려야 하나.

    "몇 번 반복해야 하나요"에 답하기 위한 것이다. 기본값 p=0.5 는 가장 보수적인
    가정이다 — 분산이 최대라 필요 표본이 가장 크게 나온다. 실제 언급률을 대략
    안다면 넣어주면 표본이 줄어든다.

    Wald 기반 근사식이라 실제 Wilson 구간과 정확히 일치하지는 않는다.
    **계획용 어림값이지 결과 보고용이 아니다.**
    """
    if not 0 < margin < 1:
        raise ValueError(f"margin 은 0과 1 사이여야 한다: {margin}")
    if not 0 <= p_expected <= 1:
        raise ValueError(f"p_expected 는 0과 1 사이여야 한다: {p_expected}")

    z = _z(confidence)
    return math.ceil(z * z * p_expected * (1 - p_expected) / (margin * margin))


def overlaps(a: Interval, b: Interval) -> bool:
    """두 구간이 겹치는가.

    겹치면 "이 브랜드가 저 브랜드보다 많이 언급된다"고 말할 수 없다.
    점추정치만 비교해 순위를 매기는 것이 이 분야의 흔한 실수다.

    주의 — 구간이 겹치지 않으면 차이가 유의하지만, **겹친다고 해서 차이가 없다는
    뜻은 아니다.** 겹침 판정은 보수적인 검사이지 가설검정이 아니다.
    """
    return a.low <= b.high and b.low <= a.high
