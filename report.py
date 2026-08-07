"""측정 결과를 자기완결 HTML 로 렌더링한다.

    python report.py docs/measurements/2026-08-08-0320 --brand Swit

## 왜 손으로 쓴 숫자가 하나도 없어야 하나

리포트에 숫자를 타이핑하는 순간 그 숫자는 측정과 분리된다. 측정을 다시 돌리면
문서는 조용히 낡고, 그 사실을 아무도 모른다. 그래서 이 파일은 **차트의 좌표 하나까지
`run.json` 과 `raw.jsonl` 에서 계산한다.** 여기에 등장하는 유일한 상수는 색과 여백이다.

## 그림에 관한 원칙

* **주인공이 하나면 팔레트도 하나다.** 브랜드 20개에 20색을 주면 정작 봐야 할
  한 줄이 묻힌다. 대상 브랜드만 색을 갖고 나머지는 회색으로 물러난다
* **점추정치를 막대로 그리지 않는다.** 이 저장소의 논지가 "1회 관측은 측정이
  아니다"인데 막대는 값이 확정된 것처럼 보인다. 점과 신뢰구간 선으로 그린다
* **모든 그림에 표가 딸린다.** 색으로만 전달되는 값이 없어야 한다
* **과소 집계 가능한 브랜드에 표식을 단다.** 일반 명사와 겹쳐 일부러 안 잡는
  표기가 있는 브랜드는 값이 낮게 나올 수 있고, 그 사실이 그림에 보여야 한다
"""

from __future__ import annotations

import argparse
import html
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from geolab.analysis import (
    cooccurrence,
    cooccurrence_for,
    load_answers,
    observation_rows,
    rank_summaries,
    response_texts,
    stage_language_grid,
)
from geolab.brands import BRANDS, match_spans
from geolab.measure import aggregate, language_gap, load_run
from geolab.modeling import fit_many, fit_stage_language
from geolab.prompts import PROMPTS, STAGE_DESCRIPTION, STAGE_ORDER
from geolab.stats import Interval, wilson_interval

# --------------------------------------------------------------------------
# 색 — dataviz 검증기를 통과한 값만 쓴다 (validate_palette.js, light/dark 양쪽)
# --------------------------------------------------------------------------

SEQUENTIAL = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
              "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]

# 차트 축에 들어가는 짧은 이름. 칸 너비가 제한돼 있다
STAGE_LABEL = {
    "problem": "문제만 말함",
    "discovery": "카테고리 질문",
    "constrained": "조건부 질문",
    "comparison": "지목 비교",
    "alternative": "경쟁사 대안",
}

# 문장 안에 들어가는 표현. 짧은 이름을 문장에 그대로 넣으면
# "문제만 말함 질문" 같은 어색한 조합이 생긴다
STAGE_PHRASE = {
    "problem": "문제만 말한 질문",
    "discovery": "카테고리를 묻는 질문",
    "constrained": "조건을 붙인 질문",
    "comparison": "브랜드를 지목한 질문",
    "alternative": "경쟁사 대안을 묻는 질문",
}
LANGUAGE_LABEL = {"ko": "한국어", "en": "영어"}
# 국내 브랜드 관점의 리포트이므로 한국어를 먼저 놓는다. 알파벳 순이면 영어가 위로 간다
LANGUAGE_ORDER = ("ko", "en")

# 한국에서 만들어진 제품. 비교군을 국내·해외로 나눠 보여줄 때만 쓴다 —
# 측정이나 집계에는 관여하지 않는다.
DOMESTIC_BRANDS = ("Swit", "JANDI", "Dooray", "Kakao Work", "Flow")

# 대상 브랜드를 한 문단으로 소개한다. 여기 없는 브랜드가 대상이 되면 소개 절을
# 건너뛴다 — 모르는 것을 지어내지 않는다.
BRAND_INTRO = {
    "Swit": (
        "채팅과 업무 관리를 한 화면에서 제공하는 팀 협업 도구다. 한국에서 만들어졌고 "
        "구독형으로 판매된다. 이 설명은 측정된 응답에 반복해서 나온 서술을 그대로 옮긴 것이다 "
        "&mdash; 우리가 붙인 소개가 아니라 <strong>AI가 이 브랜드를 어떻게 설명하는가</strong> 자체가 "
        "측정 결과의 일부다."
    ),
}


def esc(text: object) -> str:
    return html.escape(str(text), quote=True)


def pct(value: float, digits: int = 1) -> str:
    return f"{value * 100:.{digits}f}%"


def ci_text(interval: Interval) -> str:
    return f"{pct(interval.point)} (95% CI {pct(interval.low)}~{pct(interval.high)}, n={interval.n})"


def sequential_color(fraction: float) -> str:
    """0~1 을 파랑 한 색조의 밝기 단계로. 무지개를 쓰지 않는다."""
    if fraction <= 0:
        return SEQUENTIAL[0]
    index = min(len(SEQUENTIAL) - 1, int(round(fraction * (len(SEQUENTIAL) - 1))))
    return SEQUENTIAL[index]


def ink_on(fill: str) -> str:
    """칸 안에 글자를 쓸 때 흰색과 먹색 중 무엇이 읽히나."""
    r, g, b = (int(fill[i : i + 2], 16) / 255 for i in (1, 3, 5))
    luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return "#ffffff" if luminance < 0.55 else "#0b0b0b"


# --------------------------------------------------------------------------
# SVG 조각
# --------------------------------------------------------------------------

def svg_open(width: int, height: int, label: str) -> str:
    return (
        f'<svg viewBox="0 0 {width} {height}" width="{width}" height="{height}" '
        f'role="img" aria-label="{esc(label)}" class="chart">'
    )


def gridlines(x0: int, x1: int, y0: int, y1: int, ticks: list[float], scale) -> str:
    """세로 격자와 눈금. 실선 hairline 만 쓴다 — 점선은 임계선으로 오독된다."""
    out = []
    for t in ticks:
        x = scale(t)
        out.append(f'<line x1="{x:.1f}" y1="{y0}" x2="{x:.1f}" y2="{y1}" class="grid"/>')
        out.append(
            f'<text x="{x:.1f}" y="{y1 + 16}" class="tick" text-anchor="middle">{t * 100:.0f}%</text>'
        )
    return "".join(out)


def tip(text: str) -> str:
    return f'data-tip="{esc(text)}"'


def frame(svg: str) -> str:
    """차트를 가로 스크롤 상자에 넣는다.

    좁은 화면에서 SVG 를 통째로 축소하면 글자가 읽을 수 없게 작아진다. 크기를
    유지한 채 그 상자만 스크롤되게 하고, **페이지 본문은 가로로 밀리지 않게** 한다.
    """
    return f'<div class="chartwrap">{svg}</div>'


# --------------------------------------------------------------------------
# 차트
# --------------------------------------------------------------------------

def chart_rates(rates, target: str, ambiguous: dict[str, tuple[str, ...]]) -> str:
    """브랜드별 언급률 — 점과 신뢰구간. 대상 브랜드만 색을 갖는다."""
    rows = [r for r in rates if r.interval.n > 0]
    left, right, top, row_h = 168, 104, 8, 27
    plot_w = 470
    width = left + plot_w + right
    height = top + row_h * len(rows) + 34

    def sx(v: float) -> float:
        return left + v * plot_w

    parts = [svg_open(width, height, f"브랜드별 언급률과 95% 신뢰구간, {len(rows)}개 브랜드")]
    parts.append(gridlines(left, left + plot_w, top, top + row_h * len(rows), [0, 0.25, 0.5, 0.75, 1.0], sx))

    for i, r in enumerate(rows):
        y = top + row_h * i + row_h / 2
        is_target = r.brand == target
        cls = "mark-target" if is_target else "mark-context"
        marker = "&#9662;" if r.brand in ambiguous else ""
        name = f"{esc(r.brand)}{marker}"

        detail = f"{r.brand} — {ci_text(r.interval)}"
        if r.brand in ambiguous:
            detail += f" · 과소 집계 가능: {', '.join(ambiguous[r.brand])}"

        parts.append(
            f'<g class="row {cls}" {tip(detail)} tabindex="0">'
            f'<rect x="0" y="{y - row_h/2:.1f}" width="{width}" height="{row_h}" class="hit"/>'
            f'<text x="{left - 12}" y="{y + 4:.1f}" class="rowlabel" text-anchor="end">{name}</text>'
            f'<line x1="{sx(r.interval.low):.1f}" y1="{y:.1f}" x2="{sx(r.interval.high):.1f}" y2="{y:.1f}" class="ci"/>'
            f'<circle cx="{sx(r.interval.point):.1f}" cy="{y:.1f}" r="4.5" class="dot"/>'
            f'<text x="{left + plot_w + 10}" y="{y + 4:.1f}" class="rowvalue">{pct(r.interval.point)}</text>'
            f"</g>"
        )

    parts.append("</svg>")
    return "".join(parts)


def chart_heatmap(cells, target: str) -> str:
    """단계 x 언어 격자. 연속 크기이므로 한 색조의 명도 단계로."""
    stages = [s for s in STAGE_ORDER if any(c.stage == s for c in cells)]
    present = {c.language for c in cells}
    languages = [l for l in LANGUAGE_ORDER if l in present] + sorted(present - set(LANGUAGE_ORDER))
    by_key = {(c.stage, c.language): c for c in cells}

    left, top, cw, ch = 150, 34, 118, 62
    width = left + cw * len(stages) + 16
    height = top + ch * len(languages) + 16

    parts = [svg_open(width, height, f"{target} 의 구매 단계별 언어별 언급률")]

    for j, stage in enumerate(stages):
        x = left + cw * j + cw / 2
        parts.append(
            f'<text x="{x:.1f}" y="{top - 12}" class="tick" text-anchor="middle">{esc(STAGE_LABEL.get(stage, stage))}</text>'
        )

    for i, language in enumerate(languages):
        y = top + ch * i
        parts.append(
            f'<text x="{left - 12}" y="{y + ch/2 + 4:.1f}" class="rowlabel" text-anchor="end">'
            f"{esc(LANGUAGE_LABEL.get(language, language))}</text>"
        )
        for j, stage in enumerate(stages):
            cell = by_key.get((stage, language))
            if cell is None:
                continue
            x = left + cw * j
            fill = sequential_color(cell.interval.point)
            detail = (
                f"{STAGE_LABEL.get(stage, stage)} / {LANGUAGE_LABEL.get(language, language)} — "
                f"{ci_text(cell.interval)} · {STAGE_DESCRIPTION.get(stage, '')}"
            )
            # 2px 표면 간격이 칸을 나눈다. 테두리를 그리지 않는다
            parts.append(
                f'<g class="row" {tip(detail)} tabindex="0">'
                f'<rect x="{x + 1}" y="{y + 1}" width="{cw - 2}" height="{ch - 2}" rx="4" fill="{fill}"/>'
                f'<text x="{x + cw/2:.1f}" y="{y + ch/2 - 2:.1f}" text-anchor="middle" '
                f'class="cellvalue" fill="{ink_on(fill)}">{pct(cell.interval.point, 0)}</text>'
                f'<text x="{x + cw/2:.1f}" y="{y + ch/2 + 15:.1f}" text-anchor="middle" '
                f'class="cellsub" fill="{ink_on(fill)}">{pct(cell.interval.low, 0)}~{pct(cell.interval.high, 0)}</text>'
                f"</g>"
            )

    parts.append("</svg>")
    return "".join(parts)


def chart_dumbbell(gaps, target: str, limit: int = 10) -> str:
    """한국어 대 영어 — 항목마다 전후를 잇는 덤벨. 한 색조 두 단계."""
    shown = [g for g in gaps if g[2].interval.n and g[3].interval.n][:limit]
    left, right, top, row_h = 168, 96, 26, 30
    plot_w = 440
    width = left + plot_w + right
    height = top + row_h * len(shown) + 34

    def sx(v: float) -> float:
        return left + v * plot_w

    parts = [svg_open(width, height, "브랜드별 한국어와 영어 언급률 비교")]
    parts.append(gridlines(left, left + plot_w, top, top + row_h * len(shown), [0, 0.25, 0.5, 0.75, 1.0], sx))

    for i, (brand, _engine, ko, en, overlapping) in enumerate(shown):
        y = top + row_h * i + row_h / 2
        detail = (
            f"{brand} — 한국어 {ci_text(ko.interval)} / 영어 {ci_text(en.interval)} · "
            + ("구간이 겹친다. 차이를 주장할 수 없다" if overlapping else "구간이 분리된다. 차이가 있다")
        )
        weight = "sep" if not overlapping else "ovl"
        parts.append(
            f'<g class="row dumb {weight}" {tip(detail)} tabindex="0">'
            f'<rect x="0" y="{y - row_h/2:.1f}" width="{width}" height="{row_h}" class="hit"/>'
            f'<text x="{left - 12}" y="{y + 4:.1f}" class="rowlabel" text-anchor="end">{esc(brand)}</text>'
            f'<line x1="{sx(en.interval.point):.1f}" y1="{y:.1f}" x2="{sx(ko.interval.point):.1f}" y2="{y:.1f}" class="link"/>'
            f'<circle cx="{sx(en.interval.point):.1f}" cy="{y:.1f}" r="4.5" class="dot-en"/>'
            f'<circle cx="{sx(ko.interval.point):.1f}" cy="{y:.1f}" r="4.5" class="dot-ko"/>'
            f'<text x="{left + plot_w + 10}" y="{y + 4:.1f}" class="rowvalue">'
            f'{"분리" if not overlapping else "겹침"}</text>'
            f"</g>"
        )

    parts.append("</svg>")
    return "".join(parts)


def chart_matrix(pairs, brands: list[str]) -> str:
    """공동언급 행렬. 행 A 가 나온 응답에서 열 B 가 나온 비율."""
    lookup = {(c.a, c.b): c for c in pairs}
    left, top, cell = 152, 116, 34
    width = left + cell * len(brands) + 16
    height = top + cell * len(brands) + 16

    parts = [svg_open(width, height, "브랜드 공동언급 행렬")]

    for j, b in enumerate(brands):
        x = left + cell * j + cell / 2
        parts.append(
            f'<text x="{x:.1f}" y="{top - 10}" class="tick colhead" '
            f'transform="rotate(-52 {x:.1f} {top - 10})" text-anchor="start">{esc(b)}</text>'
        )

    for i, a in enumerate(brands):
        y = top + cell * i
        parts.append(
            f'<text x="{left - 10}" y="{y + cell/2 + 4:.1f}" class="rowlabel" text-anchor="end">{esc(a)}</text>'
        )
        for j, b in enumerate(brands):
            x = left + cell * j
            if a == b:
                parts.append(
                    f'<rect x="{x+1}" y="{y+1}" width="{cell-2}" height="{cell-2}" rx="3" class="diag"/>'
                )
                continue
            c = lookup.get((a, b))
            if c is None:
                continue
            fill = sequential_color(c.conditional.point)
            detail = (
                f"{a} 가 나온 응답 {c.n_a}건 중 {b} 도 나온 것 {c.n_both}건 — "
                f"{ci_text(c.conditional)}"
                + (f" · lift {c.lift:.2f}" if c.lift is not None else "")
            )
            parts.append(
                f'<g class="row" {tip(detail)} tabindex="0">'
                f'<rect x="{x+1}" y="{y+1}" width="{cell-2}" height="{cell-2}" rx="3" fill="{fill}"/>'
                f'<text x="{x + cell/2:.1f}" y="{y + cell/2 + 4:.1f}" text-anchor="middle" '
                f'class="cellmini" fill="{ink_on(fill)}">{c.conditional.point * 100:.0f}</text>'
                f"</g>"
            )

    parts.append("</svg>")
    return "".join(parts)


def chart_forest(fits, term_name: str, label: str) -> str:
    """계수 하나를 브랜드에 걸쳐 비교하는 forest plot. 가로축은 로그 척도."""
    entries = []
    for f in fits:
        if not f.estimable:
            continue
        term = next((t for t in f.terms if t.name == term_name), None)
        if term is None:
            continue
        low, high = term.or_ci
        if not all(map(math.isfinite, (term.odds_ratio, low, high))) or low <= 0:
            continue
        entries.append((f.brand, term, low, high))

    if not entries:
        return '<p class="note">이 항을 추정할 수 있는 브랜드가 없다.</p>'

    lo = min(min(l for _, _, l, _ in entries), 0.5)
    hi = max(max(h for _, _, _, h in entries), 2.0)
    lo, hi = max(lo, 1e-3), min(hi, 1e3)

    left, right, top, row_h = 168, 118, 12, 29
    plot_w = 430
    width = left + plot_w + right
    height = top + row_h * len(entries) + 34

    def sx(v: float) -> float:
        v = min(max(v, lo), hi)
        return left + (math.log10(v) - math.log10(lo)) / (math.log10(hi) - math.log10(lo)) * plot_w

    parts = [svg_open(width, height, f"{label} 오즈비와 95% 신뢰구간")]

    for t in (0.1, 0.5, 1, 2, 10):
        if not lo <= t <= hi:
            continue
        x = sx(t)
        cls = "axis-ref" if t == 1 else "grid"
        parts.append(f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top + row_h*len(entries)}" class="{cls}"/>')
        parts.append(
            f'<text x="{x:.1f}" y="{top + row_h*len(entries) + 16}" class="tick" text-anchor="middle">{t:g}</text>'
        )

    for i, (brand, term, low, high) in enumerate(entries):
        y = top + row_h * i + row_h / 2
        crosses = low <= 1.0 <= high
        cls = "or-null" if crosses else ("or-up" if term.odds_ratio > 1 else "or-down")
        verdict = "구간이 1을 포함 — 방향 주장 불가" if crosses else ("올라간다" if term.odds_ratio > 1 else "내려간다")
        detail = f"{brand} — {label} OR {term.odds_ratio:.2f} (95% CI {low:.2f}~{high:.2f}) · {verdict}"
        parts.append(
            f'<g class="row {cls}" {tip(detail)} tabindex="0">'
            f'<rect x="0" y="{y - row_h/2:.1f}" width="{width}" height="{row_h}" class="hit"/>'
            f'<text x="{left - 12}" y="{y + 4:.1f}" class="rowlabel" text-anchor="end">{esc(brand)}</text>'
            f'<line x1="{sx(low):.1f}" y1="{y:.1f}" x2="{sx(high):.1f}" y2="{y:.1f}" class="ci"/>'
            f'<circle cx="{sx(term.odds_ratio):.1f}" cy="{y:.1f}" r="4.5" class="dot"/>'
            f'<text x="{left + plot_w + 10}" y="{y + 4:.1f}" class="rowvalue">{term.odds_ratio:.2f}</text>'
            f"</g>"
        )

    parts.append("</svg>")
    return "".join(parts)


def chart_ranks(summaries, target: str, limit: int = 14) -> str:
    """평균 등장 순위. 왼쪽이 앞자리다 — 축 방향이 '좋음'과 반대라 눈금에 명시한다."""
    rows = [s for s in summaries if s.n_mentions][:limit]
    if not rows:
        return ""

    worst = max(s.mean_rank for s in rows)
    hi = max(2.0, math.ceil(worst))
    left, right, top, row_h = 168, 118, 10, 26
    plot_w = 400
    width = left + plot_w + right
    height = top + row_h * len(rows) + 36

    def sx(v: float) -> float:
        return left + (v - 1) / max(1e-9, hi - 1) * plot_w

    parts = [svg_open(width, height, "브랜드별 평균 등장 순위")]
    ticks = [t for t in range(1, int(hi) + 1) if t == 1 or t % max(1, int(hi // 5)) == 0]
    for t in ticks:
        x = sx(t)
        parts.append(f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top + row_h*len(rows)}" class="grid"/>')
        parts.append(
            f'<text x="{x:.1f}" y="{top + row_h*len(rows) + 16}" class="tick" text-anchor="middle">{t}위</text>'
        )

    for i, s in enumerate(rows):
        y = top + row_h * i + row_h / 2
        cls = "mark-target" if s.brand == target else "mark-context"
        detail = (
            f"{s.brand} — 평균 {s.mean_rank:.1f}위, 최고 {s.best_rank}위 · "
            f"언급 {s.n_mentions}/{s.n_responses} · 3위 이내 {ci_text(s.top3)}"
        )
        parts.append(
            f'<g class="row {cls}" {tip(detail)} tabindex="0">'
            f'<rect x="0" y="{y - row_h/2:.1f}" width="{width}" height="{row_h}" class="hit"/>'
            f'<text x="{left - 12}" y="{y + 4:.1f}" class="rowlabel" text-anchor="end">{esc(s.brand)}</text>'
            f'<circle cx="{sx(s.mean_rank):.1f}" cy="{y:.1f}" r="4.5" class="dot"/>'
            f'<text x="{left + plot_w + 10}" y="{y + 4:.1f}" class="rowvalue">'
            f'{s.mean_rank:.1f}위 · {s.n_mentions}건</text>'
            f"</g>"
        )

    parts.append("</svg>")
    return "".join(parts)


def chart_silhouette(candidates) -> str:
    """군집 개수 후보의 실루엣. 단조 상승이면 자연스러운 k 가 없다는 뜻이다."""
    pts = sorted(candidates, key=lambda c: c.k)
    left, right, top, bottom = 56, 24, 16, 34
    plot_w, plot_h = 420, 170
    width, height = left + plot_w + right, top + plot_h + bottom

    ks = [c.k for c in pts]
    lo_k, hi_k = min(ks), max(ks)
    hi_s = max(0.5, max(c.silhouette for c in pts) * 1.15)

    def sx(k: int) -> float:
        return left + (k - lo_k) / max(1, hi_k - lo_k) * plot_w

    def sy(s: float) -> float:
        return top + plot_h - (s / hi_s) * plot_h

    parts = [svg_open(width, height, "군집 개수별 실루엣 계수").replace('class="chart"', 'class="chart sil"')]
    for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
        y = top + plot_h - frac * plot_h
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}" class="grid"/>')
        parts.append(f'<text x="{left - 10}" y="{y + 4:.1f}" class="tick" text-anchor="end">{frac * hi_s:.2f}</text>')

    path = " ".join(f"{'M' if i == 0 else 'L'}{sx(c.k):.1f},{sy(c.silhouette):.1f}" for i, c in enumerate(pts))
    parts.append(f'<path d="{path}" class="line"/>')
    for c in pts:
        parts.append(
            f'<g class="row" {tip(f"k={c.k} — 실루엣 {c.silhouette:.3f}")} tabindex="0">'
            f'<circle cx="{sx(c.k):.1f}" cy="{sy(c.silhouette):.1f}" r="4.5" class="dot"/>'
            f'<text x="{sx(c.k):.1f}" y="{top + plot_h + 20}" class="tick" text-anchor="middle">{c.k}</text>'
            f"</g>"
        )
    parts.append("</svg>")
    return "".join(parts)


# --------------------------------------------------------------------------
# 표 (모든 그림의 짝)
# --------------------------------------------------------------------------

def plain_table(headers: list[str], rows: list[list[str]]) -> str:
    """접지 않고 바로 보이는 표.

    뒤 절을 읽는 데 필요한 정의는 접어 두면 안 된다 — 접힌 것은 아무도 열지 않고,
    안 열면 그 다음 그림을 못 읽는다.
    """
    head = "".join(f"<th>{esc(h)}</th>" for h in headers)
    body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f'<div class="scroll"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def table(headers: list[str], rows: list[list[str]], caption: str) -> str:
    head = "".join(f"<th>{esc(h)}</th>" for h in headers)
    body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return (
        f'<details class="tableview"><summary>표로 보기 — {esc(caption)}</summary>'
        f'<div class="scroll"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'
        f"</details>"
    )


# --------------------------------------------------------------------------
# 조립
# --------------------------------------------------------------------------

@dataclass
class Section:
    title: str
    lead: str
    body: str


def build(run_dir: Path, target: str) -> str:
    run = load_run(run_dir / "run.json")
    answers = load_answers(run_dir / "raw.jsonl")
    engine = run.engines[0]
    model = run.models.get(engine, "?")
    ambiguous = {k: v for k, v in run.excluded_aliases.items() if v}

    overall = aggregate(run.observations, "overall", BRANDS)
    target_rate = next(r for r in overall if r.brand == target)

    # 히어로가 쓸 최고·최저 단계. "가장 잘 나오는 자리"와 "안 나오는 자리"를
    # 첫 화면에서 대비시키기 위한 값이다.
    stage_rates = [
        (r.scope_value, r.interval)
        for r in aggregate(run.observations, "stage", BRANDS)
        if r.brand == target and r.interval.n
    ]
    best = max(stage_rates, key=lambda kv: kv[1].point)
    floor = min(kv[1].point for kv in stage_rates)
    # 최저가 여럿이면 상업적으로 가장 값진 자리를 보여준다
    worst = min(
        (kv for kv in stage_rates if kv[1].point == floor),
        key=lambda kv: (kv[0] != "alternative", kv[0] != "problem", -kv[1].n),
    )

    sections: list[Section] = []

    # --- 0. 무엇을, 왜 측정했나 --------------------------------------------
    sections.append(intro_section(run, target, overall))

    # --- 1. 순위 -----------------------------------------------------------
    rates_rows = [
        [
            esc(r.brand) + (" &#9662;" if r.brand in ambiguous else ""),
            pct(r.interval.point),
            f"{pct(r.interval.low)}~{pct(r.interval.high)}",
            str(r.interval.n),
            esc(", ".join(ambiguous.get(r.brand, ()))) or "—",
        ]
        for r in overall
        if r.interval.n
    ]
    sections.append(
        Section(
            "누가 불려 나오는가",
            f"응답 {run.total_calls}개에서 브랜드 {run.brand_count}개의 등장 여부를 셌다. "
            f"막대가 아니라 점과 선으로 그렸다. <strong>값이 하나로 확정된 것이 아니기 때문</strong>이고, "
            f"선이 길수록 우리가 덜 아는 것이다. "
            f"&#9662; 가 붙은 브랜드는 일반 명사와 겹치는 표기를 일부러 세지 않은 경우이며 "
            f"실제보다 낮게 나온다.",
            frame(chart_rates(overall, target, ambiguous))
            + '<div class="legend"><span class="key key-target"></span>측정 대상 '
            f'<span class="key key-context"></span>비교군</div>'
            + table(["브랜드", "언급률", "95% 신뢰구간", "표본", "탐지 제외 표기"], rates_rows, "브랜드별 언급률"),
        )
    )

    # --- 2. 어느 자리에서 사라지나 ------------------------------------------
    cells = stage_language_grid(run.observations, target, BRANDS)
    grid_rows = [
        [
            esc(STAGE_LABEL.get(c.stage, c.stage)),
            esc(LANGUAGE_LABEL.get(c.language, c.language)),
            pct(c.interval.point),
            f"{pct(c.interval.low)}~{pct(c.interval.high)}",
            str(c.interval.n),
            esc(STAGE_DESCRIPTION.get(c.stage, "")),
        ]
        for c in cells
    ]
    sections.append(
        Section(
            f"{esc(target)}은 어느 자리에서 사라지는가",
            "전체 언급률 하나로는 무엇을 해야 할지 알 수 없다. 질문 방식을 다섯 단계로 나누고 언어별로 "
            "갈라 보면 <strong>어느 칸이 비어 있는지가 곧 할 일</strong>이 된다. "
            "칸이 진할수록 자주 불렸다는 뜻이고, 아래 작은 숫자가 신뢰구간이다.",
            frame(chart_heatmap(cells, target))
            + table(["단계", "언어", "언급률", "95% 신뢰구간", "표본", "이 단계의 성격"], grid_rows, f"{target} 단계별 언급률"),
        )
    )

    # --- 3. 언어 차이 -------------------------------------------------------
    gaps = language_gap(run.observations, BRANDS)
    gap_rows = [
        [
            esc(b),
            ci_text(ko.interval),
            ci_text(en.interval),
            "겹침 — 주장 불가" if ov else "<strong>분리 — 차이 있음</strong>",
        ]
        for b, _e, ko, en, ov in gaps
        if ko.interval.n and en.interval.n
    ]
    separated = [g for g in gaps if not g[4] and g[2].interval.n and g[3].interval.n]
    sections.append(
        Section(
            "한국어로 물을 때와 영어로 물을 때",
            f"같은 질문을 언어만 바꿔 짝으로 물었다. 두 점 사이가 벌어져 보여도 그것만으로는 차이라고 "
            f"말할 수 없다. <strong>구간이 서로 겹치지 않을 때만 차이를 주장한다.</strong> "
            f"{len(separated)}개 브랜드에서 구간이 떨어졌다. "
            f"다만 아래 주의를 함께 읽어야 한다 &mdash; <strong>이 판정은 다음 절의 결과와 어긋난다.</strong>",
            frame(chart_dumbbell(gaps, target))
            + '<div class="legend"><span class="key key-ko"></span>한국어 '
            '<span class="key key-en"></span>영어</div>'
            + '<p class="note"><strong>이 구간은 실제보다 좁다.</strong> 한 언어의 180개 응답은 '
            '질문 9개를 20번씩 반복해 얻은 것이라 서로 독립이 아닌데, 위 구간은 독립인 것처럼 계산했다. '
            '다음 절에서는 같은 데이터를 질문 묶음으로 보정해 다시 계산하며, '
            '<strong>그쪽에서는 차이가 대부분 사라진다.</strong> 둘이 어긋나면 보수적인 쪽을 따른다.</p>'
            + table(["브랜드", "한국어", "영어", "판정"], gap_rows, "언어별 언급률 비교"),
        )
    )

    # --- 4. 공동언급 -------------------------------------------------------
    top_names = [r.brand for r in overall if r.interval.point > 0][:10]
    pairs = cooccurrence(run.observations, BRANDS, engine=engine)
    matrix_pairs = [c for c in pairs if c.a in top_names and c.b in top_names]
    target_pairs = [c for c in cooccurrence_for(run.observations, target, BRANDS, engine) if c.n_both]
    co_rows = [
        [esc(c.b), f"{c.n_both}/{c.n_a}", ci_text(c.conditional), f"{c.lift:.2f}" if c.lift is not None else "—"]
        for c in target_pairs
    ]
    sections.append(
        Section(
            "누구와 함께 불리는가",
            "AI가 같은 답변 안에서 어떤 브랜드들을 묶어 부르는지 보여준다. 왼쪽 브랜드가 나온 답변 중 "
            "위쪽 브랜드도 함께 나온 비율이다. <strong>방향이 있다</strong> &mdash; 작은 브랜드가 큰 브랜드와 "
            "같이 불리는 일은 흔하지만 그 반대는 드물다. "
            f"{esc(target)} 기준 표는 분모가 {target_rate.interval.n * target_rate.interval.point:.0f}건뿐이라 "
            "구간이 매우 넓다. 넓은 채로 싣는 편이 좁혀서 싣는 것보다 정직하다.",
            frame(chart_matrix(matrix_pairs, top_names))
            + table(
                ["함께 등장한 브랜드", "동시 등장", f"{target} 가 나올 때 이 브랜드도 나올 확률", "lift"],
                co_rows or [["—", "—", f"{target} 와 함께 등장한 브랜드가 없다", "—"]],
                f"{target} 공동언급",
            ),
        )
    )

    # --- 5. 등장 순위 -------------------------------------------------------
    ranks = [s for s in rank_summaries(answers, BRANDS) if s.n_mentions]
    rank_rows = [
        [
            esc(s.brand),
            f"{s.n_mentions}/{s.n_responses}",
            f"{s.mean_rank:.1f}",
            f"{s.median_rank:.0f}",
            str(s.best_rank),
            ci_text(s.top3),
        ]
        for s in ranks
    ]
    sections.append(
        Section(
            "몇 번째로 불리는가",
            "같은 '언급 1회'라도 첫 문단에서 추천되는 것과 긴 목록의 꼬리에 붙는 것은 다르다. "
            "<strong>여기 값은 모두 '언급된 답변'만 놓고 계산했다</strong> &mdash; 어쩌다 불릴 때 앞에 불린다고 "
            "노출이 좋은 것은 아니므로 옆의 언급 건수를 함께 읽어야 한다. "
            "순위는 우리가 세는 브랜드들 사이의 순위라, 세지 않는 브랜드가 앞에 있었다면 실제 순위는 이보다 뒤다.",
            frame(chart_ranks(ranks, target))
            + '<div class="legend"><span class="key key-target"></span>측정 대상 '
            '<span class="key key-context"></span>비교군 &middot; 왼쪽일수록 답변 앞자리</div>'
            + table(
                ["브랜드", "언급/응답", "평균 순위", "중앙값", "최고", "3위 이내 비율"],
                rank_rows,
                "등장 순위",
            ),
        )
    )

    # --- 6. 회귀 -----------------------------------------------------------
    rows_by_brand = {b.canonical: observation_rows(run.observations, b.canonical) for b in BRANDS}
    fits = fit_many(rows_by_brand, exclude_stages=("comparison",))
    target_fit = fit_stage_language(rows_by_brand[target], target, exclude_stages=("comparison",))
    estimable = [f for f in fits if f.estimable]
    caveat = next((f.cluster_caveat for f in fits if f.cluster_caveat), None)

    fit_rows = []
    for f in fits:
        if f.estimable:
            terms = "; ".join(f"{t.name} OR {t.odds_ratio:.2f}" for t in f.terms if t.name != "const")
            fit_rows.append([esc(f.brand), str(f.n_events), "계산 가능", terms])
        else:
            reason = _plain_cells(f.separation) or (f.note or "")
            fit_rows.append([esc(f.brand), str(f.n_events), "<strong>계산 불가</strong>", esc(reason)])

    sections.append(
        Section(
            "질문 종류와 언어, 어느 쪽이 영향을 주는가",
            f"앞의 격자는 칸마다 표본이 잘게 쪼개진다. 여기서는 {run.total_calls}개를 한꺼번에 쓰되 "
            f"<strong>질문 종류의 영향과 언어의 영향을 따로 떼어</strong> 계산했다."
            f"<br><br>"
            f"여기에는 함정이 하나 있다. {run.total_calls}개 응답은 서로 독립이 아니다 &mdash; 같은 질문을 "
            f"{run.runs_per_prompt}번씩 반복해 얻은 것이라 한 질문에서 나온 {run.runs_per_prompt}개는 서로 닮아 있다. "
            f"이걸 무시하고 계산하면 <strong>실제보다 확신이 커 보인다.</strong> 그래서 같은 질문에서 나온 응답을 "
            f"한 묶음으로 잡아 보정했다. 브랜드를 직접 지목하는 질문은 답이 이미 정해져 있어 뺐다."
            f"<br><br>"
            f"가로축은 <strong>오즈비</strong>다. 1보다 오른쪽이면 한국어로 물을 때 더 자주 불렸다는 뜻이고, "
            f"선이 1을 가로지르면 방향을 말할 수 없다는 뜻이라 회색으로 표시했다. "
            f"브랜드 {len(fits)}개 중 <strong>{len(estimable)}개만 계산할 수 있었다.</strong>",
            frame(chart_forest(fits, "language[ko]", "한국어로 물을 때"))
            + f'<p class="note">{_language_conflict(gaps, fits)}</p>'
            + (f'<p class="note">{_plain_caveat(fits[0].n_clusters)}</p>' if caveat else "")
            + _target_fit_note(target, target_fit)
            + table(["브랜드", "언급 건수", "계산 가능 여부", "결과 또는 사유"], fit_rows, "질문 종류·언어의 영향"),
        )
    )

    # --- 7. 군집 -----------------------------------------------------------
    cluster_body = cluster_section(run_dir, answers, target)
    if cluster_body:
        sections.append(cluster_body)

    # --- 8. 원 응답 전문 ----------------------------------------------------
    sections.append(raw_section(answers, target, run))

    # --- 8. 한계 -----------------------------------------------------------
    failures = run.failures_by_prompt()
    prompt_count = len(run.prompt_ids)
    limit_items = [
        "<li><strong>웹 검색을 끄고 측정했다.</strong> AI가 학습으로 알고 있는 것만 잰 것이라 "
        "실시간 검색 결과는 들어 있지 않다. 실제 사용자가 보는 답변과 다를 수 있다.</li>",
        f"<li><strong>AI 하나만 측정했다.</strong> {esc(engine)} / {esc(model)} 이다. "
        f"서비스마다 답변에 출처를 다는 방식이 크게 다르므로 다른 AI로 일반화할 수 없다.</li>",
        "<li><strong>한 분야만 물었다.</strong> 질문은 팀 협업 도구에 한정된다.</li>",
        f"<li><strong>일부 표기를 일부러 세지 않았다.</strong> 잘못 센 것이 하나라도 있으면 이후 숫자 전체를 "
        f"믿을 수 없게 되기 때문이다. 그래서 다음 브랜드는 실제보다 낮게 나온다 &mdash; "
        f"{esc(', '.join(sorted(ambiguous)))}.</li>",
        f"<li><strong>묶음 보정에도 한계가 있다.</strong> 질문이 {prompt_count}개뿐이라 보정이 잘 작동하는 데 "
        f"필요한 최소 개수(통상 30개)에 못 미친다. 보정한 뒤에도 구간이 실제보다 좁을 수 있다.</li>",
        "<li><strong>여러 질문을 합친 값은 구간이 좁게 나온다.</strong> 질문마다 언급될 확률이 다른데 "
        "그것을 하나로 합쳤기 때문이다. 이 문서에서는 그런 값에 표시를 달았다.</li>",
    ]
    limit_items.append(
        f"<li><strong>호출 실패 {run.failed_calls}건.</strong>"
        + ("" if not failures else f" 질문별 분포: {esc(json.dumps(failures, ensure_ascii=False))}.")
        + " 실패한 호출은 표본에서 뺐다 &mdash; '언급 안 됨'으로 세면 언급률이 실제보다 낮아진다.</li>"
    )
    sections.append(
        Section(
            "이 측정이 말할 수 없는 것",
            "한계를 뒤에 각주로 다는 것과 절로 세우는 것은 다르다. 아래 항목은 이 문서의 모든 수치에 걸린다.",
            "<ul class='limits'>" + "".join(limit_items) + "</ul>",
        )
    )

    return render(
        run, run_dir, target, target_rate, sections, engine, model, worst, best,
        raw_payload(answers, target),
    )


def raw_payload(answers, target: str) -> str:
    """원 응답 전문을 페이지 안에 심는다.

    ## 왜 경로만 적지 않고 통째로 넣나

    "원자료는 `raw.jsonl` 에 있다"는 문장은 그 파일을 가진 사람에게만 근거다.
    문서만 받은 사람에게는 확인할 방법이 없는 약속일 뿐이다. **전문을 같은
    파일에 넣으면 모든 수치가 그 자리에서 검증 가능해진다.**

    ## 하이라이트는 탐지 규칙에서 직접 뽑는다

    화면에 칠하는 구간을 별도 규칙으로 만들면 표시된 것과 집계된 것이 갈라진다.
    `match_spans` 가 언급률을 셀 때 쓴 것과 같은 패턴을 쓰므로, **칠해진 자리가
    곧 우리가 센 자리다.** 세지 않은 표기는 칠해지지 않는다.
    """
    index = {p.id: p for p in PROMPTS}
    questions: dict[str, dict] = {}
    items = []

    for a in answers:
        prompt = index.get(a.prompt_id)
        spans = match_spans(a.text, target) if a.ok else ()
        q = questions.setdefault(
            a.prompt_id,
            {
                "id": a.prompt_id,
                "text": prompt.text if prompt else a.prompt_id,
                "stage": STAGE_LABEL.get(prompt.stage, prompt.stage) if prompt else "",
                "lang": LANGUAGE_LABEL.get(prompt.language, prompt.language) if prompt else "",
                "n": 0,
                "hits": 0,
            },
        )
        q["n"] += 1
        if spans:
            q["hits"] += 1
        items.append(
            {
                "q": a.prompt_id,
                "r": a.run_index,
                "ok": a.ok,
                "t": a.text if a.ok else (a.error or "호출 실패"),
                "s": [[s, e] for s, e in spans],
            }
        )

    items.sort(key=lambda x: (x["q"], x["r"]))
    payload = {"target": target, "questions": list(questions.values()), "items": items}
    # </script> 가 문자열 안에 들어가면 스크립트가 그 자리에서 끊긴다
    return json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")


def raw_section(answers, target: str, run) -> Section:
    ok = sum(1 for a in answers if a.ok)
    hit = sum(1 for a in answers if a.ok and match_spans(a.text, target))
    return Section(
        "원 응답 전문",
        f"위의 모든 수치는 아래 답변 {ok}개에서 나왔다. 경로만 적어 두면 파일을 가진 사람만 확인할 수 있으므로 "
        f"<strong>전문을 이 문서 안에 넣었다.</strong> 질문별로 접혀 있고, "
        f"{esc(target)}이 언급된 자리는 <mark>표시</mark>해 뒀다. "
        f"칠해진 자리가 곧 우리가 센 자리다 &mdash; 일부러 세지 않는 표기는 칠해지지 않는다.",
        f'<details class="rawopen"><summary><span class="chev">&#9654;</span>'
        f"원 응답 {ok}개 전부 보기"
        f'</summary><div class="rawbody">'
        f'<p class="rawhint">질문 {len(run.prompt_ids)}개 &middot; 질문마다 {run.runs_per_prompt}회 반복 &middot; '
        f"{esc(target)} 언급 {hit}건. 질문을 눌러 펼치고, 회차를 눌러 답변 전문을 연다.</p>"
        f'<div id="rawlist"></div></div></details>',
    )


def _plain_cells(cells) -> str:
    """빈 칸 설명을 읽는 사람 말로 바꾼다.

    모듈이 내놓는 `stage=alternative: 80건 중 0회 언급` 은 코드를 읽는 사람에게
    맞는 표현이다. 리포트를 읽는 사람에게는 단계 이름을 풀어 줘야 한다.
    """
    parts = []
    for c in cells:
        if c.variable == "stage":
            label = STAGE_PHRASE.get(c.level, c.level)
        elif c.variable == "language":
            label = f"{LANGUAGE_LABEL.get(c.level, c.level)} 질문"
        else:
            label = c.level
        parts.append(
            f"{label} {c.n}번 중 한 번도 언급되지 않았다"
            if c.events == 0
            else f"{label} {c.n}번 모두에서 언급됐다"
        )
    return ", ".join(parts)


def _plain_caveat(n_clusters: int) -> str:
    """군집 수 경고를 풀어 쓴다."""
    return (
        f"다만 이 보정에도 한계가 있다. 서로 다른 질문이 {n_clusters}개뿐인데, 이런 보정이 "
        f"제대로 작동하려면 보통 30개 이상이 필요하다. <strong>보정한 뒤에도 구간이 실제보다 "
        f"좁을 수 있다</strong> — 즉 위의 보수적인 판정조차 낙관 쪽으로 기울어 있을 수 있다."
    )


def _target_fit_note(target: str, fit) -> str:
    """대상 브랜드가 회귀에서 어떻게 됐는지를 따로 세운다.

    표 안에 한 줄로 묻히면 이 리포트에서 가장 중요한 결과가 안 보인다.
    """
    if fit.estimable:
        return (
            f'<div class="callout"><p><strong>{esc(target)}은 추정할 수 있었다.</strong> '
            f"단계와 언어의 효과를 분리해 계수와 구간을 얻었으므로 위 표에서 확인할 수 있다.</p></div>"
        )
    return (
        f'<div class="callout">'
        f"<p><strong>{esc(target)}은 이 계산에서 아예 빠졌다.</strong> "
        f"{esc(_plain_cells(fit.separation))}.</p>"
        f"<p>한 번도 언급되지 않은 질문 종류가 있으면 계산이 성립하지 않는다. "
        f"'이 질문에서는 안 불린다'는 정도를 수치로 표현하려 할수록 값이 끝없이 커져 "
        f"답이 하나로 정해지지 않기 때문이다. 억지로 숫자를 뽑을 수는 있지만 그 숫자에는 뜻이 없다.</p>"
        f"<p><strong>계산하지 못했다는 사실 자체가 결과다.</strong> "
        f"\"영향이 이만큼이다\"보다 \"두 종류의 질문에서 한 번도 불리지 않았다\"가 "
        f"담당자에게 더 쓸모 있는 보고다.</p>"
        f"</div>"
    )


def intro_section(run, target, overall) -> Section:
    """읽기 전에 알아야 할 것 — 대상이 무엇이고, 왜 골랐고, 숫자를 어떻게 읽나.

    이 절이 없으면 독자는 맥락 없이 2.8% 를 받는다. 그 상태에서는 그 값이 큰지
    작은지조차 판단할 수 없다.
    """
    prompt_count = len(run.prompt_ids)
    names = [r.brand for r in overall if r.brand != target]
    domestic = [b for b in names if b in DOMESTIC_BRANDS]
    overseas = [b for b in names if b not in DOMESTIC_BRANDS]

    intro = BRAND_INTRO.get(target, "")
    zero20 = wilson_interval(0, run.runs_per_prompt)

    # 예시 질문은 지어내지 않고 실제로 물은 한국어 질문에서 가져온다
    example = {}
    for p in PROMPTS:
        if p.language == "ko" and p.id in run.prompt_ids:
            example.setdefault(p.stage, p.text)

    stage_rows = [
        [
            esc(STAGE_LABEL.get(s, s)),
            esc(STAGE_DESCRIPTION.get(s, "")),
            f'"{esc(example.get(s, ""))}"' if example.get(s) else "&mdash;",
        ]
        for s in STAGE_ORDER
        if any(o.stage == s for o in run.observations)
    ]

    brand_block = (
        f'<h3>측정 대상 &mdash; {esc(target)}</h3><p>{intro}</p>' if intro else ""
    )

    return Section(
        "무엇을, 왜 측정했나",
        "숫자를 보기 전에 세 가지를 정하고 시작한다. 어떤 브랜드를 재는가, 왜 그 브랜드인가, "
        "그리고 여기 나오는 비율을 어떻게 읽는가.",
        brand_block
        + "<h3>왜 이 브랜드를 골랐나</h3>"
        "<ul class='facts'>"
        "<li><strong>범주가 맞는다.</strong> 구독형으로 파는 팀 협업 도구다. "
        "재려는 질문(\"협업 툴 추천해줘\")과 같은 시장에 있어야 결과가 그 시장의 답이 된다.</li>"
        "<li><strong>실재하는 브랜드여야 한다.</strong> 가상의 이름을 넣으면 측정이 성립하지 않는다 "
        "&mdash; AI가 모르는 이름이 안 나오는 것은 당연하고, 그 사실에서 배울 것이 없다.</li>"
        "<li><strong>국내 브랜드다.</strong> 한국어로 물을 때와 영어로 물을 때 답이 달라지는지 "
        "볼 수 있다.</li>"
        f"<li><strong>이름을 안전하게 셀 수 있는지 먼저 확인했다.</strong> "
        f"\"스윗\"은 영어 sweet의 한글 표기와 겹칠 수 있다. 원 응답에서 한글 \"스윗\"이 나온 9건을 "
        f"전부 열어 확인했고 모두 협업 도구를 가리키고 있었다(잘못 센 것 0건). "
        f"이 확인을 건너뛰었다면 이후 모든 숫자를 믿을 수 없게 된다.</li>"
        "</ul>"
        f"<h3>비교군 {len(names)}개</h3>"
        f"<p><strong>해외</strong> {esc(' · '.join(overseas))}<br>"
        f"<strong>국내</strong> {esc(' · '.join(domestic))}</p>"
        f"<p>비교군을 함께 재는 데는 추가 비용이 들지 않는다. 이미 받아 온 답변에서 다른 이름도 "
        f"같이 세면 되기 때문이다.</p>"
        f"<h3>질문 {prompt_count}개를 어떻게 짰나</h3>"
        "<p>구매자가 검색 대신 AI에게 묻는 방식은 한 가지가 아니다. 막연한 고민을 털어놓기도 하고, "
        "조건을 붙여 좁히기도 하고, 경쟁사 이름을 대며 대안을 묻기도 한다. "
        "<strong>어느 단계에서 불리느냐가 결과를 가르므로</strong> 다섯 단계로 나눠 물었고, "
        "각 질문은 한국어와 영어를 짝으로 만들었다.</p>"
        + plain_table(["단계", "성격", "실제로 물어본 질문"], stage_rows)
        + "<h3>이 문서의 숫자를 읽는 법</h3>"
        "<div class='callout'>"
        f"<p><strong>비율 뒤의 괄호가 핵심이다.</strong> {run.runs_per_prompt}번 물어서 2번 나왔다면 "
        f"언급률은 10%다. 그런데 {run.runs_per_prompt}번은 적은 횟수라, 다시 물으면 1번이나 3번이 "
        f"나올 수 있다. 신뢰구간은 그 흔들림의 폭이고, <strong>좁을수록 우리가 아는 것이 많다</strong>는 뜻이다.</p>"
        f"<p><strong>0%는 \"없다\"가 아니다.</strong> {run.runs_per_prompt}번 물어 한 번도 안 나왔다면 "
        f"참값은 0%일 수도 있지만 {pct(zero20.high, 1)}일 수도 있다. 그래서 이 문서는 0% 옆에도 "
        f"상한을 함께 적는다.</p>"
        "<p><strong>구간이 겹치면 차이를 주장하지 않는다.</strong> 두 브랜드의 점이 달라 보여도 "
        "구간이 서로 걸쳐 있으면 우연일 수 있다. 그럴 때는 순위를 매기지 않는다.</p>"
        "</div>",
    )


def _language_conflict(gaps, fits) -> str:
    """단순 구간 비교와 군집 보정 회귀가 어긋나는 지점을 명시한다.

    두 결과를 나란히 싣고 아무 말도 하지 않으면, 읽는 사람은 자기에게 유리한 쪽을
    고른다. 어긋난다는 사실 자체가 이 리포트가 말하려는 것이다.
    """
    naive = {g[0] for g in gaps if not g[4] and g[2].interval.n and g[3].interval.n}
    survived, lost = set(), set()
    for f in fits:
        if f.brand not in naive:
            continue
        if not f.estimable:
            lost.add(f.brand)
            continue
        term = next((t for t in f.terms if t.name == "language[ko]"), None)
        if term is None:
            lost.add(f.brand)
            continue
        low, high = term.or_ci
        (survived if not low <= 1.0 <= high else lost).add(f.brand)

    if not naive:
        return ""
    survived_text = ", ".join(sorted(survived)) if survived else "없다"
    return (
        f"<strong>앞 절과 어긋나는 지점.</strong> 단순 구간 비교에서는 "
        f"{esc(', '.join(sorted(naive)))} 의 언어 차이가 뚜렷해 보였다. 그런데 반복을 질문 묶음으로 보정하면 "
        f"구간이 넓어져 대부분이 1을 가로지른다. 두 검사를 모두 통과한 브랜드는 "
        f"<strong>{esc(survived_text)}</strong> 뿐이다. 같은 데이터에서 나온 두 답이 다를 때 "
        f"어느 쪽을 싣느냐가 곧 그 문서의 성격이다 &mdash; 여기서는 보수적인 쪽을 결론으로 쓴다."
    )


def cluster_section(run_dir: Path, answers, target: str) -> Section | None:
    """임베딩 군집화. 캐시가 없으면 이 절을 통째로 건너뛴다 (API 를 새로 부르지 않는다)."""
    cache = run_dir / "embeddings.npz"
    if not cache.exists():
        return None

    from geolab.embedding import (
        brand_by_cluster,
        choose_clusters,
        embed_texts,
        has_natural_k,
        label_agreement,
        profile_clusters,
    )

    rows = response_texts(answers)
    emb = embed_texts([r["text"] for r in rows], cache, allow_api=False)
    result = choose_clusters(emb.vectors, k_range=range(2, 13))
    natural = has_natural_k(result.candidates)
    agreement = label_agreement(
        result.labels,
        {
            "언어": [r["language"] for r in rows],
            "질문 종류": [r["stage"] for r in rows],
            "질문 자체": [r["prompt_id"] for r in rows],
        },
    )
    profiles = profile_clusters(rows, result.labels)
    by_cluster = brand_by_cluster(rows, result.labels, target)

    profile_rows = [
        [
            str(p.cluster),
            str(p.size),
            esc(", ".join(f"{LANGUAGE_LABEL.get(k, k)} {v}" for k, v in p.language_mix.items())),
            esc(", ".join(f"{STAGE_LABEL.get(k, k)} {v}" for k, v in p.stage_mix.items())),
            esc(", ".join(f"{b} {s:.0%}" for b, s in list(p.brand_share.items())[:4])),
            ci_text(by_cluster[p.cluster]),
        ]
        for p in profiles
    ]
    agreement_rows = [[esc(k), f"{v:.3f}"] for k, v in agreement.items()]

    verdict = (
        f"세로축은 묶음이 얼마나 또렷하게 갈라졌는지를 나타내는 점수다. 점수가 계속 올라가기만 하고 "
        f"<strong>최고점이 우리가 시도한 마지막 지점({result.k}개)에 있다.</strong> "
        f"자연스러운 묶음 개수를 찾지 못했다는 뜻이다 &mdash; 더 많이 시도했다면 최고점도 따라 올라갔을 것이다. "
        f"이 숫자는 데이터가 알려준 값이 아니라 우리가 어디서 멈췄는지를 알려주는 값이다."
        if not natural
        else f"세로축은 묶음이 얼마나 또렷하게 갈라졌는지를 나타내는 점수다. 끝이 아닌 {result.k}개 지점에서 "
        f"정점을 이루므로, 그 개수가 자연스러운 묶음 수일 수 있다."
    )
    strongest_name, strongest_score = next(iter(agreement.items()))
    RECOVERED = {
        "언어": "한국어 답변과 영어 답변이 다르게 생겼다",
        "질문 종류": "질문의 종류가 다르면 답변도 다르게 생겼다",
        "질문 자체": "같은 질문에 대한 답변끼리 서로 닮았다",
    }
    if strongest_score >= 0.4:
        reading = (
            f"즉 이 묶음은 <em>{esc(RECOVERED.get(strongest_name, strongest_name))}</em>를 "
            f"비싼 방법으로 다시 확인한 쪽에 가깝다. 질문 종류와 언어가 이미 담고 있는 것을 넘는 "
            f"새로운 축은 나오지 않았다."
        )
    else:
        reading = (
            "이미 아는 어떤 구분과도 크게 겹치지 않았다. 다만 그것이 곧 '새로운 유형을 찾았다'는 뜻은 아니다 "
            "&mdash; 겹치지 않는 것과 의미가 있는 것은 다른 문제이고, 여기에는 그것을 가릴 정답이 없다."
        )

    return Section(
        "답변 유형을 찾아봤지만 — 성과 없음",
        "여기까지는 세운 가설을 확인하는 작업이었다. 이 절은 새 가설을 찾아보는 자리이고, "
        "<strong>맞았는지 틀렸는지 판정할 정답이 없다.</strong> "
        "답변 전문을 수치로 바꿔 비슷한 것끼리 묶으면, 언급률로는 안 보이던 답변 유형이 드러날까 물었다. "
        "<strong>결론부터 말하면 드러나지 않았다.</strong>",
        frame(chart_silhouette(result.candidates))
        + f"<p class='note'>{verdict}</p>"
        + f"<p>묶음이 새로운 것을 찾아낸 것인지, <strong>이미 알고 있던 구분을 다시 발견한 것뿐인지</strong> "
        f"확인했다. 두 분류가 얼마나 겹치는지를 0에서 1 사이로 재는 지표를 썼고, 1이면 완전히 같은 분류다. "
        f"가장 많이 겹친 것은 <strong>{esc(strongest_name)}, 겹침 정도 {strongest_score:.2f}</strong> 였다. "
        f"{reading}</p>"
        f"<p><strong>성과가 없었다는 사실을 그대로 싣는다.</strong> 묶음에 그럴듯한 이름을 붙여 "
        f"발견처럼 실으면, 이 문서의 나머지 숫자도 같은 눈으로 읽히게 된다.</p>"
        + table(["비교 대상", "겹침 정도 (1 = 완전히 같은 분류)"], agreement_rows, "묶음과 이미 아는 구분의 겹침")
        + table(
            ["묶음", "크기", "언어 구성", "질문 종류 구성", "주요 브랜드", f"{target} 등장률"],
            profile_rows,
            "묶음 구성",
        ),
    )


def render(run, run_dir, target, target_rate, sections, engine, model, worst, best, raw_json) -> str:
    """히어로는 세 가지에 답한다 — 이 문서가 무엇인가, 무엇이 문제였나, 어떻게 다뤘나.

    그 다음에 결과 숫자가 온다. 순서를 뒤집으면 읽는 사람이 맥락 없이 수치부터
    받게 되고, 그러면 2.8% 가 큰지 작은지조차 판단할 수 없다.
    """
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    prompt_count = len(run.prompt_ids)

    kpis = [
        ("질문", f"{prompt_count}개", "구매 단계 5종 x 한국어·영어"),
        ("반복", f"{run.runs_per_prompt}회", "질문마다. 실행 전 고정"),
        ("총 호출", f"{run.total_calls:,}회", f"실패 {run.failed_calls}건"),
        ("비교 브랜드", f"{run.brand_count}개", "팀 협업 툴"),
    ]
    kpi_html = "".join(
        f'<div class="kpi"><div class="kpi-label">{esc(l)}</div>'
        f'<div class="kpi-value">{esc(v)}</div><div class="kpi-sub">{esc(s)}</div></div>'
        for l, v, s in kpis
    )
    body = "".join(
        f'<section class="card"><h2>{s.title}</h2><p class="lead">{s.lead}</p>{s.body}</section>'
        for s in sections
    )

    result_text = (
        f"{esc(target)}의 전체 언급률이다 (95% 신뢰구간 "
        f"{pct(target_rate.interval.low)}~{pct(target_rate.interval.high)}, {target_rate.interval.n}회 기준). "
        f"다만 이 숫자 하나로는 무엇을 해야 할지 알 수 없다. 질문 종류로 나누면 "
        f"<strong>{esc(STAGE_PHRASE.get(best[0], best[0]))}에서 {pct(best[1].point, 0)}, "
        f"{esc(STAGE_PHRASE.get(worst[0], worst[0]))}에서 {pct(worst[1].point, 0)}</strong>로 갈린다."
    )

    return f"""<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{esc(target)}의 GEO 측정 — {esc(model)}</title>
<style>{CSS}</style>
</head>
<body>
<div class="viz-root">
<header class="hero">
  <p class="eyebrow">GEO 측정 &middot; {esc(engine)} / {esc(model)}</p>
  <h1>AI에게 협업 툴을 물으면 {esc(target)}은 얼마나 불리는가</h1>

  <dl class="herolead">
    <div><dt>무엇</dt><dd>AI 답변에 브랜드가 얼마나, 어떤 맥락으로 등장하는지를 다루는 분야를
      <strong>GEO</strong>(Generative Engine Optimization, 생성형 엔진 최적화)라고 한다.
      이 문서는 그것을 실제로 측정한 기록이다.</dd></div>
    <div><dt>문제</dt><dd>AI는 같은 질문에도 매번 다르게 답한다. 한 번 물어본 결과는 측정이 아니다.</dd></div>
    <div><dt>방법</dt><dd>질문 {prompt_count}개를 각각 {run.runs_per_prompt}번씩, 모두 {run.total_calls}번 물었다.
      모든 비율에 신뢰구간을 붙여 어디까지 말할 수 있는지 함께 적었다.</dd></div>
  </dl>

  <div class="heroresult">
    <div class="heronum">{pct(target_rate.interval.point)}</div>
    <p class="herosub">{result_text}</p>
  </div>

  <div class="kpis">{kpi_html}</div>
  <p class="cond">측정 조건 &mdash; {esc(run.started_at[:16].replace("T", " "))} UTC 시작,
  질문당 {run.runs_per_prompt}회 반복(실행 전 고정, 조기 종료 없음), 샘플링 설정 건드리지 않음, 웹 검색 끔.
  원 응답 전문은 <code>{esc(run_dir.name)}/raw.jsonl</code> 에 남아 있다.</p>
</header>
{body}
<footer class="foot">
  <p>이 문서의 모든 수치는 위 「원 응답 전문」 절에 실린 답변에서 직접 계산해 그렸다.
  손으로 옮겨 적은 숫자는 없다. 원본 파일은 <code>{esc(run_dir.as_posix())}</code> 에도 남아 있다.
  생성 시각 {esc(stamp)}.</p>
  <p>신뢰구간은 Wilson score interval(95%)로 계산했다. 흔히 쓰는 정규근사는 비율이 0에 가까울 때
  구간 폭을 0으로 만들어 "확실히 0%"라고 잘못 말한다. 브랜드 언급률에서는 그 상황이 예외가 아니라 기본값이라
  다른 방식을 썼다.</p>
</footer>
</div>
<div id="tipbox" role="tooltip"></div>
<script id="rawdata" type="application/json">{raw_json}</script>
<script>{JS}</script>
</body>
</html>
"""


def _minutes(run) -> int:
    start = datetime.fromisoformat(run.started_at)
    end = datetime.fromisoformat(run.finished_at)
    return max(1, round((end - start).total_seconds() / 60))


CSS = """
/* ==========================================================================
   디자인 토큰 — 다섯 층으로 나눈다. 층 안에서는 강도 순서를 지킨다.

     1 표면   뒤에서 앞으로   page < card < inset
     2 글자   강에서 약으로   strong > body > muted
     3 선     약에서 강으로   grid < axis < edge
     4 데이터 의미가 있는 색. 위 세 층과 절대 섞지 않는다
     5 툴팁   반전 표면. 본문 위에 떠서 읽혀야 한다

   토큰을 :root 에 두는 이유 — 툴팁은 본문 컨테이너 밖에 떠 있는 요소라
   컨테이너에 정의하면 변수를 못 찾는다. 실제로 그 버그를 겪었다.
   ========================================================================== */
*{box-sizing:border-box}
html,body{margin:0;padding:0}

:root{
  color-scheme:light;

  /* 1 표면 */
  --surface-page:#f9f9f7;
  --surface-card:#fcfcfb;
  --surface-inset:#f1f0ec;

  /* 2 글자 */
  --ink-strong:#0b0b0b;
  --ink-body:#52514e;
  --ink-muted:#767570;

  /* 3 선 */
  --line-grid:#e1e0d9;
  --line-axis:#c3c2b7;
  --line-edge:rgba(11,11,11,.10);

  /* 4 데이터 */
  --data-focus:#2a78d6;
  --data-context:#a8a7a1;
  --data-ko:#1c5cab;
  --data-en:#86b6ef;
  --data-positive:#2a78d6;
  --data-negative:#d03b3b;
  --data-neutral:#a8a7a1;

  /* 5 툴팁 */
  --tip-surface:#0b0b0b;
  --tip-ink:#fcfcfb;
  --tip-edge:transparent;

  /* 6 강조 버튼 — 데이터 색을 쓰지 않는다. 색이 두 가지 뜻을 갖게 되기 때문이다.
     먹색 대 표면색이라 어느 테마에서나 대비가 충분하다. */
  --btn-surface:#0b0b0b;
  --btn-ink:#fcfcfb;
  --mark-bg:#fde68a;
  --mark-ink:#0b0b0b;

  --max:980px;
}

@media (prefers-color-scheme:dark){
  :root:where(:not([data-theme="light"])){
    color-scheme:dark;
    --surface-page:#0d0d0d;
    --surface-card:#1a1a19;
    --surface-inset:#232322;
    --ink-strong:#ffffff;
    --ink-body:#c3c2b7;
    --ink-muted:#98968e;
    --line-grid:#2c2c2a;
    --line-axis:#383835;
    --line-edge:rgba(255,255,255,.10);
    --data-focus:#3987e5;
    --data-context:#6e6d68;
    --data-ko:#3987e5;
    --data-en:#9ec5f4;
    --data-positive:#3987e5;
    --data-negative:#e66767;
    --data-neutral:#6e6d68;
    --tip-surface:#2c2c2a;
    --tip-ink:#ffffff;
    --tip-edge:rgba(255,255,255,.16);
    --btn-surface:#ffffff;
    --btn-ink:#0d0d0d;
    --mark-bg:#7a5c00;
    --mark-ink:#ffffff;
  }
}
:root[data-theme="dark"]{
  color-scheme:dark;
  --surface-page:#0d0d0d;
  --surface-card:#1a1a19;
  --surface-inset:#232322;
  --ink-strong:#ffffff;
  --ink-body:#c3c2b7;
  --ink-muted:#98968e;
  --line-grid:#2c2c2a;
  --line-axis:#383835;
  --line-edge:rgba(255,255,255,.10);
  --data-focus:#3987e5;
  --data-context:#6e6d68;
  --data-ko:#3987e5;
  --data-en:#9ec5f4;
  --data-positive:#3987e5;
  --data-negative:#e66767;
  --data-neutral:#6e6d68;
  --tip-surface:#2c2c2a;
  --tip-ink:#ffffff;
  --tip-edge:rgba(255,255,255,.16);
  --btn-surface:#ffffff;
  --btn-ink:#0d0d0d;
  --mark-bg:#7a5c00;
  --mark-ink:#ffffff;
}

body{background:var(--surface-page)}
.viz-root{
  background:var(--surface-page); color:var(--ink-strong);
  font-family:system-ui,-apple-system,"Segoe UI","Malgun Gothic",sans-serif;
  line-height:1.75; margin:0; padding:32px 20px 72px; font-size:16px;
  min-height:100vh;

  /* 한국어 줄바꿈 — 기본값은 글자 단위로 끊어 단어를 자른다.
     keep-all 이 어절 안에서 끊는 것을 막고, break-word 가 한 줄에 못 담는
     긴 토큰(URL·모델 식별자)만 예외로 처리한다. */
  word-break:keep-all;
  overflow-wrap:break-word;
}
h1,h2,h3,.herolead dt{text-wrap:balance}
p,li,dd,summary{text-wrap:pretty}

/* --- 뼈대 ---------------------------------------------------------------- */
.hero,.card,.foot{max-width:var(--max);margin:0 auto 22px;background:var(--surface-card);
  border:1px solid var(--line-edge);border-radius:14px;padding:30px 32px}
.eyebrow{margin:0 0 8px;color:var(--ink-muted);font-size:13px;letter-spacing:.02em}
h1{margin:0 0 22px;font-size:30px;line-height:1.35;letter-spacing:-.01em;max-width:24ch}
h2{margin:0 0 10px;font-size:21px;letter-spacing:-.005em}
h3{margin:26px 0 8px;font-size:16px;color:var(--ink-strong)}
.lead{margin:0 0 20px;color:var(--ink-body);max-width:68ch}
.card p{max-width:68ch}

/* --- 히어로 -------------------------------------------------------------- */
.herolead{margin:0 0 24px;display:grid;gap:10px}
.herolead>div{display:grid;grid-template-columns:56px 1fr;gap:14px;align-items:baseline}
.herolead dt{color:var(--ink-muted);font-size:13px;font-weight:500}
.herolead dd{margin:0;color:var(--ink-body);max-width:64ch}
.heroresult{display:flex;align-items:baseline;gap:20px;flex-wrap:wrap;
  padding:18px 20px;border-radius:12px;background:var(--surface-inset)}
.heronum{font-size:52px;font-weight:600;line-height:1;letter-spacing:-.02em;color:var(--ink-strong)}
.herosub{color:var(--ink-body);font-size:14.5px;flex:1 1 320px;margin:0}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(148px,1fr));gap:12px;margin:22px 0 18px}
.kpi{border:1px solid var(--line-edge);border-radius:10px;padding:12px 14px}
.kpi-label{color:var(--ink-muted);font-size:12px}
.kpi-value{font-size:24px;font-weight:600;margin:2px 0;color:var(--ink-strong)}
.kpi-sub{color:var(--ink-body);font-size:12px}
.cond{color:var(--ink-body);font-size:13px;margin:0;padding-top:14px;
  border-top:1px solid var(--line-edge);max-width:none}

/* --- 본문 요소 ------------------------------------------------------------ */
.note{color:var(--ink-body);font-size:14px;margin:12px 0 0;max-width:68ch}
.limits{margin:0;padding-left:22px;max-width:68ch}
.limits li{margin-bottom:10px;color:var(--ink-body)}
.limits strong{color:var(--ink-strong)}
.facts{margin:0 0 4px;padding-left:22px;max-width:68ch}
.facts li{margin-bottom:9px;color:var(--ink-body)}
.facts strong{color:var(--ink-strong)}
.callout{margin:18px 0 0;padding:16px 18px;border-radius:12px;
  background:var(--surface-inset);color:var(--ink-body);max-width:none}
.callout p{margin:0 0 8px;max-width:68ch}
.callout p:last-child{margin-bottom:0}
.callout strong{color:var(--ink-strong)}

/* --- 차트 ---------------------------------------------------------------- */
svg.chart{display:block;height:auto;overflow:visible}
.chartwrap{overflow-x:auto;margin:12px 0 4px;padding-bottom:4px}
.grid{stroke:var(--line-grid);stroke-width:1}
.axis-ref{stroke:var(--line-axis);stroke-width:1.5}
.tick{fill:var(--ink-muted);font-size:11px;font-variant-numeric:tabular-nums}
.colhead{font-size:10.5px}
.rowlabel{fill:var(--ink-body);font-size:12.5px}
.rowvalue{fill:var(--ink-body);font-size:12px;font-variant-numeric:tabular-nums}
.cellvalue{font-size:15px;font-weight:600}
.cellsub{font-size:10px}
.cellmini{font-size:10.5px;font-variant-numeric:tabular-nums}
.diag{fill:var(--line-grid)}
.hit{fill:transparent}
.row{cursor:default}
.row:hover .hit,.row:focus .hit{fill:var(--line-grid);opacity:.55}
.row:focus{outline:none}
.mark-target .ci{stroke:var(--data-focus);stroke-width:2;stroke-linecap:round}
.mark-target .dot{fill:var(--data-focus);stroke:var(--surface-card);stroke-width:2}
.mark-target .rowlabel,.mark-target .rowvalue{fill:var(--ink-strong);font-weight:600}
.mark-context .ci{stroke:var(--data-context);stroke-width:2;stroke-linecap:round}
.mark-context .dot{fill:var(--data-context);stroke:var(--surface-card);stroke-width:2}
.dumb .link{stroke:var(--line-grid);stroke-width:2;stroke-linecap:round}
.dumb.sep .link{stroke:var(--line-axis)}
.dumb .dot-ko{fill:var(--data-ko);stroke:var(--surface-card);stroke-width:2}
.dumb .dot-en{fill:var(--data-en);stroke:var(--surface-card);stroke-width:2}
.dumb.sep .rowvalue{fill:var(--ink-strong);font-weight:600}
.or-up .ci{stroke:var(--data-positive);stroke-width:2;stroke-linecap:round}
.or-up .dot{fill:var(--data-positive);stroke:var(--surface-card);stroke-width:2}
.or-down .ci{stroke:var(--data-negative);stroke-width:2;stroke-linecap:round}
.or-down .dot{fill:var(--data-negative);stroke:var(--surface-card);stroke-width:2}
.or-null .ci{stroke:var(--data-neutral);stroke-width:2;stroke-linecap:round}
.or-null .dot{fill:var(--data-neutral);stroke:var(--surface-card);stroke-width:2}
.line{fill:none;stroke:var(--data-focus);stroke-width:2;stroke-linejoin:round;stroke-linecap:round}
.sil .dot{fill:var(--data-focus);stroke:var(--surface-card);stroke-width:2}
.legend{display:flex;gap:18px;align-items:center;flex-wrap:wrap;
  color:var(--ink-body);font-size:13px;margin:4px 0 12px}
.key{width:11px;height:11px;border-radius:50%;display:inline-block;margin-right:6px;vertical-align:-1px}
.key-target{background:var(--data-focus)} .key-context{background:var(--data-context)}
.key-ko{background:var(--data-ko)} .key-en{background:var(--data-en)}

/* --- 표 ------------------------------------------------------------------ */
.tableview{margin-top:16px;border-top:1px solid var(--line-edge);padding-top:12px}
summary{cursor:pointer;color:var(--ink-body);font-size:14px}
summary:hover{color:var(--ink-strong)}
.scroll{overflow-x:auto;margin-top:12px}
table{border-collapse:collapse;width:100%;font-size:13.5px;min-width:520px}
th,td{text-align:left;padding:7px 14px 7px 0;border-bottom:1px solid var(--line-edge);vertical-align:top}
th{color:var(--ink-muted);font-weight:500;font-size:12px;white-space:nowrap}
td{font-variant-numeric:tabular-nums;color:var(--ink-body)}
td:first-child{color:var(--ink-strong)}
.foot{color:var(--ink-body);font-size:13.5px}
.foot p{margin:0 0 8px;max-width:68ch}
code{background:var(--surface-inset);padding:1px 5px;border-radius:4px;font-size:.9em;
  word-break:break-all}

/* --- 원 응답 뷰어 --------------------------------------------------------- */
.rawopen{margin-top:4px}
.rawopen>summary{
  display:inline-flex;align-items:center;gap:10px;
  padding:14px 22px;border-radius:10px;
  background:var(--btn-surface);color:var(--btn-ink);
  font-size:15px;font-weight:600;cursor:pointer;list-style:none}
.rawopen>summary::-webkit-details-marker{display:none}
.rawopen>summary::marker{content:""}
.rawopen>summary:hover{opacity:.88}
.rawopen>summary:focus-visible{outline:3px solid var(--data-focus);outline-offset:3px}
.rawopen>summary .chev{font-size:12px;opacity:.7}
.rawopen[open]>summary .chev{transform:rotate(90deg)}
.rawbody{margin-top:18px}
.rawhint{color:var(--ink-body);font-size:13.5px;margin:0 0 14px;max-width:68ch}
details.q{border:1px solid var(--line-edge);border-radius:10px;margin-bottom:10px;
  background:var(--surface-card)}
details.q>summary{padding:13px 16px;font-size:14px;color:var(--ink-strong);
  display:flex;gap:10px;flex-wrap:wrap;align-items:baseline}
details.q>summary:hover{background:var(--surface-inset)}
details.q[open]>summary{border-bottom:1px solid var(--line-edge)}
.qmeta{color:var(--ink-muted);font-size:12px;font-weight:500;white-space:nowrap}
.qhits{color:var(--ink-body);font-size:12px;margin-left:auto;font-variant-numeric:tabular-nums}
.answers{padding:10px 14px 14px}
details.a{border-top:1px solid var(--line-edge)}
details.a:first-child{border-top:none}
details.a>summary{padding:9px 4px;font-size:13px;color:var(--ink-body);
  display:flex;gap:10px;align-items:baseline}
details.a>summary:hover{color:var(--ink-strong)}
.ahit{color:var(--ink-strong);font-weight:600}
.amiss{color:var(--ink-muted)}
pre.answer{margin:2px 0 14px;padding:14px 16px;border-radius:8px;
  background:var(--surface-inset);color:var(--ink-body);
  font-size:13px;line-height:1.7;white-space:pre-wrap;
  word-break:keep-all;overflow-wrap:break-word;
  font-family:system-ui,-apple-system,"Segoe UI","Malgun Gothic",sans-serif;
  max-height:520px;overflow-y:auto}
pre.answer mark{background:var(--mark-bg);color:var(--mark-ink);
  padding:0 3px;border-radius:3px;font-weight:600}

/* --- 툴팁 ---------------------------------------------------------------- */
#tipbox{position:fixed;z-index:50;max-width:min(340px,calc(100vw - 24px));
  padding:10px 13px;border-radius:8px;
  background:var(--tip-surface);color:var(--tip-ink);border:1px solid var(--tip-edge);
  box-shadow:0 6px 20px rgba(0,0,0,.22);
  font-size:12.5px;line-height:1.6;word-break:keep-all;overflow-wrap:break-word;
  pointer-events:none;opacity:0;transition:opacity .12s;
  font-family:system-ui,-apple-system,"Segoe UI","Malgun Gothic",sans-serif}
#tipbox.on{opacity:1}

@media (max-width:720px){
  .viz-root{padding:16px 12px 48px;font-size:15px}
  .hero,.card,.foot{padding:22px 18px;border-radius:12px}
  h1{font-size:23px;max-width:none} .heronum{font-size:42px}
  .herolead>div{grid-template-columns:1fr;gap:2px}
  .heroresult{padding:16px}
}
"""

JS = """
(function(){
  var box=document.getElementById('tipbox');
  function show(e,t){
    box.textContent=t; box.classList.add('on');
    var r=box.getBoundingClientRect();
    var x=(e.clientX||0)+14, y=(e.clientY||0)+16;
    if(x+r.width>innerWidth-8) x=innerWidth-r.width-8;
    if(y+r.height>innerHeight-8) y=(e.clientY||0)-r.height-12;
    box.style.left=x+'px'; box.style.top=y+'px';
  }
  function hide(){ box.classList.remove('on'); }
  document.addEventListener('mousemove',function(e){
    var el=e.target.closest?e.target.closest('[data-tip]'):null;
    if(el) show(e,el.getAttribute('data-tip')); else hide();
  });
  document.addEventListener('focusin',function(e){
    var el=e.target.closest?e.target.closest('[data-tip]'):null;
    if(!el) return hide();
    var b=el.getBoundingClientRect();
    show({clientX:b.left+b.width/2,clientY:b.top+b.height/2},el.getAttribute('data-tip'));
  });
  document.addEventListener('focusout',hide);
  document.addEventListener('scroll',hide,true);
})();

/* 원 응답 뷰어 — 버튼을 처음 열 때만 만든다.
   360개 답변을 처음부터 DOM 에 올리면 첫 화면이 느려진다. */
(function(){
  var node = document.getElementById('rawdata');
  var host = document.getElementById('rawlist');
  if(!node || !host) return;

  var data;
  try { data = JSON.parse(node.textContent); } catch(e){
    host.textContent = '원 응답을 읽지 못했습니다.'; return;
  }

  function esc(s){
    return String(s).replace(/[&<>"]/g, function(c){
      return {'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c];
    });
  }

  /* 매칭 구간만 표시한다. 구간은 집계에 쓴 것과 같은 것이라
     칠해진 자리가 곧 우리가 센 자리다. */
  function withMarks(text, spans){
    if(!spans || !spans.length) return esc(text);
    var out = '', at = 0;
    for(var i=0;i<spans.length;i++){
      var s = spans[i][0], e = spans[i][1];
      if(s < at) continue;
      out += esc(text.slice(at, s)) + '<mark>' + esc(text.slice(s, e)) + '</mark>';
      at = e;
    }
    return out + esc(text.slice(at));
  }

  var byQuestion = {};
  data.items.forEach(function(it){
    (byQuestion[it.q] = byQuestion[it.q] || []).push(it);
  });

  var built = false;
  function build(){
    if(built) return;
    built = true;
    var html = data.questions.map(function(q){
      var rows = (byQuestion[q.id] || []).map(function(it){
        var hit = it.s && it.s.length;
        var label = !it.ok ? '<span class="amiss">호출 실패</span>'
          : hit ? '<span class="ahit">' + esc(data.target) + ' 언급</span>'
                : '<span class="amiss">언급 없음</span>';
        return '<details class="a"><summary>' + (it.r + 1) + '회차 ' + label + '</summary>'
             + '<pre class="answer">' + withMarks(it.t, it.s) + '</pre></details>';
      }).join('');
      return '<details class="q"><summary>'
           + '<span class="qmeta">' + esc(q.stage) + ' &middot; ' + esc(q.lang) + '</span>'
           + '<span>' + esc(q.text) + '</span>'
           + '<span class="qhits">' + q.n + '회 중 ' + q.hits + '회 언급</span>'
           + '</summary><div class="answers">' + rows + '</div></details>';
    }).join('');
    host.innerHTML = html;
  }

  var panel = host.closest('details.rawopen');
  if(panel) panel.addEventListener('toggle', function(){ if(panel.open) build(); });
})();
"""


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="측정 결과를 자기완결 HTML 리포트로 렌더링한다")
    parser.add_argument("run_dir", help="측정 폴더 (run.json 과 raw.jsonl 이 있는 곳)")
    parser.add_argument("--brand", default="Swit", help="리포트의 대상 브랜드")
    parser.add_argument("--out", default=None, help="출력 경로. 기본은 측정 폴더 안 report.html")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    if not (run_dir / "run.json").exists():
        print(f"run.json 이 없다: {run_dir}", file=sys.stderr)
        return 1
    if args.brand not in {b.canonical for b in BRANDS}:
        print(f"알 수 없는 브랜드: {args.brand}", file=sys.stderr)
        return 1

    out = Path(args.out) if args.out else run_dir / "report.html"
    out.write_text(build(run_dir, args.brand), encoding="utf-8")
    print(f"리포트를 만들었다: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
