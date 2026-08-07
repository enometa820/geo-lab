"""링크 하나로 전부 볼 수 있는 사이트를 만들어 GitHub Pages 에 올린다.

    python scripts/build-site.py --dry-run   만들기만 하고 열어본다
    python scripts/build-site.py             gh-pages 브랜치로 push 한다

## 왜 사이트인가

PDF 첨부는 오프라인에서도 열린다는 보증이고, 링크는 **아무것도 내려받지 않고
바로 본다**는 편의다. 둘은 대체 관계가 아니라 각자 다른 상황을 맡는다.

## 무엇을 싣고 무엇을 빼는가

답안 마크다운과 실측 리포트를 싣는다. **제출자 실명은 싣지 않는다** — 공개
저장소에서 메일 포장 문서를 뺀 것과 같은 판단이다. 메일을 받은 쪽은 보낸 사람을
이미 알고, 공개 웹에 이름을 색인시킬 이유는 없다.

## 검색 색인은 막는다

링크를 받은 사람은 그대로 열 수 있지만 검색에는 걸리지 않게 `noindex` 를 넣는다.
과제 문항은 의뢰한 회사가 다른 지원자에게도 쓸 수 있는 자산이고, 그것이 검색으로
발견되는 것과 링크로 공유되는 것은 다른 일이다.

## 정본은 마크다운이다

이 사이트는 생성물이다. 내용을 고치려면 `submission/*.md` 를 고치고 다시 돌린다.
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import markdown

ROOT = Path(__file__).resolve().parent.parent
SUBMISSION = ROOT / "submission"
REMOTE = "https://github.com/enometa820/geo-lab.git"
REPO_URL = "https://github.com/enometa820/geo-lab"

PAGES = [
    ("01-문항1-좋은-GEO의-기준.md", "munhang1.html", "문항 1", "좋은 GEO의 기준 3가지"),
    ("02-문항2-PRD.md", "munhang2.html", "문항 2", "AI 답변 노출 측정 도구 PRD"),
    ("03-작업-기록.md", "worklog.html", "작업 기록", "무엇을 어떤 순서로 했고 무엇을 버렸나"),
]

FRONTMATTER = re.compile(r"\A---\n.*?\n---\n", re.S)

TOKENS = """
:root{
  color-scheme:light;
  --surface-page:#f9f9f7; --surface-card:#fcfcfb; --surface-inset:#f1f0ec;
  --ink-strong:#0b0b0b; --ink-body:#52514e; --ink-muted:#767570;
  --line-edge:rgba(11,11,11,.10);
  --accent:#2a78d6;
}
@media (prefers-color-scheme:dark){
  :root:where(:not([data-theme="light"])){
    color-scheme:dark;
    --surface-page:#0d0d0d; --surface-card:#1a1a19; --surface-inset:#232322;
    --ink-strong:#ffffff; --ink-body:#c3c2b7; --ink-muted:#98968e;
    --line-edge:rgba(255,255,255,.10);
    --accent:#3987e5;
  }
}
/* 미디어 쿼리는 운영체제 설정만 따른다. 이 블록이 없으면 명시적 지정이
   먹지 않아 다크 모드를 확인조차 할 수 없다 — 실제로 그래서 못 볼 뻔했다. */
:root[data-theme="dark"]{
  color-scheme:dark;
  --surface-page:#0d0d0d; --surface-card:#1a1a19; --surface-inset:#232322;
  --ink-strong:#ffffff; --ink-body:#c3c2b7; --ink-muted:#98968e;
  --line-edge:rgba(255,255,255,.10);
  --accent:#3987e5;
}
*{box-sizing:border-box}
html,body{margin:0;padding:0;background:var(--surface-page)}
body{
  color:var(--ink-strong);
  font-family:system-ui,-apple-system,"Segoe UI","Malgun Gothic",sans-serif;
  font-size:16px; line-height:1.75;
  word-break:keep-all; overflow-wrap:break-word;
  min-height:100vh;
}
h1,h2,h3,h4{text-wrap:balance;letter-spacing:-.01em}
p,li,dd,td{text-wrap:pretty}
.wrap{max-width:980px;margin:0 auto;padding:32px 20px 72px}
.card{background:var(--surface-card);border:1px solid var(--line-edge);
  border-radius:14px;padding:30px 32px;margin-bottom:22px}
a{color:var(--accent);text-decoration:none}
a:hover{text-decoration:underline}
.eyebrow{margin:0 0 8px;color:var(--ink-muted);font-size:13px}
.back{display:inline-block;margin-bottom:18px;font-size:14px;color:var(--ink-muted)}
.back:hover{color:var(--ink-strong)}
code{background:var(--surface-inset);padding:1px 5px;border-radius:4px;
  font-size:.9em;overflow-wrap:break-word}
/* break-all 은 'D1' 같은 두 글자 식별자까지 쪼갠다. 표의 좁은 칸에서 드러난다.
   break-word 는 한 줄에 못 담을 때만 끊고 최소 폭 계산도 망가뜨리지 않는다. */
pre{background:var(--surface-inset);padding:14px 16px;border-radius:8px;
  overflow-x:auto;font-size:13px;line-height:1.6}
pre code{background:none;padding:0}
blockquote{margin:16px 0;padding:14px 18px;background:var(--surface-inset);
  border-radius:10px;color:var(--ink-body)}
blockquote p:last-child{margin-bottom:0}
table{border-collapse:collapse;width:100%;margin:14px 0 20px;font-size:14px;min-width:480px}
.scroll{overflow-x:auto}
th,td{text-align:left;padding:8px 14px 8px 0;border-bottom:1px solid var(--line-edge);
  vertical-align:top}
th{color:var(--ink-muted);font-weight:500;font-size:12.5px;white-space:nowrap}
td{color:var(--ink-body);font-variant-numeric:tabular-nums}
td:first-child{color:var(--ink-strong)}
hr{border:none;border-top:1px solid var(--line-edge);margin:28px 0}
"""

DOC_CSS = TOKENS + """
.doc h1{font-size:29px;line-height:1.34;margin:0 0 22px;padding-bottom:16px;
  border-bottom:2px solid var(--ink-strong)}
.doc h2{font-size:21px;margin:38px 0 10px;padding-top:14px;border-top:1px solid var(--line-edge)}
.doc h3{font-size:16.5px;margin:26px 0 8px}
.doc h4{font-size:15px;margin:20px 0 6px}
.doc p,.doc li{color:var(--ink-body);max-width:68ch}
.doc strong{color:var(--ink-strong)}
.doc ul,.doc ol{padding-left:22px}
.doc li{margin-bottom:6px}
"""

INDEX_CSS = TOKENS + """
h1{font-size:31px;line-height:1.34;margin:0 0 22px;max-width:22ch}
.lead{margin:0 0 24px;display:grid;gap:10px}
.lead>div{display:grid;grid-template-columns:56px 1fr;gap:14px;align-items:baseline}
.lead dt{color:var(--ink-muted);font-size:13px;font-weight:500;margin:0}
.lead dd{margin:0;color:var(--ink-body);max-width:64ch}
.result{display:flex;align-items:baseline;gap:20px;flex-wrap:wrap;
  padding:18px 20px;border-radius:12px;background:var(--surface-inset);margin-bottom:26px}
.num{font-size:50px;font-weight:600;line-height:1;letter-spacing:-.02em}
.numsub{color:var(--ink-body);font-size:14.5px;flex:1 1 320px;margin:0}
/* 타일은 전부 같은 표면을 쓴다. 하나만 반전시키면 "누를 것"이 아니라
   "지금 보고 있는 페이지"로 읽힌다 — 반전 채우기는 활성 상태의 관습이다.
   위계는 색이 아니라 크기·테두리·행동 문구로 준다. */
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:14px;margin:8px 0 4px}
.tile{display:flex;flex-direction:column;padding:20px 22px;
  border:1px solid var(--line-edge);border-radius:12px;
  background:var(--surface-card);color:inherit}
.tile:hover{border-color:var(--accent);text-decoration:none}
.tile .n{color:var(--ink-muted);font-size:12px}
.tile .t{font-size:17px;font-weight:600;margin:4px 0 6px;color:var(--ink-strong)}
.tile .d{font-size:13.5px;color:var(--ink-body);margin:0}
.tile .cta{margin-top:auto;padding-top:12px;color:var(--accent);
  font-size:14px;font-weight:600}
.tile:hover .cta{text-decoration:underline}

/* 주 행동은 한 줄을 통째로 차지한다. 4개가 3+1 로 갈라져 마지막 하나가
   외톨이가 되던 것도 함께 해결된다. */
.tile.primary{grid-column:1 / -1;border:2px solid var(--accent);padding:24px 26px}
.tile.primary .t{font-size:21px}
.tile.primary .d{font-size:14.5px;max-width:62ch}
.tile.primary .cta{font-size:15px}
.meta{color:var(--ink-body);font-size:13.5px;margin:0}
.meta+.meta{margin-top:8px}
"""


def render_markdown(md: str) -> str:
    html = markdown.markdown(
        FRONTMATTER.sub("", md),
        extensions=["tables", "fenced_code", "sane_lists", "attr_list"],
        output_format="html5",
    )
    # 표는 좁은 화면에서 제 상자 안에서만 스크롤되게 한다
    return html.replace("<table>", '<div class="scroll"><table>').replace("</table>", "</table></div>")


def doc_page(md_path: Path, title: str) -> str:
    return f"""<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="robots" content="noindex, nofollow">
<title>{title} — GEO 측정</title>
<style>{DOC_CSS}</style>
</head>
<body>
<div class="wrap">
<a class="back" href="./">&#8592; 처음으로</a>
<article class="card doc">
{render_markdown(md_path.read_text(encoding="utf-8"))}
</article>
</div>
</body>
</html>
"""


def index_page(facts: dict) -> str:
    tiles = "".join(
        f'<a class="tile" href="{href}"><div class="n">{n}</div>'
        f'<div class="t">{title}</div><p class="d">{desc}</p>'
        f'<span class="cta">{cta} &rarr;</span></a>'
        for n, href, title, desc, cta in [
            ("문항 1", "munhang1.html", "좋은 GEO의 기준", "기준 3가지와 근거", "답안 읽기"),
            ("문항 2", "munhang2.html", "측정 도구 PRD", "기능 2개와 우선순위 근거", "답안 읽기"),
            ("부록", "worklog.html", "작업 기록", "무엇을 버렸고 왜", "기록 보기"),
        ]
    )
    return f"""<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="robots" content="noindex, nofollow">
<title>GEO 측정 — 과제 제출물</title>
<meta name="description" content="AI 답변에 브랜드가 얼마나, 어떻게 등장하는지 실제로 측정한 기록과 그에 기반한 답안.">
<style>{INDEX_CSS}</style>
</head>
<body>
<div class="wrap">

<section class="card">
  <p class="eyebrow">GEO 측정 &middot; {facts['model']}</p>
  <h1>AI에게 협업 툴을 물으면 어떤 브랜드가 불리는가</h1>

  <dl class="lead">
    <div><dt>무엇</dt><dd>AI 답변에 브랜드가 얼마나, 어떤 맥락으로 등장하는지를 다루는 분야를
      <strong>GEO</strong>(Generative Engine Optimization, 생성형 엔진 최적화)라고 합니다.
      이 페이지는 과제 답안과, 그 답안의 근거로 삼은 실제 측정을 함께 모은 것입니다.</dd></div>
    <div><dt>문제</dt><dd>AI는 같은 질문에도 매번 다르게 답합니다. 한 번 물어본 결과는 측정이 아닙니다.</dd></div>
    <div><dt>방법</dt><dd>질문 {facts['prompts']}개를 각각 {facts['runs']}번씩, 모두 {facts['calls']}번 물었습니다.
      모든 비율에 신뢰구간을 붙여 어디까지 말할 수 있는지 함께 적었습니다.</dd></div>
  </dl>

  <div class="result">
    <div class="num">{facts['rate']}</div>
    <p class="numsub">측정한 국내 협업 툴 한 곳의 전체 언급률입니다. 다만 이 숫자 하나로는 무엇을
      해야 할지 알 수 없습니다 &mdash; 질문 종류로 나누면 <strong>조건을 붙인 질문에서 {facts['best']},
      경쟁사 대안을 묻는 질문에서 {facts['worst']}</strong>로 갈립니다.</p>
  </div>

  <div class="grid">
    <a class="tile primary" href="report.html">
      <div class="n">실측 리포트</div>
      <div class="t">직접 재본 결과 보기</div>
      <p class="d">브랜드 20개의 언급률과 신뢰구간, 질문 종류별 분해, 그리고 <strong>답변 원문 {facts['calls']}건</strong>이
        페이지 안에 들어 있습니다. 여기부터 보시면 나머지 문서가 무엇을 근거로 쓰였는지 바로 보입니다.</p>
      <span class="cta">리포트 열기 &rarr;</span>
    </a>
    {tiles}
  </div>
</section>

<section class="card">
  <h2 style="margin-top:0;border:none;padding:0;font-size:19px">읽는 순서</h2>
  <p class="meta"><strong>1.</strong> 문항 1 먼저 &mdash; 기준 3가지가 이후 모든 판단의 뼈대입니다.</p>
  <p class="meta"><strong>2.</strong> 실측 리포트 &mdash; 세 번째 기준(<em>효과를 확인할 수 있는가</em>)을
    말로 주장하는 대신 실제로 측정해 보인 것입니다.</p>
  <p class="meta"><strong>3.</strong> 문항 2 PRD &mdash; 위 측정에서 확인한 것을 근거로 기능 두 개를 정의했습니다.</p>
</section>

<section class="card">
  <h2 style="margin-top:0;border:none;padding:0;font-size:19px">측정 조건</h2>
  <div class="scroll"><table>
    <tr><th>모델</th><td>{facts['model']} (단일 엔진)</td></tr>
    <tr><th>질문</th><td>{facts['prompts']}개 &mdash; 구매 여정 5단계 x 한국어·영어 짝</td></tr>
    <tr><th>반복</th><td>질문당 {facts['runs']}회 고정, 조기 종료 없음</td></tr>
    <tr><th>총 호출</th><td>{facts['calls']}회 &middot; 실패 {facts['failed']}건</td></tr>
    <tr><th>샘플링</th><td>기본값 (temperature 등 미지정)</td></tr>
    <tr><th>웹 검색</th><td>끔 &mdash; 모델이 학습으로 아는 것만 분리해 측정</td></tr>
  </table></div>
  <p class="meta">한계도 각 문서의 마지막 절에 적었습니다 &mdash; 엔진 하나, 카테고리 하나, 웹 검색 끔입니다.
    측정 도구와 원자료는 <a href="{REPO_URL}">GitHub 저장소</a>에 있습니다.</p>
</section>

</div>
</body>
</html>
"""


def gather_facts() -> dict:
    sys.path.insert(0, str(ROOT))
    from geolab.brands import BRANDS
    from geolab.measure import aggregate, load_run

    run_dir = sorted((ROOT / "docs" / "measurements").glob("*/run.json"))[-1].parent
    run = load_run(run_dir / "run.json")
    overall = aggregate(run.observations, "overall", BRANDS)
    target = next(r for r in overall if r.brand == "Swit")

    stages = {
        r.scope_value: r.interval
        for r in aggregate(run.observations, "stage", BRANDS)
        if r.brand == "Swit"
    }
    pct = lambda i: f"{i.point * 100:.0f}%"

    return {
        "model": run.models.get(run.engines[0], "?"),
        "prompts": len(run.prompt_ids),
        "runs": run.runs_per_prompt,
        "calls": run.total_calls,
        "failed": run.failed_calls,
        "rate": f"{target.interval.point * 100:.1f}%",
        "best": pct(stages["constrained"]),
        "worst": pct(stages["alternative"]),
        "run_dir": run_dir,
    }


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="답안과 리포트를 한 사이트로 묶어 게시한다")
    parser.add_argument("--dry-run", action="store_true", help="만들기만 하고 push 하지 않는다")
    args = parser.parse_args()

    facts = gather_facts()
    work = Path(tempfile.mkdtemp(prefix="geo-lab-site-"))
    site = work / "site"
    site.mkdir()

    print("1. 페이지 생성")
    (site / "index.html").write_text(index_page(facts), encoding="utf-8")
    for name, out, title, _desc in PAGES:
        src = SUBMISSION / name
        if not src.exists():
            print(f"   건너뜀 (없음): {name}")
            continue
        (site / out).write_text(doc_page(src, title), encoding="utf-8")
        print(f"   {out}")

    report = facts["run_dir"] / "report.html"
    shutil.copy2(report, site / "report.html")
    print(f"   report.html  ({report.stat().st_size / 1024 / 1024:.1f} MB)")

    # Pages 가 밑줄로 시작하는 경로를 Jekyll 로 처리하지 않게 한다
    (site / ".nojekyll").write_text("", encoding="utf-8")

    if args.dry_run:
        print(f"\n--dry-run. 결과: {site}")
        print(f"  미리보기: python -m http.server 8080 --directory \"{site}\"")
        return 0

    print("2. gh-pages 브랜치로 push")
    subprocess.run(["git", "init", "-q", "-b", "gh-pages"], cwd=site, check=True)
    subprocess.run(["git", "add", "-A"], cwd=site, check=True)
    subprocess.run(
        ["git", "commit", "-q", "-m",
         "site: 답안과 실측 리포트를 한 링크로\n\n"
         "scripts/build-site.py 가 생성한다. 정본은 submission/*.md 다.\n\n"
         "Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"],
        cwd=site, check=True,
    )
    subprocess.run(["git", "remote", "add", "origin", REMOTE], cwd=site, check=True)
    subprocess.run(["git", "push", "--force", "origin", "gh-pages"], cwd=site, check=True)

    shutil.rmtree(work, ignore_errors=True)
    print(f"\n올렸다. Pages 설정 후 https://enometa820.github.io/geo-lab/ 에서 열린다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
