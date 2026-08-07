"""제출용 PDF 를 만든다 — 마크다운 답안을 리포트와 같은 디자인으로 인쇄한다.

    python scripts/build-submission.py

## 왜 별도 스크립트인가

답안은 마크다운이 정본이다. 고칠 때 마크다운을 고치고 여기서 다시 뽑는다.
PDF 를 손으로 만들면 정본이 둘이 되고, 둘은 반드시 갈라진다.

## 디자인을 새로 만들지 않는다

`report.py` 가 쓰는 것과 같은 토큰과 활자를 쓴다. 같은 봉투에 들어가는 문서가
서로 다른 얼굴을 하고 있으면 읽는 사람이 두 번 적응해야 한다.

인쇄본이므로 라이트 모드 고정이다. 화면용 다크 모드는 종이에서 의미가 없다.

## 이 스크립트가 루트에 없는 이유

루트에는 진입점만 둔다 — `run.py`(측정) · `report.py`(리포트) · `worklog.py`(기록).
이건 제출물 포장용 보조 도구라 `scripts/` 에 산다.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
from pathlib import Path

import markdown

ROOT = Path(__file__).resolve().parent.parent
SUBMISSION = ROOT / "submission"
RENDERER = Path.home() / ".claude" / "skills" / "html-to-pdf" / "scripts" / "render.js"

# 순서가 곧 읽는 순서다
DOCS = [
    ("README.md", "제출 안내"),
    ("01-문항1-좋은-GEO의-기준.md", "문항 1"),
    ("02-문항2-PRD.md", "문항 2"),
    ("03-작업-기록.md", "작업 기록"),
]

FRONTMATTER = re.compile(r"\A---\n.*?\n---\n", re.S)

CSS = """
:root{
  --surface-page:#ffffff;
  --surface-card:#ffffff;
  --surface-inset:#f1f0ec;
  --ink-strong:#0b0b0b;
  --ink-body:#3d3c3a;
  --ink-muted:#767570;
  --line-edge:rgba(11,11,11,.12);
  --accent:#1c5cab;
}
*{box-sizing:border-box}
html,body{margin:0;padding:0;background:var(--surface-page)}
body{
  color:var(--ink-strong);
  font-family:system-ui,-apple-system,"Segoe UI","Malgun Gothic",sans-serif;
  font-size:10.5pt; line-height:1.72;
  /* 한국어는 기본값이 글자 단위로 끊어 단어를 자른다 */
  word-break:keep-all; overflow-wrap:break-word;
}
.doc{max-width:172mm;margin:0 auto;padding:14mm 0}
h1,h2,h3,h4{text-wrap:balance;letter-spacing:-.01em}
p,li,td{text-wrap:pretty}
h1{font-size:20pt;line-height:1.32;margin:0 0 6mm;padding-bottom:4mm;
   border-bottom:2px solid var(--ink-strong)}
h2{font-size:14pt;margin:9mm 0 3mm;padding-top:3mm;border-top:1px solid var(--line-edge)}
h3{font-size:11.5pt;margin:6mm 0 2mm;color:var(--ink-strong)}
h4{font-size:10.5pt;margin:5mm 0 2mm}
p{margin:0 0 3.2mm;color:var(--ink-body)}
strong{color:var(--ink-strong)}
ul,ol{margin:0 0 3.2mm;padding-left:6mm}
li{margin-bottom:1.6mm;color:var(--ink-body)}
blockquote{margin:4mm 0;padding:3.5mm 5mm;background:var(--surface-inset);
  border-radius:2mm;color:var(--ink-body)}
blockquote p:last-child{margin-bottom:0}
table{border-collapse:collapse;width:100%;margin:3.5mm 0 5mm;font-size:9.3pt}
th,td{text-align:left;padding:2.2mm 3mm 2.2mm 0;border-bottom:1px solid var(--line-edge);
  vertical-align:top}
th{color:var(--ink-muted);font-weight:600;font-size:8.6pt;
   border-bottom:1px solid var(--ink-muted)}
td{color:var(--ink-body);font-variant-numeric:tabular-nums}
td:first-child{color:var(--ink-strong)}
code{background:var(--surface-inset);padding:.4mm 1.2mm;border-radius:1mm;
  font-size:.92em;overflow-wrap:break-word}
pre{background:var(--surface-inset);padding:3.5mm 4mm;border-radius:2mm;
  overflow-x:auto;font-size:8.8pt;line-height:1.55}
pre code{background:none;padding:0}
hr{border:none;border-top:1px solid var(--line-edge);margin:7mm 0}
a{color:var(--accent);text-decoration:none;word-break:break-all}
.docfoot{margin-top:10mm;padding-top:4mm;border-top:1px solid var(--line-edge);
  color:var(--ink-muted);font-size:8.6pt}

/* 인쇄 — 통짜 섹션에 break-inside:avoid 를 걸면 빈 페이지가 생긴다.
   표의 행이나 목록 항목 같은 작은 단위만 안 쪼개지게 한다. */
@media print{
  @page{ margin:14mm 16mm; }
  html,body{ background:#fff !important; }
  .doc{ max-width:100%; padding:0; }
  tr,li,blockquote{ break-inside:avoid; }
  h1,h2,h3,h4{ break-after:avoid; }
  table,pre{ break-inside:auto; }
}
"""


def to_html(md_path: Path, label: str) -> str:
    raw = FRONTMATTER.sub("", md_path.read_text(encoding="utf-8"))
    body = markdown.markdown(
        raw,
        extensions=["tables", "fenced_code", "sane_lists", "attr_list"],
        output_format="html5",
    )
    title = re.search(r"<h1[^>]*>(.*?)</h1>", body)
    heading = re.sub(r"<[^>]+>", "", title.group(1)) if title else label
    return f"""<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8">
<title>{heading}</title>
<style>{CSS}</style>
</head>
<body>
<div class="doc">
{body}
<p class="docfoot">스퀘어스 과제 제출물 &middot; {label} &middot; 측정 도구 https://github.com/enometa820/geo-lab</p>
</div>
</body>
</html>
"""


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")

    if not RENDERER.exists():
        print(f"렌더러를 찾을 수 없다: {RENDERER}", file=sys.stderr)
        return 1

    work = ROOT / ".build-submission"
    work.mkdir(exist_ok=True)
    made = []

    try:
        for name, label in DOCS:
            src = SUBMISSION / name
            if not src.exists():
                print(f"  건너뜀 (없음): {name}")
                continue

            html_path = work / (src.stem + ".html")
            html_path.write_text(to_html(src, label), encoding="utf-8")

            result = subprocess.run(
                ["node", str(RENDERER), str(html_path), "portrait"],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            if result.returncode != 0:
                print(f"  실패: {name}\n{result.stdout}\n{result.stderr}", file=sys.stderr)
                return 1

            pdf = html_path.with_suffix(".pdf")
            target = SUBMISSION / (src.stem + ".pdf")
            shutil.move(str(pdf), str(target))
            made.append(target)

            box = re.search(r"MediaBox=\[[^\]]*\]", result.stdout or "")
            print(f"  {target.name}  {box.group(0) if box else ''}")
    finally:
        shutil.rmtree(work, ignore_errors=True)

    copied = copy_latest_report()
    if copied:
        print(f"  {copied.name}  (최신 측정에서 복사)")

    print(f"\n제출물 {len(made) + (1 if copied else 0)}개를 만들었다. 위치: {SUBMISSION}")
    return 0


def copy_latest_report() -> Path | None:
    """가장 최근 측정의 리포트를 제출 폴더로 복사한다.

    정본은 측정 폴더 안의 것이다 — 결과와 그 재료가 같이 있어야 하기 때문이다.
    여기 있는 것은 **메일에 붙이기 위한 사본**이고, 이 스크립트를 돌릴 때마다
    다시 만들어진다. 손으로 고치면 다음 실행에서 덮어쓴다.
    """
    runs = sorted((ROOT / "docs" / "measurements").glob("*/report.html"))
    if not runs:
        return None
    target = SUBMISSION / "report.html"
    shutil.copy2(runs[-1], target)
    return target


if __name__ == "__main__":
    raise SystemExit(main())
