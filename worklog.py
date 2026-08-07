"""작업 기록 생성기 — 저장소 상태에서 제출용 문서를 만든다.

    python worklog.py              submission/03-작업-기록.md 를 갱신한다
    python worklog.py --stdout     파일을 쓰지 않고 화면에만 출력한다

## 왜 손으로 쓰지 않는가

작업 기록을 손으로 쓰면 두 가지가 일어난다. 기억에 의존해 실제와 어긋나고,
바빠지면 갱신이 멈춘다. 그러면 **틀린 기록이 남는 것이 없느니만 못한 상태**가
된다.

여기서는 이미 저장소에 있는 사실만 모은다 — 커밋 이력, 판단 기록, 조사 문서의
frontmatter, 측정 결과 파일, 테스트 개수. 지어낼 여지가 없고, 저장소가 바뀌면
다시 돌리기만 하면 된다.

## 무엇을 추상화하는가

커밋 하나하나를 나열하는 것은 로그이지 기록이 아니다. 여기서는 세 가지로 접는다.

  1. 무엇을 어떤 순서로 했나   — 날짜별로 묶은 흐름
  2. **무엇을 버렸나**          — 판단 기록에서 뽑은 폐기 목록
  3. 무엇을 실제로 쟀나         — 측정 이력과 그 조건

2번이 이 문서의 핵심이다. 결과만 보면 안 보이는 것이 거기 있다.

## 표준 라이브러리만 쓴다

`geolab/stats.py` 와 같은 이유다 — 읽는 사람이 설치 없이 돌려볼 수 있어야 한다.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path

ROOT = Path(__file__).parent
OUT_PATH = ROOT / "submission" / "03-작업-기록.md"

DECISIONS_PATH = ROOT / "docs" / "DECISIONS.md"
RESEARCH_DIR = ROOT / "research"
MEASURE_DIR = ROOT / "docs" / "measurements"
TESTS_DIR = ROOT / "tests"

# 커밋 메시지 접두어를 사람이 읽는 말로. 없는 접두어는 그대로 둔다.
COMMIT_KINDS = {
    "feat": "기능",
    "fix": "수정",
    "docs": "문서",
    "chore": "정리",
    "test": "테스트",
    "refactor": "구조 변경",
}


@dataclass(frozen=True)
class Commit:
    date: str
    subject: str

    @property
    def kind(self) -> str:
        head = self.subject.split(":", 1)[0].strip()
        return COMMIT_KINDS.get(head, "작업")

    @property
    def body(self) -> str:
        return self.subject.split(":", 1)[1].strip() if ":" in self.subject else self.subject


@dataclass(frozen=True)
class Decision:
    id: str
    title: str
    discarded: str | None


@dataclass(frozen=True)
class ResearchDoc:
    filename: str
    title: str
    sources: str
    verified_at: str


@dataclass(frozen=True)
class Measurement:
    started_at: str
    models: dict[str, str]
    runs_per_prompt: int
    total_calls: int
    failed_calls: int
    prompt_count: int


def _run(cmd: list[str], *, check: bool = False, timeout: int | None = None) -> str:
    """외부 명령을 돌리고 표준 출력을 돌려준다. 실패하면 빈 문자열.

    `errors="replace"` 를 쓰는 이유 — 윈도우 콘솔은 UTF-8 이 아닐 수 있고,
    도구가 뱉는 바이트를 우리가 통제하지 못한다. 여기서 예외가 나면 **문서
    생성 전체가 멈춘다.** 몇 글자가 깨지는 것보다 그게 나쁘다.
    """
    try:
        result = subprocess.run(
            cmd, cwd=ROOT, capture_output=True, text=True,
            encoding="utf-8", errors="replace", check=check, timeout=timeout,
        )
        return result.stdout or ""
    except (subprocess.SubprocessError, OSError):
        return ""


def _run_git(*args: str) -> str:
    return _run(["git", *args], check=True)


def collect_commits() -> list[Commit]:
    raw = _run_git("log", "--pretty=format:%ad\t%s", "--date=short", "--reverse")
    commits = []
    for line in raw.splitlines():
        if "\t" not in line:
            continue
        date, subject = line.split("\t", 1)
        commits.append(Commit(date=date.strip(), subject=subject.strip()))
    return commits


def collect_decisions() -> list[Decision]:
    """판단 기록에서 제목과 '버린 것'을 뽑는다.

    본문은 옮기지 않는다 — 같은 사실을 두 곳에 쓰면 한쪽이 반드시 낡는다.
    이 문서는 목록만 보여주고 상세는 원본을 가리킨다.
    """
    if not DECISIONS_PATH.exists():
        return []

    text = DECISIONS_PATH.read_text(encoding="utf-8")
    blocks = re.split(r"^### (D\d+)\. ", text, flags=re.M)

    decisions = []
    # blocks = [머리말, id, 본문, id, 본문, ...]
    for i in range(1, len(blocks) - 1, 2):
        ident, body = blocks[i], blocks[i + 1]
        title = body.splitlines()[0].strip()

        match = re.search(r"\*\*버린 것\*\*\s*[—-]\s*(.+)", body)
        discarded = match.group(1).strip() if match else None
        if discarded:
            # 표 안에서는 강조 마크업이 노이즈다. 굵게·기울임을 벗기고 평문으로.
            discarded = re.sub(r"\*\*(.+?)\*\*", r"\1", discarded)
            discarded = re.sub(r"\*(.+?)\*", r"\1", discarded)
            discarded = discarded.replace("|", "·").rstrip(".")

        decisions.append(Decision(id=ident, title=title, discarded=discarded))

    return decisions


def _frontmatter(text: str) -> dict[str, str]:
    if not text.startswith("---"):
        return {}
    end = text.find("---", 3)
    if end == -1:
        return {}
    fields = {}
    for line in text[3:end].splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            fields[key.strip()] = value.strip()
    return fields


def collect_research() -> list[ResearchDoc]:
    docs = []
    for path in sorted(RESEARCH_DIR.glob("*.md")) if RESEARCH_DIR.exists() else []:
        text = path.read_text(encoding="utf-8")
        meta = _frontmatter(text)
        heading = next((l[2:].strip() for l in text.splitlines() if l.startswith("# ")), path.stem)
        docs.append(
            ResearchDoc(
                filename=path.name,
                title=heading,
                sources=meta.get("sources", "-"),
                verified_at=meta.get("verified_at", "-"),
            )
        )
    return docs


def collect_measurements() -> list[Measurement]:
    results = []
    for path in sorted(MEASURE_DIR.glob("*/run.json")) if MEASURE_DIR.exists() else []:
        try:
            meta = json.loads(path.read_text(encoding="utf-8"))["meta"]
        except (json.JSONDecodeError, KeyError):
            continue
        results.append(
            Measurement(
                started_at=meta.get("started_at", "-")[:16].replace("T", " "),
                models=meta.get("models", {}),
                runs_per_prompt=meta.get("runs_per_prompt", 0),
                total_calls=meta.get("total_calls", 0),
                failed_calls=meta.get("failed_calls", 0),
                prompt_count=len(meta.get("prompt_ids", [])),
            )
        )
    return results


def count_tests() -> tuple[int, str]:
    """테스트 개수를 센다. `(개수, 세는 방법)` 을 돌려준다.

    `def test_` 를 세면 실제 케이스 수보다 적게 나온다 — 하나의 함수가
    파라미터화로 여러 케이스를 만들기 때문이다. 그래서 수집기를 먼저 쓰고,
    쓸 수 없을 때만 정규식으로 내려간다. **어느 쪽으로 셌는지 함께 표시한다.**
    """
    if not TESTS_DIR.exists():
        return 0, "없음"

    collected = _run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q", str(TESTS_DIR)],
        timeout=120,
    )
    match = re.search(r"(\d+)\s+tests?\s+collected", collected)
    if match:
        return int(match.group(1)), "수집된 케이스"

    functions = sum(
        len(re.findall(r"^def test_", p.read_text(encoding="utf-8"), flags=re.M))
        for p in TESTS_DIR.glob("test_*.py")
    )
    return functions, "테스트 함수 (파라미터화 케이스는 이보다 많다)"


def build() -> str:
    commits = collect_commits()
    decisions = collect_decisions()
    research = collect_research()
    measurements = collect_measurements()
    tests, test_basis = count_tests()

    discarded = [d for d in decisions if d.discarded]
    span = f"{commits[0].date} ~ {commits[-1].date}" if commits else "-"

    # 프롬프트 몇 개짜리는 연결 확인용이다. 본 측정과 섞어 보이면 오해를 부른다.
    full_runs = [m for m in measurements if m.prompt_count > 4]
    smoke_runs = [m for m in measurements if m.prompt_count <= 4]

    out: list[str] = []
    w = out.append

    w("---")
    w("type: submission")
    w(f"updated: {date.today().isoformat()}")
    w("derived_from: docs/DECISIONS.md, research/, docs/measurements/")
    w("---")
    w("")
    w("# 작업 기록")
    w("")
    w("> **이 문서는 저장소 상태에서 자동 생성됩니다.** 직접 고치지 마세요 — 다시 생성하면 사라집니다.")
    w("> 내용을 바꾸려면 원천(판단 기록·조사 문서·측정 결과)을 고치고 `python worklog.py` 를 다시 실행합니다.")
    w("")
    w("결과만 보면 보이지 않는 것을 남기기 위한 문서입니다. **무엇을 검토했고 무엇을 왜 버렸는지**가 핵심입니다.")
    w("")
    w("---")
    w("")

    # --- 한눈에 ---
    w("## 한눈에")
    w("")
    w("| | |")
    w("|---|---|")
    w(f"| 작업 기간 | {span} |")
    w(f"| 커밋 | {len(commits)}건 |")
    w(f"| 조사 문서 | {len(research)}편 |")
    w(f"| 기록된 판단 | {len(decisions)}건 — **모두 무엇을 버렸는지 함께 적혀 있습니다** |")
    w(f"| 측정 실행 | 본 측정 {len(full_runs)}회 · 연결 확인 {len(smoke_runs)}회 |")
    w(f"| 자동 테스트 | {tests}개 ({test_basis}) |")
    w("")

    # --- 흐름 ---
    w("## 어떤 순서로 했나")
    w("")
    if commits:
        by_date: dict[str, list[Commit]] = {}
        for c in commits:
            by_date.setdefault(c.date, []).append(c)
        for day, items in by_date.items():
            w(f"**{day}**")
            w("")
            for c in items:
                w(f"- {c.kind} — {c.body}")
            w("")
    else:
        w("커밋 이력을 읽지 못했습니다.")
        w("")

    # --- 버린 것 ---
    w("## 무엇을 버렸나")
    w("")
    w("조사와 구현 중에 검토했다가 접은 것들입니다. **접은 이유가 이 저장소의 판단 기준을 보여줍니다.**")
    w("")
    if discarded:
        w("| | 무엇을 버렸나 | 어떤 판단에서 |")
        w("|---|---|---|")
        for d in discarded:
            w(f"| `{d.id}` | {d.discarded} | {d.title} |")
        w("")
        w("각 항목의 상세한 이유는 [`docs/DECISIONS.md`](../docs/DECISIONS.md)에 있습니다.")
    else:
        w("아직 기록된 폐기 판단이 없습니다.")
    w("")

    # --- 조사 ---
    w("## 조사 자산")
    w("")
    if research:
        w("| 문서 | 교차검증 출처 | 내용 확인일 |")
        w("|---|---|---|")
        for r in research:
            w(f"| [{r.title}](../research/{r.filename}) | {r.sources} | {r.verified_at} |")
        w("")
        w("`내용 확인일`은 파일을 고친 날이 아니라 **내용이 아직 맞다고 실제로 확인한 날**입니다.")
    else:
        w("조사 문서가 없습니다.")
    w("")

    # --- 측정 ---
    w("## 무엇을 실제로 쟀나")
    w("")
    if full_runs:
        w("| 측정 시각 (UTC) | 대상 모델 | 프롬프트 | 반복 | 총 호출 | 실패(표본 제외) |")
        w("|---|---|---|---|---|---|")
        for m in full_runs:
            models = ", ".join(f"{v}" for v in m.models.values()) or "-"
            rate = f"{m.failed_calls}회" if not m.failed_calls else f"{m.failed_calls}회 ({m.failed_calls / m.total_calls:.1%})"
            w(
                f"| {m.started_at} | {models} | {m.prompt_count}개 | {m.runs_per_prompt}회 "
                f"| {m.total_calls:,}회 | {rate} |"
            )
        w("")
        w("**실패한 호출은 표본에서 제외합니다.** 실패를 '언급 안 됨'으로 세면 언급률이 아래로 편향됩니다.")
        w("")
        w("측정 조건을 함께 적는 이유는 **조건 없는 수치는 재현할 수 없어 근거가 아니기** 때문입니다.")
        if smoke_runs:
            w("")
            w(f"위 표에는 연결 확인용 소규모 실행 {len(smoke_runs)}회를 넣지 않았습니다.")
    else:
        w("아직 본 측정을 실행하지 않았습니다.")
        if smoke_runs:
            w("")
            w(f"연결 확인용 소규모 실행 {len(smoke_runs)}회만 있습니다.")
    w("")

    # --- 한계 ---
    w("## 이 기록의 한계")
    w("")
    w("- **커밋 단위로 묶은 흐름입니다.** 커밋 사이에 있었던 시행착오는 드러나지 않습니다")
    w("- **판단 기록에 남긴 것만 보입니다.** 기록하지 않고 지나간 판단은 여기 없습니다")
    w("- **측정 이력은 저장된 실행만 셉니다.** 저장하지 않고 돌린 것은 여기 없습니다")
    w("- **연결 확인용 소규모 실행은 본 측정 표에서 제외했습니다.** 섞으면 표본 규모를 오해하게 됩니다")
    w("")

    return "\n".join(out) + "\n"


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="저장소 상태에서 제출용 작업 기록을 생성한다")
    parser.add_argument("--stdout", action="store_true", help="파일을 쓰지 않고 화면에 출력한다")
    args = parser.parse_args()

    content = build()

    if args.stdout:
        print(content)
        return 0

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(content, encoding="utf-8")
    print(f"생성했습니다: {OUT_PATH.relative_to(ROOT)}  ({len(content.splitlines())}줄)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
