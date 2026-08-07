"""측정 실행 진입점.

    python run.py --dry-run              무엇을 몇 번 호출할지만 계산한다
    python run.py --smoke                프롬프트 2개 x 2회. 연결 확인용
    python run.py --runs 20              실제 측정
    python run.py --report <경로>        저장된 측정을 다시 집계한다 (재호출 없음)

## 출력에 한계를 항상 붙이는 이유

이 저장소의 규격이 "한계를 먼저 적는다"이므로, 리포트를 만드는 코드가 그걸
집행한다. 사람이 기억해서 적는 것에 맡기면 바쁠 때 빠진다.
"""

from __future__ import annotations

import argparse
import sys
import unicodedata
from datetime import datetime
from pathlib import Path

from geolab.brands import BRANDS
from geolab.engines import available_engines
from geolab.measure import (
    MeasurementRun,
    aggregate,
    build_engines,
    language_gap,
    load_run,
    measure,
    save_run,
)
from geolab.prompts import PROMPTS, STAGE_DESCRIPTION
from geolab.stats import wilson_interval

OUT_ROOT = Path("docs/measurements")


def _width(text: str) -> int:
    """한글·한자는 터미널에서 두 칸을 차지한다. 표를 맞추려면 세어야 한다."""
    return sum(2 if unicodedata.east_asian_width(c) in ("W", "F") else 1 for c in text)


def _pad(text: str, width: int) -> str:
    return text + " " * max(0, width - _width(text))


def _hr(char: str = "-", width: int = 72) -> str:
    return char * width


def print_conditions(engines: list[str], models: dict[str, str], runs: int, prompt_count: int) -> None:
    """측정 조건. 이게 없으면 결과는 재현 불가라 근거가 아니다."""
    total = len(engines) * prompt_count * runs
    print(_hr("="))
    print("측정 조건")
    print(_hr("="))
    for name in engines:
        print(f"  엔진        {name} / {models.get(name, '?')}")
    print(f"  프롬프트    {prompt_count}개")
    print(f"  반복        프롬프트당 {runs}회 (고정. 조기 종료 없음)")
    print(f"  총 호출     {total:,}회")
    print(f"  샘플링      기본 설정 그대로 (temperature 등 미지정)")
    print(f"  웹 검색     끔 — '모델이 브랜드를 아는가'만 분리해 잰다")
    print()


def print_expected_precision(runs: int, prompt_count: int, engine_count: int) -> None:
    """이 표본으로 무엇을 주장할 수 있는지 미리 보여준다.

    측정을 다 돌리고 나서 "구간이 너무 넓어 아무 말도 못 한다"를 알게 되면
    비용을 이미 쓴 뒤다.
    """
    print(_hr())
    print("이 표본으로 기대할 수 있는 정밀도 (p=0.5 기준, 가장 넓은 경우)")
    print(_hr())

    per_lang = prompt_count // 2 * runs
    rows = [
        ("프롬프트별 (동질)", runs),
        ("언어별 (혼합)", per_lang),
        ("전체 (혼합)", prompt_count * runs),
    ]
    for label, n in rows:
        if n <= 0:
            continue
        i = wilson_interval(round(n * 0.5), n)
        print(f"  {_pad(label, 22)} n={n:>5}   95% CI 폭 {(i.high - i.low) * 100:5.1f}pp")

    zero = wilson_interval(0, runs)
    print()
    print(f"  * 프롬프트별로 한 번도 언급되지 않으면: 0% ~ {zero.high * 100:.1f}%")
    print(f"  * '혼합'은 프롬프트를 합친 것이라 구간이 실제 불확실성을 과소평가할 수 있다")
    print()


def print_rates(run: MeasurementRun, scope: str, top: int = 12) -> None:
    """엔진별로 나눠 출력한다. 합산하지 않는다."""
    rates = aggregate(run.observations, scope, BRANDS)

    for engine in run.engines:
        subset = [r for r in rates if r.engine == engine and r.interval.n > 0]
        subset = [r for r in subset if r.interval.point > 0][:top]

        print(_hr())
        print(f"{engine} / {run.models.get(engine, '?')} — 언급률 ({scope})")
        print(_hr())

        if not subset:
            print("  언급된 브랜드 없음. 또는 표본이 전부 실패했다")
            print()
            continue

        for r in subset:
            label = r.brand if scope == "overall" else f"{r.brand} [{r.scope_value}]"
            marker = "" if r.homogeneous else " ~"
            note = f"  실패 {r.n_failed} 제외" if r.n_failed else ""
            print(f"  {_pad(label, 26)} {r.interval.format_pct()}{marker}{note}")

        print()
        print("  ~ = 여러 프롬프트를 합친 값. 구간이 실제보다 좁을 수 있다")
        print()


def print_language_gap(run: MeasurementRun, top: int = 8) -> None:
    gaps = language_gap(run.observations, BRANDS)
    shown = [g for g in gaps if g[2].interval.n > 0 and g[3].interval.n > 0][:top]

    if not shown:
        return

    print(_hr())
    print("한국어 / 영어 차이")
    print(_hr())
    print(f"  {_pad('브랜드', 20)} {_pad('엔진', 12)} {_pad('한국어', 12)} {_pad('영어', 12)} 판정")

    for brand, engine, ko, en, overlapping in shown:
        verdict = "구간 겹침 — 차이 주장 불가" if overlapping else "구간 분리 — 차이 있음"
        print(
            f"  {_pad(brand, 20)} {_pad(engine, 12)}"
            f" {_pad(f'{ko.interval.point * 100:.0f}%', 12)}"
            f" {_pad(f'{en.interval.point * 100:.0f}%', 12)} {verdict}"
        )

    print()
    print("  * 겹치지 않을 때만 차이를 주장할 수 있다. 겹친다고 차이가 없다는 뜻은 아니다")
    print()


def print_limits(run: MeasurementRun) -> None:
    """무엇을 못 쟀는지. 이 절은 생략하지 않는다."""
    print(_hr("="))
    print("이 측정의 한계")
    print(_hr("="))

    failed = run.failed_calls
    if failed:
        print(f"  실패한 호출 {failed}/{run.total_calls}건을 표본에서 제외했다")
        by_prompt = run.failures_by_prompt()
        worst = sorted(by_prompt.items(), key=lambda kv: -kv[1])[:5]
        print(f"    프롬프트별: {', '.join(f'{k}={v}' for k, v in worst)}")
        if len(by_prompt) <= 2 and failed > 3:
            print("    ** 실패가 소수 프롬프트에 몰려 있다. 제외가 무작위가 아닐 수 있다 **")
    else:
        print("  실패한 호출 없음")

    if run.excluded_aliases:
        print()
        print("  일반 명사와 겹쳐 일부러 탐지하지 않은 표기 (해당 브랜드는 과소 집계될 수 있다):")
        for canonical, aliases in run.excluded_aliases.items():
            print(f"    {_pad(canonical, 20)} {', '.join(aliases)}")

    print()
    print("  웹 검색을 끄고 측정했다. 학습된 지식만 잰 것이며 실시간 검색 결과는 포함되지 않는다")
    print("  프롬프트 세트는 팀 협업 툴 카테고리에 한정된다. 다른 카테고리로 일반화할 수 없다")
    print()


def report(run: MeasurementRun, scope: str) -> None:
    print()
    print_rates(run, scope)
    print_language_gap(run)
    print_limits(run)


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="AI 답변에서의 브랜드 언급률을 신뢰구간과 함께 측정한다")
    parser.add_argument("--runs", type=int, default=20, help="프롬프트당 반복 횟수 (기본 20)")
    parser.add_argument("--engines", type=str, default=None, help="쉼표로 구분. 기본은 키가 있는 것 전부")
    parser.add_argument("--workers", type=int, default=4, help="동시 호출 수 (기본 4)")
    parser.add_argument("--scope", default="overall", choices=["overall", "prompt", "stage", "language"])
    parser.add_argument("--out", type=str, default=None, help="결과 저장 디렉터리")
    parser.add_argument("--dry-run", action="store_true", help="호출하지 않고 계획만 출력")
    parser.add_argument("--smoke", action="store_true", help="프롬프트 2개 x 2회. 연결 확인용")
    parser.add_argument("--report", type=str, default=None, help="저장된 측정을 다시 집계한다")
    args = parser.parse_args()

    # 재집계 — 호출하지 않는다
    if args.report:
        run = load_run(args.report)
        print(f"\n저장된 측정을 다시 집계한다: {args.report}")
        print(f"  측정 시각 {run.started_at} ~ {run.finished_at}")
        print(f"  반복 {run.runs_per_prompt}회 / 총 {run.total_calls:,}호출")
        report(run, args.scope)
        return 0

    names = args.engines.split(",") if args.engines else available_engines()
    if not names:
        print("사용 가능한 엔진이 없다. ANTHROPIC_API_KEY / OPENAI_API_KEY 를 확인한다.", file=sys.stderr)
        print("환경변수는 새 셸에서만 반영된다.", file=sys.stderr)
        return 1

    prompts = PROMPTS[:2] if args.smoke else PROMPTS
    runs = 2 if args.smoke else args.runs

    if args.dry_run:
        print_conditions(names, {n: "(미확인 — 실행 시 결정)" for n in names}, runs, len(prompts))
        print_expected_precision(runs, len(prompts), len(names))
        print("실제 호출은 하지 않았다. --dry-run 을 빼면 측정을 시작한다.")
        print("** 실행하면 API 비용이 발생한다. 위 호출 수를 확인한다. **")
        return 0

    try:
        engines = build_engines(names)
    except (RuntimeError, ValueError) as exc:
        print(f"엔진을 만들 수 없다: {exc}", file=sys.stderr)
        return 1

    print_conditions(names, {e.name: e.model for e in engines}, runs, len(prompts))
    if not args.smoke:
        print_expected_precision(runs, len(prompts), len(names))

    stamp = datetime.now().strftime("%Y-%m-%d-%H%M")
    out_dir = Path(args.out) if args.out else OUT_ROOT / stamp
    out_dir.mkdir(parents=True, exist_ok=True)

    def progress(done: int, total: int) -> None:
        bar = int(done / total * 30)
        print(f"\r  [{'#' * bar}{'.' * (30 - bar)}] {done}/{total}", end="", flush=True)

    run = measure(
        engines,
        runs=runs,
        prompts=prompts,
        workers=args.workers,
        on_progress=progress,
        raw_path=out_dir / "raw.jsonl",
    )
    print()

    save_run(out_dir / "run.json", run)
    report(run, args.scope)

    print(_hr("="))
    print(f"원 응답  {out_dir / 'raw.jsonl'}")
    print(f"관측     {out_dir / 'run.json'}")
    print(f"재집계   python run.py --report {out_dir / 'run.json'} --scope language")
    print(_hr("="))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
