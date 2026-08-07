"""측정 루프 — 반복 호출하고, 실패를 걸러내고, 신뢰구간과 함께 집계한다.

이 모듈이 이 저장소의 주장을 실제로 집행하는 자리다. 규칙은 넷이다.

1. **실패한 호출은 표본에서 뺀다.** 호출 실패를 "언급 안 됨"으로 세면 언급률이
   아래로 편향된다 — 조용히 틀리는 전형적인 방식이다.
2. **원 응답을 전부 저장한다.** 인용 가능한 증거이자 재분석의 재료다. 집계만
   남기면 나중에 다른 질문을 못 던진다.
3. **엔진을 합산하지 않는다.** 플랫폼마다 인용 행동이 달라 합산 점수는 구성비에
   좌우된다. 그래서 이 모듈은 합산 API 를 아예 제공하지 않는다 (§ aggregate).
4. **반복 수를 미리 정하고 그만큼 채운다.** 조기 종료를 넣지 않는다 (§ measure).

## 왜 조기 종료가 없나

신뢰구간 폭이 목표치 아래로 내려가면 멈추는 것이 자연스러워 보이지만, 그렇게
하면 안 된다. arXiv 2603.08924 이 실측한 바로 **CI 폭은 표본을 늘려도 단조
감소하지 않는다** — 좁아졌다 다시 넓어진다. 우연히 좁아진 지점에서 멈추면
불확실성을 실제보다 작게 보고하게 된다.

그래서 반복 수는 호출자가 미리 정하고, 이 모듈은 그만큼 채우기만 한다.

## 집계 단위에 관하여

Wilson 구간은 **같은 분포에서 나온 독립 시행**을 가정한다. 그런데 프롬프트마다
브랜드가 언급될 확률이 다르므로, 프롬프트 여러 개를 합친 비율은 그 가정을
어긴다.

합쳐서 보는 것 자체가 틀린 것은 아니다 — "이 프롬프트 세트 전체에서 몇 퍼센트
나오나"는 실무적으로 의미가 있다. 다만 **그 수치는 프롬프트 세트에 조건부이고,
구간 폭이 실제 불확실성을 과소평가할 수 있다.** 그래서 `Rate.homogeneous` 가
이 구분을 값에 달고 다닌다. 리포트는 이 플래그를 반드시 표시해야 한다.
"""

from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable, Literal, Sequence

from .brands import BRANDS, Brand, detect_mentions, excluded_aliases
from .engines import Answer, Engine
from .prompts import PROMPTS, Prompt
from .stats import Interval, wilson_interval

ScopeKind = Literal["prompt", "stage", "language", "overall"]

# 레이트리밋 때문에 보수적으로 잡는다. 측정이 빨리 끝나는 것보다 끝까지 도는 것이 중요하다.
DEFAULT_WORKERS = 4


@dataclass(frozen=True)
class Observation:
    """호출 1회의 결과.

    (엔진, 프롬프트, 반복 회차) 하나가 관측 하나다. `mentions` 는 그 응답에서
    탐지된 브랜드의 canonical 이름들이고, 한 응답에 같은 브랜드가 여러 번 나와도
    한 번으로 센다 — 재는 것이 "등장했는가"라는 베르누이 시행이기 때문이다.

    `ok=False` 면 집계에서 제외된다. 지우지 않고 남기는 이유는 **얼마나 제외됐고
    어디서 제외됐는지를 리포트에 적어야** 하기 때문이다.
    """

    engine: str
    model: str
    prompt_id: str
    stage: str
    language: str
    pair_id: str
    run_index: int
    mentions: tuple[str, ...]
    ok: bool
    error: str | None = None

    def scope_value(self, kind: ScopeKind) -> str:
        if kind == "prompt":
            return self.prompt_id
        if kind == "stage":
            return self.stage
        if kind == "language":
            return self.language
        return "all"


@dataclass(frozen=True)
class Rate:
    """언급률 하나.

    **무엇에 대한 비율인지를 값이 스스로 들고 다닌다.** brand 와 engine 만 있는
    수치는 해석할 수 없다 — 어느 프롬프트 집합에서, 몇 번 시도해서, 몇 번 실패한
    표본인지가 함께 있어야 한다.

    n_attempted   시도한 호출 수 (실패 포함)
    n_failed      실패해서 표본에서 제외된 수
    homogeneous   단일 프롬프트 기반인가. False 면 이질적 시행을 섞은 것이라
                  구간 폭이 실제 불확실성을 과소평가할 수 있다
    """

    brand: str
    engine: str
    model: str
    scope_kind: ScopeKind
    scope_value: str
    interval: Interval
    n_attempted: int
    n_failed: int
    homogeneous: bool

    @property
    def failure_rate(self) -> float:
        if self.n_attempted == 0:
            return 0.0
        return self.n_failed / self.n_attempted

    def describe(self) -> str:
        """사람이 읽을 한 줄. 조건을 빼놓지 않는다."""
        base = f"{self.brand} / {self.engine}: {self.interval.format_pct()}"
        if not self.homogeneous:
            base += " [혼합]"
        if self.n_failed:
            base += f" [실패 {self.n_failed}/{self.n_attempted} 제외]"
        return base


@dataclass(frozen=True)
class MeasurementRun:
    """측정 1회분 전체. 결과와 그 결과를 낳은 조건을 같이 든다.

    측정 조건 없는 수치는 재현 불가라 근거가 아니다. 그래서 무엇을 언제 어떻게
    쟀는지가 결과와 같은 객체에 들어 있다.
    """

    started_at: str
    finished_at: str
    runs_per_prompt: int
    engines: tuple[str, ...]
    models: dict[str, str]
    prompt_ids: tuple[str, ...]
    brand_count: int
    observations: tuple[Observation, ...]
    excluded_aliases: dict[str, tuple[str, ...]]
    web_search_enabled: bool = False

    @property
    def total_calls(self) -> int:
        return len(self.observations)

    @property
    def failed_calls(self) -> int:
        return sum(1 for o in self.observations if not o.ok)

    def failures_by_prompt(self) -> dict[str, int]:
        """어느 프롬프트에서 실패가 났나.

        실패가 특정 프롬프트에 몰려 있으면 **제외가 무작위가 아니다.** 그 경우
        표본에서 빼는 것 자체가 편향을 만들 수 있으므로 리포트에 드러내야 한다.
        """
        counts: dict[str, int] = {}
        for o in self.observations:
            if not o.ok:
                counts[o.prompt_id] = counts.get(o.prompt_id, 0) + 1
        return counts


def observations_from_answers(
    answers: Iterable[Answer],
    prompts: Sequence[Prompt] = PROMPTS,
    brands: tuple[Brand, ...] = BRANDS,
) -> tuple[Observation, ...]:
    """응답을 관측으로 바꾼다. 순수 함수 — 네트워크를 타지 않는다.

    브랜드 탐지가 여기서 일어나므로 이 함수만 테스트하면 탐지와 집계 사이의
    연결을 외부 호출 없이 검증할 수 있다.
    """
    index = {p.id: p for p in prompts}
    result = []

    for answer in answers:
        prompt = index.get(answer.prompt_id)
        if prompt is None:
            raise KeyError(f"알 수 없는 prompt_id: {answer.prompt_id}")

        # 실패한 응답은 텍스트를 보지 않는다. 빈 문자열에서 탐지를 돌리면
        # "언급 없음"이라는 관측이 생겨 실패와 미언급이 뒤섞인다.
        mentions = tuple(sorted(detect_mentions(answer.text, brands))) if answer.ok else ()

        result.append(
            Observation(
                engine=answer.engine,
                model=answer.model,
                prompt_id=answer.prompt_id,
                stage=prompt.stage,
                language=prompt.language,
                pair_id=prompt.pair_id,
                run_index=answer.run_index,
                mentions=mentions,
                ok=answer.ok,
                error=answer.error,
            )
        )

    return tuple(result)


def aggregate(
    observations: Sequence[Observation],
    scope_kind: ScopeKind = "overall",
    brands: tuple[Brand, ...] = BRANDS,
) -> tuple[Rate, ...]:
    """관측을 언급률로 집계한다. 순수 함수.

    **엔진은 항상 분리된다.** 이 함수에 "엔진을 합쳐서" 옵션이 없는 것은 실수가
    아니라 설계다. 플랫폼마다 응답당 인용 개수가 6배까지 차이 나므로, 합산 점수는
    노출이 좋아진 것과 인용을 많이 뿌리는 플랫폼의 비중이 커진 것을 구별하지
    못한다.

    실패한 관측(ok=False)은 분모에서 빠진다. 대신 몇 개가 빠졌는지를 Rate 가
    들고 나간다.
    """
    if scope_kind not in ("prompt", "stage", "language", "overall"):
        raise ValueError(f"알 수 없는 scope_kind: {scope_kind}")

    # (brand, engine, scope) 별로 성공/시도/실패를 센다
    buckets: dict[tuple[str, str, str], dict[str, object]] = {}

    for obs in observations:
        scope = obs.scope_value(scope_kind)
        for brand in brands:
            key = (brand.canonical, obs.engine, scope)
            bucket = buckets.setdefault(
                key,
                {"model": obs.model, "successes": 0, "n": 0, "attempted": 0, "failed": 0},
            )
            bucket["attempted"] = int(bucket["attempted"]) + 1
            if not obs.ok:
                bucket["failed"] = int(bucket["failed"]) + 1
                continue
            bucket["n"] = int(bucket["n"]) + 1
            if brand.canonical in obs.mentions:
                bucket["successes"] = int(bucket["successes"]) + 1

    rates = [
        Rate(
            brand=brand,
            engine=engine,
            model=str(bucket["model"]),
            scope_kind=scope_kind,
            scope_value=scope,
            interval=wilson_interval(int(bucket["successes"]), int(bucket["n"])),
            n_attempted=int(bucket["attempted"]),
            n_failed=int(bucket["failed"]),
            homogeneous=(scope_kind == "prompt"),
        )
        for (brand, engine, scope), bucket in buckets.items()
    ]

    # 언급률 높은 순. 같으면 이름 순으로 고정해 출력이 실행마다 흔들리지 않게 한다.
    rates.sort(key=lambda r: (-r.interval.point, r.brand, r.engine, r.scope_value))
    return tuple(rates)


def measure(
    engines: Sequence[Engine],
    runs: int,
    prompts: Sequence[Prompt] = PROMPTS,
    brands: tuple[Brand, ...] = BRANDS,
    workers: int = DEFAULT_WORKERS,
    on_progress: Callable[[int, int], None] | None = None,
    raw_path: str | Path | None = None,
) -> MeasurementRun:
    """엔진 × 프롬프트 × 반복 만큼 호출하고 관측을 모은다.

    `runs` 는 프롬프트 하나당 반복 횟수다. 총 호출 수는
    `len(engines) * len(prompts) * runs` 이고, 실행 전에 계산할 수 있다.
    도중에 줄이거나 늘리지 않는다.

    실패한 호출은 재시도하지 않는다. 재시도하면 "몇 번 시도했나"가 흐려져
    실패율 자체를 보고할 수 없게 된다. 실패는 실패로 기록한다.

    **`raw_path` 를 주면 원 응답 전문을 거기에 JSONL 로 저장한다.** 관측
    (`Observation`)은 탐지 결과만 들고 응답 본문은 버리므로, 이걸 지정하지
    않으면 증거가 남지 않는다. 실제 측정에서는 반드시 지정한다 — 기본값이
    `None` 인 것은 테스트에서 파일을 만들지 않기 위해서다.
    """
    if runs < 1:
        raise ValueError(f"runs 는 1 이상이어야 한다: {runs}")
    if not engines:
        raise ValueError("측정할 엔진이 없다")
    if not prompts:
        raise ValueError("측정할 프롬프트가 없다")

    started_at = datetime.now(timezone.utc).isoformat()
    total = len(engines) * len(prompts) * runs
    answers: list[Answer] = []
    done = 0

    tasks = [
        (engine, prompt, run_index)
        for engine in engines
        for prompt in prompts
        for run_index in range(runs)
    ]

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(engine.ask, prompt.id, prompt.text, run_index): (engine, prompt, run_index)
            for engine, prompt, run_index in tasks
        }
        for future in as_completed(futures):
            engine, prompt, run_index = futures[future]
            try:
                answers.append(future.result())
            except Exception as exc:
                # 어댑터가 이미 예외를 잡지만, 여기서도 관측을 잃지 않게 받아둔다.
                # 호출 하나가 사라지면 분모가 조용히 줄어든다.
                answers.append(
                    Answer.failure(
                        engine.name, engine.model, prompt.id, run_index,
                        f"{type(exc).__name__}: {exc}",
                    )
                )
            done += 1
            if on_progress:
                on_progress(done, total)

    finished_at = datetime.now(timezone.utc).isoformat()

    if raw_path is not None:
        # 관측으로 변환하기 전에 저장한다. 변환이 실패해도 원본은 남아야 한다 —
        # 몇 시간짜리 측정을 파싱 버그 하나로 잃을 수는 없다.
        answers.sort(key=lambda a: (a.engine, a.prompt_id, a.run_index))
        save_raw_jsonl(raw_path, answers)

    return MeasurementRun(
        started_at=started_at,
        finished_at=finished_at,
        runs_per_prompt=runs,
        engines=tuple(e.name for e in engines),
        models={e.name: e.model for e in engines},
        prompt_ids=tuple(p.id for p in prompts),
        brand_count=len(brands),
        observations=observations_from_answers(answers, prompts, brands),
        excluded_aliases=excluded_aliases(brands),
    )


def save_raw_jsonl(path: str | Path, answers: Sequence[Answer]) -> Path:
    """원 응답을 한 줄에 하나씩 저장한다.

    **응답 본문을 자르지 않는다.** 인용할 증거이자 재분석의 재료이므로 통째로
    남긴다. 실패한 호출도 같이 남긴다 — 무엇이 왜 실패했는지가 측정 조건의
    일부다.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)

    with target.open("w", encoding="utf-8") as f:
        for answer in answers:
            f.write(json.dumps(asdict(answer), ensure_ascii=False) + "\n")

    return target


def save_run(path: str | Path, run: MeasurementRun) -> Path:
    """측정 메타와 관측을 함께 저장한다. 결과만 남기지 않는다."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "meta": {
            "started_at": run.started_at,
            "finished_at": run.finished_at,
            "runs_per_prompt": run.runs_per_prompt,
            "engines": list(run.engines),
            "models": run.models,
            "prompt_ids": list(run.prompt_ids),
            "brand_count": run.brand_count,
            "web_search_enabled": run.web_search_enabled,
            "total_calls": run.total_calls,
            "failed_calls": run.failed_calls,
            "failures_by_prompt": run.failures_by_prompt(),
            "excluded_aliases": {k: list(v) for k, v in run.excluded_aliases.items()},
        },
        "observations": [asdict(o) for o in run.observations],
    }

    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return target


def load_run(path: str | Path) -> MeasurementRun:
    """저장된 측정을 다시 읽는다. 재집계할 때 쓴다 — 다시 호출하지 않는다."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    meta = payload["meta"]

    return MeasurementRun(
        started_at=meta["started_at"],
        finished_at=meta["finished_at"],
        runs_per_prompt=meta["runs_per_prompt"],
        engines=tuple(meta["engines"]),
        models=meta["models"],
        prompt_ids=tuple(meta["prompt_ids"]),
        brand_count=meta["brand_count"],
        observations=tuple(
            Observation(
                engine=o["engine"],
                model=o["model"],
                prompt_id=o["prompt_id"],
                stage=o["stage"],
                language=o["language"],
                pair_id=o["pair_id"],
                run_index=o["run_index"],
                mentions=tuple(o["mentions"]),
                ok=o["ok"],
                error=o["error"],
            )
            for o in payload["observations"]
        ),
        excluded_aliases={k: tuple(v) for k, v in meta["excluded_aliases"].items()},
        web_search_enabled=meta.get("web_search_enabled", False),
    )


def language_gap(
    observations: Sequence[Observation],
    brands: tuple[Brand, ...] = BRANDS,
) -> tuple[tuple[str, str, Rate, Rate, bool], ...]:
    """같은 브랜드가 한국어와 영어에서 다르게 나오는가.

    반환값은 `(브랜드, 엔진, 한국어 Rate, 영어 Rate, 구간이_겹치는가)` 다.
    **겹치지 않을 때만 차이를 주장할 수 있다.** 점추정치만 비교해 "한국어에서
    더 많이 나온다"고 말하는 것이 이 분야의 흔한 실수다.

    주의 — 겹친다고 차이가 없다는 뜻은 아니다. 보수적인 검사이지 가설검정이
    아니다 (`stats.overlaps` 와 같은 한계).
    """
    from .stats import overlaps

    by_lang = aggregate(observations, "language", brands)
    ko = {(r.brand, r.engine): r for r in by_lang if r.scope_value == "ko"}
    en = {(r.brand, r.engine): r for r in by_lang if r.scope_value == "en"}

    pairs = []
    for key in sorted(set(ko) & set(en)):
        k, e = ko[key], en[key]
        pairs.append((key[0], key[1], k, e, overlaps(k.interval, e.interval)))

    # 차이가 큰 순으로. 리포트에서 위에 오는 것이 볼 가치가 있는 것이어야 한다.
    pairs.sort(key=lambda t: -abs(t[2].interval.point - t[3].interval.point))
    return tuple(pairs)


def build_engines(names: Sequence[str]) -> list[Engine]:
    """이름으로 어댑터를 만든다. 키가 없으면 그 자리에서 실패한다.

    조용히 건너뛰지 않는 이유 — 엔진 하나가 빠진 채로 측정이 돌면 결과를 볼 때
    그 사실을 잊는다. 무엇을 쟀는지가 측정 조건이다.
    """
    from .engines import AnthropicEngine, OpenAIEngine

    built: list[Engine] = []
    for name in names:
        if name == "anthropic":
            if not os.environ.get("ANTHROPIC_API_KEY"):
                raise RuntimeError("ANTHROPIC_API_KEY 가 없다. 새 셸에서 환경변수를 확인한다")
            built.append(AnthropicEngine())
        elif name == "openai":
            if not os.environ.get("OPENAI_API_KEY"):
                raise RuntimeError("OPENAI_API_KEY 가 없다. 새 셸에서 환경변수를 확인한다")
            built.append(OpenAIEngine())
        else:
            raise ValueError(f"알 수 없는 엔진: {name}")
    return built
