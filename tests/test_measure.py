"""측정 루프의 순수 함수 검증.

네트워크를 타지 않는 부분만 다룬다 — 관측 변환, 집계, 저장·복원. 여기서
검증하려는 것은 **이 저장소가 주장하는 규칙이 코드로 집행되는가**다.

  - 실패한 호출이 분모에서 빠지는가
  - 엔진이 합산되지 않는가
  - 혼합 집계에 표시가 붙는가
"""

from __future__ import annotations

import pytest

from geolab.brands import Brand
from geolab.engines import Answer
from geolab.measure import (
    Observation,
    aggregate,
    language_gap,
    load_run,
    measure,
    observations_from_answers,
    save_raw_jsonl,
    save_run,
)
from geolab.prompts import Prompt

# 테스트용 최소 세트. 실제 프롬프트 세트가 바뀌어도 테스트가 흔들리지 않게 한다.
P_KO = Prompt("t1-ko", "discovery", "ko", "협업 툴 추천해줘", "t1")
P_EN = Prompt("t1-en", "discovery", "en", "Recommend collaboration tools", "t1")
P_ALT = Prompt("t2-ko", "alternative", "ko", "슬랙 말고 다른 거", "t2")
TEST_PROMPTS = (P_KO, P_EN, P_ALT)

TEST_BRANDS = (
    Brand("Slack", ("Slack", "슬랙")),
    Brand("Notion", ("Notion", "노션")),
)


def ok_answer(prompt_id: str, text: str, run_index: int = 0, engine: str = "anthropic") -> Answer:
    return Answer(
        engine=engine,
        model=f"{engine}-test-model",
        prompt_id=prompt_id,
        run_index=run_index,
        text=text,
        ok=True,
    )


def failed_answer(prompt_id: str, run_index: int = 0, engine: str = "anthropic") -> Answer:
    return Answer.failure(engine, f"{engine}-test-model", prompt_id, run_index, "RateLimitError: 429")


# --- observations_from_answers ---


def test_성공_응답에서_브랜드를_탐지한다():
    answers = [ok_answer("t1-ko", "슬랙과 노션을 추천합니다")]
    obs = observations_from_answers(answers, TEST_PROMPTS, TEST_BRANDS)

    assert len(obs) == 1
    assert obs[0].mentions == ("Notion", "Slack")  # 정렬됨
    assert obs[0].ok is True


def test_한_응답에_같은_브랜드가_여러_번_나와도_한_번으로_센다():
    answers = [ok_answer("t1-ko", "슬랙은 좋다. 슬랙을 쓰세요. Slack 최고")]
    obs = observations_from_answers(answers, TEST_PROMPTS, TEST_BRANDS)

    assert obs[0].mentions == ("Slack",)


def test_실패한_응답은_탐지를_돌리지_않는다():
    """실패 응답의 빈 텍스트에서 탐지하면 '언급 없음' 관측이 생겨 미언급과 섞인다."""
    obs = observations_from_answers([failed_answer("t1-ko")], TEST_PROMPTS, TEST_BRANDS)

    assert obs[0].ok is False
    assert obs[0].mentions == ()
    assert obs[0].error is not None


def test_프롬프트_메타가_관측에_붙는다():
    obs = observations_from_answers([ok_answer("t1-en", "Slack")], TEST_PROMPTS, TEST_BRANDS)

    assert obs[0].stage == "discovery"
    assert obs[0].language == "en"
    assert obs[0].pair_id == "t1"


def test_모르는_프롬프트는_조용히_넘기지_않는다():
    with pytest.raises(KeyError):
        observations_from_answers([ok_answer("없는-id", "Slack")], TEST_PROMPTS, TEST_BRANDS)


# --- aggregate: 실패 제외 ---


def test_실패한_호출은_분모에서_빠진다():
    """이 저장소의 핵심 규칙. 실패를 '언급 안 됨'으로 세면 언급률이 아래로 편향된다."""
    answers = [
        ok_answer("t1-ko", "슬랙 추천", run_index=0),
        ok_answer("t1-ko", "노션 추천", run_index=1),  # Slack 미언급
        failed_answer("t1-ko", run_index=2),
        failed_answer("t1-ko", run_index=3),
    ]
    obs = observations_from_answers(answers, TEST_PROMPTS, TEST_BRANDS)
    rates = aggregate(obs, "prompt", TEST_BRANDS)

    slack = next(r for r in rates if r.brand == "Slack")

    # 성공 2건 중 1건 언급 → 50%. 실패 2건을 미언급으로 셌다면 25% 가 됐을 것이다
    assert slack.interval.n == 2
    assert slack.interval.point == pytest.approx(0.5)
    assert slack.n_attempted == 4
    assert slack.n_failed == 2


def test_전부_실패하면_표본이_0이고_구간은_전체_범위다():
    answers = [failed_answer("t1-ko", run_index=i) for i in range(3)]
    obs = observations_from_answers(answers, TEST_PROMPTS, TEST_BRANDS)
    rates = aggregate(obs, "prompt", TEST_BRANDS)

    slack = next(r for r in rates if r.brand == "Slack")

    assert slack.interval.n == 0
    assert (slack.interval.low, slack.interval.high) == (0.0, 1.0)
    assert slack.n_failed == 3


def test_한_번도_언급되지_않아도_상한은_0이_아니다():
    """0/20 을 '확실히 0%'로 보고하는 것이 Wald 구간의 실패다. Wilson 은 상한을 낸다."""
    answers = [ok_answer("t1-ko", "다른 도구를 쓰세요", run_index=i) for i in range(20)]
    obs = observations_from_answers(answers, TEST_PROMPTS, TEST_BRANDS)
    rates = aggregate(obs, "prompt", TEST_BRANDS)

    slack = next(r for r in rates if r.brand == "Slack")

    assert slack.interval.point == 0.0
    assert slack.interval.high > 0.10  # "0% 부터 약 16% 까지 가능하다"


# --- aggregate: 엔진 분리 ---


def test_엔진은_합산되지_않는다():
    """플랫폼마다 인용 행동이 달라 합산 점수는 구성비에 좌우된다."""
    answers = [
        ok_answer("t1-ko", "슬랙 추천", engine="anthropic"),
        ok_answer("t1-ko", "다른 것", engine="openai"),
    ]
    obs = observations_from_answers(answers, TEST_PROMPTS, TEST_BRANDS)
    rates = aggregate(obs, "prompt", TEST_BRANDS)

    slack_rates = {r.engine: r for r in rates if r.brand == "Slack"}

    assert set(slack_rates) == {"anthropic", "openai"}
    assert slack_rates["anthropic"].interval.point == pytest.approx(1.0)
    assert slack_rates["openai"].interval.point == pytest.approx(0.0)
    # 합쳤다면 50% 가 나왔을 것이다. 그런 값은 어디에도 없어야 한다
    assert all(r.interval.point in (0.0, 1.0) for r in slack_rates.values())


def test_모델_ID가_결과에_남는다():
    """어느 모델을 쟀는지가 곧 측정 조건이다."""
    obs = observations_from_answers([ok_answer("t1-ko", "슬랙")], TEST_PROMPTS, TEST_BRANDS)
    rates = aggregate(obs, "prompt", TEST_BRANDS)

    assert all(r.model == "anthropic-test-model" for r in rates)


# --- aggregate: scope ---


def test_프롬프트별_집계만_동질로_표시된다():
    obs = observations_from_answers([ok_answer("t1-ko", "슬랙")], TEST_PROMPTS, TEST_BRANDS)

    assert all(r.homogeneous for r in aggregate(obs, "prompt", TEST_BRANDS))
    assert not any(r.homogeneous for r in aggregate(obs, "overall", TEST_BRANDS))
    assert not any(r.homogeneous for r in aggregate(obs, "stage", TEST_BRANDS))
    assert not any(r.homogeneous for r in aggregate(obs, "language", TEST_BRANDS))


def test_언어별로_나눠_집계한다():
    answers = [
        ok_answer("t1-ko", "슬랙 추천"),
        ok_answer("t1-en", "Use Notion"),
    ]
    obs = observations_from_answers(answers, TEST_PROMPTS, TEST_BRANDS)
    rates = aggregate(obs, "language", TEST_BRANDS)

    slack = {r.scope_value: r for r in rates if r.brand == "Slack"}

    assert slack["ko"].interval.point == pytest.approx(1.0)
    assert slack["en"].interval.point == pytest.approx(0.0)


def test_단계별로_나눠_집계한다():
    answers = [
        ok_answer("t1-ko", "슬랙 추천"),
        ok_answer("t2-ko", "노션 쓰세요"),
    ]
    obs = observations_from_answers(answers, TEST_PROMPTS, TEST_BRANDS)
    rates = aggregate(obs, "stage", TEST_BRANDS)

    scopes = {r.scope_value for r in rates}
    assert scopes == {"discovery", "alternative"}


def test_알_수_없는_scope는_거부한다():
    with pytest.raises(ValueError):
        aggregate((), "brand")  # type: ignore[arg-type]


def test_언급되지_않은_브랜드도_결과에_나온다():
    """0% 도 측정 결과다. 빠뜨리면 '재지 않은 것'과 '0 이었던 것'이 구별되지 않는다."""
    obs = observations_from_answers([ok_answer("t1-ko", "슬랙만 언급")], TEST_PROMPTS, TEST_BRANDS)
    rates = aggregate(obs, "prompt", TEST_BRANDS)

    assert {r.brand for r in rates} == {"Slack", "Notion"}


# --- language_gap ---


def test_언어_격차는_구간_겹침을_함께_돌려준다():
    answers = (
        [ok_answer("t1-ko", "슬랙 추천", run_index=i) for i in range(20)]
        + [ok_answer("t1-en", "Use something else", run_index=i) for i in range(20)]
    )
    obs = observations_from_answers(answers, TEST_PROMPTS, TEST_BRANDS)
    gaps = language_gap(obs, TEST_BRANDS)

    slack = next(g for g in gaps if g[0] == "Slack")
    _, _, ko, en, overlapping = slack

    assert ko.interval.point == pytest.approx(1.0)
    assert en.interval.point == pytest.approx(0.0)
    assert overlapping is False  # 20/20 대 0/20 이면 겹치지 않는다


def test_표본이_적으면_큰_차이도_겹친다():
    """1/1 대 0/1 은 차이처럼 보이지만 구간이 겹친다. 주장할 수 없다."""
    answers = [ok_answer("t1-ko", "슬랙"), ok_answer("t1-en", "nothing here")]
    obs = observations_from_answers(answers, TEST_PROMPTS, TEST_BRANDS)
    gaps = language_gap(obs, TEST_BRANDS)

    slack = next(g for g in gaps if g[0] == "Slack")
    assert slack[4] is True


# --- MeasurementRun ---


def test_실패가_어느_프롬프트에서_났는지_센다():
    """실패가 특정 프롬프트에 몰리면 제외가 무작위가 아니다 — 편향 신호다."""
    answers = [
        failed_answer("t1-ko", run_index=0),
        failed_answer("t1-ko", run_index=1),
        ok_answer("t1-en", "Slack", run_index=0),
    ]
    obs = observations_from_answers(answers, TEST_PROMPTS, TEST_BRANDS)
    run = _run_with(obs)

    assert run.failures_by_prompt() == {"t1-ko": 2}
    assert run.failed_calls == 2
    assert run.total_calls == 3


# --- 저장·복원 ---


def test_측정을_저장하고_다시_읽으면_관측이_보존된다(tmp_path):
    answers = [ok_answer("t1-ko", "슬랙과 노션"), failed_answer("t1-en", run_index=0)]
    obs = observations_from_answers(answers, TEST_PROMPTS, TEST_BRANDS)
    run = _run_with(obs)

    path = save_run(tmp_path / "run.json", run)
    restored = load_run(path)

    assert restored.observations == run.observations
    assert restored.runs_per_prompt == run.runs_per_prompt
    assert restored.models == run.models


def test_원_응답은_본문까지_저장한다(tmp_path):
    """인용 가능한 증거이자 재분석의 재료다. 자르지 않는다."""
    body = "슬랙을 추천합니다. " * 200
    path = save_raw_jsonl(tmp_path / "raw.jsonl", [ok_answer("t1-ko", body)])

    lines = path.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 1

    import json

    assert json.loads(lines[0])["text"] == body


def test_실패한_호출도_원본에_남긴다(tmp_path):
    path = save_raw_jsonl(tmp_path / "raw.jsonl", [failed_answer("t1-ko")])

    import json

    record = json.loads(path.read_text(encoding="utf-8").strip())
    assert record["ok"] is False
    assert "429" in record["error"]


# --- measure 인자 검증 ---


def test_반복_수가_0이면_거부한다():
    with pytest.raises(ValueError):
        measure([_FakeEngine()], runs=0, prompts=TEST_PROMPTS, brands=TEST_BRANDS)


def test_엔진이_없으면_거부한다():
    with pytest.raises(ValueError):
        measure([], runs=1, prompts=TEST_PROMPTS, brands=TEST_BRANDS)


def test_고정된_반복_수만큼_정확히_호출한다():
    """조기 종료가 없다. CI 폭은 단조 감소하지 않으므로 도중에 멈추면 안 된다."""
    engine = _FakeEngine()
    run = measure([engine], runs=5, prompts=TEST_PROMPTS, brands=TEST_BRANDS, workers=2)

    assert engine.calls == len(TEST_PROMPTS) * 5
    assert run.total_calls == len(TEST_PROMPTS) * 5
    assert run.runs_per_prompt == 5


def test_어댑터가_던진_예외에도_관측을_잃지_않는다():
    """호출 하나가 사라지면 분모가 조용히 줄어든다."""
    engine = _ExplodingEngine()
    run = measure([engine], runs=2, prompts=TEST_PROMPTS, brands=TEST_BRANDS, workers=2)

    assert run.total_calls == len(TEST_PROMPTS) * 2
    assert run.failed_calls == len(TEST_PROMPTS) * 2


class _FakeEngine:
    """네트워크를 타지 않는 어댑터. 호출 횟수만 센다."""

    name = "fake"
    model = "fake-model"

    def __init__(self) -> None:
        self.calls = 0

    def ask(self, prompt_id: str, text: str, run_index: int) -> Answer:
        self.calls += 1
        return Answer(self.name, self.model, prompt_id, run_index, "슬랙 추천", ok=True)


class _ExplodingEngine:
    name = "boom"
    model = "boom-model"

    def ask(self, prompt_id: str, text: str, run_index: int) -> Answer:
        raise RuntimeError("어댑터가 예외를 흘렸다")


def _run_with(observations: tuple[Observation, ...]):
    from geolab.measure import MeasurementRun

    return MeasurementRun(
        started_at="2026-08-08T00:00:00+00:00",
        finished_at="2026-08-08T00:01:00+00:00",
        runs_per_prompt=1,
        engines=("anthropic",),
        models={"anthropic": "anthropic-test-model"},
        prompt_ids=tuple(p.id for p in TEST_PROMPTS),
        brand_count=len(TEST_BRANDS),
        observations=observations,
        excluded_aliases={},
    )
