"""측정 대상 모델 어댑터.

여기서 모델은 **도구가 아니라 측정 대상**이다. 우리가 재려는 것은 "이 모델이
브랜드를 어떻게 다루는가"이므로, 어댑터는 실제 사용자가 받는 것과 최대한 같은
응답을 받아와야 한다.

## 샘플링 파라미터를 건드리지 않는 이유

두 가지가 겹친다.

1. **재려는 것이 변동 그 자체다.** temperature 를 0으로 고정하면 변동이 줄어
   재현성은 얻지만, 실제 사용자가 겪는 현실을 재는 게 아니게 된다.
2. **현행 모델은 아예 받지 않는다.** Anthropic 의 Claude Opus 5 계열은
   `temperature` / `top_p` / `top_k` 를 제거해서, 보내면 400 에러가 난다.

즉 이건 설계 취향이 아니라 제약이다. 기본 설정 그대로 호출한다.

## 웹 검색을 켜지 않는 이유 (v1)

조사에서 확인한 대로 생성형 엔진은 2층이다 — 학습된 지식과 실시간 검색. 검색을
끄고 물으면 **"모델이 이 브랜드를 알고 있는가"**만 분리해서 잴 수 있고, 그것이
조사에서 정리한 3단계 중 [1] 닿는가에 해당한다. 검색을 켠 측정은 [2] 선택되는가와
섞이므로 별도 실험으로 나눈다.

**이건 한계이자 설계다. 리포트에 명시한다.**
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class Answer:
    """모델 응답 한 건.

    ok=False 인 관측은 **표본에서 제외한다.** 호출이 실패한 것을 "언급 안 됨"으로
    세면 언급률이 아래로 편향된다 — 조용히 틀리는 전형적인 방식이다.
    """

    engine: str
    model: str
    prompt_id: str
    run_index: int
    text: str
    ok: bool
    error: str | None = None

    @classmethod
    def failure(cls, engine: str, model: str, prompt_id: str, run_index: int, error: str) -> "Answer":
        return cls(
            engine=engine,
            model=model,
            prompt_id=prompt_id,
            run_index=run_index,
            text="",
            ok=False,
            error=error,
        )


class Engine(Protocol):
    name: str
    model: str

    def ask(self, prompt_id: str, text: str, run_index: int) -> Answer: ...


class AnthropicEngine:
    """Claude. 기본 설정으로 호출한다 — 시스템 프롬프트도 붙이지 않는다.

    시스템 프롬프트를 붙이면 우리가 만든 맥락을 재게 된다. 우리가 재려는 것은
    맨몸의 모델이 브랜드를 어떻게 다루는가다.
    """

    def __init__(self, model: str | None = None, max_tokens: int = 2048) -> None:
        import anthropic  # 지연 import — 이 엔진을 안 쓰면 패키지도 필요 없다

        self.name = "anthropic"
        self.model = model or os.environ.get("GEOLAB_ANTHROPIC_MODEL", "claude-opus-5")
        self.max_tokens = max_tokens
        self._client = anthropic.Anthropic()

    def ask(self, prompt_id: str, text: str, run_index: int) -> Answer:
        try:
            response = self._client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                messages=[{"role": "user", "content": text}],
            )
        except Exception as exc:  # 실패는 기록한다. 삼키지 않는다
            return Answer.failure(self.name, self.model, prompt_id, run_index, f"{type(exc).__name__}: {exc}")

        if response.stop_reason == "refusal":
            return Answer.failure(self.name, self.model, prompt_id, run_index, "refusal")

        body = "\n".join(block.text for block in response.content if block.type == "text")
        return Answer(
            engine=self.name,
            model=self.model,
            prompt_id=prompt_id,
            run_index=run_index,
            text=body,
            ok=True,
        )


class OpenAIEngine:
    """ChatGPT 계열.

    모델 ID 는 환경변수 `GEOLAB_OPENAI_MODEL` 로 넘긴다. 여기에 특정 ID 를 기본값으로
    박아두지 않는 이유는, 틀린 ID 를 조용히 쓰는 것보다 실행 시점에 명시적으로
    실패하는 편이 낫기 때문이다 — 어느 모델을 쟀는지가 곧 측정 조건이다.
    """

    def __init__(self, model: str | None = None, max_tokens: int = 8192) -> None:
        """`max_tokens` 기본값이 큰 이유 — 추론 토큰이 여기 포함된다.

        gpt-5 계열은 답을 내기 전에 추론 토큰을 먼저 쓰고, 그 소비량이
        `max_completion_tokens` 한도에 함께 잡힌다. 한도가 빠듯하면 추론에
        다 쓰이고 **본문이 빈 채로 반환된다.**

        2026-08-08 관측 — 한도 2048로 4회 호출했더니 1회가 빈 응답이었고,
        그게 한국어 프롬프트였다. 여기서 멈추면 안 되는 이유는 이렇다:

          빈 응답은 실패로 분류되어 표본에서 빠진다 (설계상 맞다)
          → 그런데 한국어가 더 긴 추론을 유발한다면 실패가 한국어에 몰린다
          → 한국어 표본만 체계적으로 줄어든다
          → **언어 간 차이를 재려는 측정이 바로 그 축에서 오염된다**

        실패 제외는 편향을 막는 장치인데, 실패가 무작위가 아니면 그 장치가
        거꾸로 편향을 만든다. 그래서 한도를 넉넉히 잡아 **실패 자체를 줄인다.**

        비용은 상한이 아니라 실제 사용량으로 매겨지므로 한도를 올려도
        평소 호출 비용은 늘지 않는다.
        """
        from openai import OpenAI  # 지연 import

        resolved = model or os.environ.get("GEOLAB_OPENAI_MODEL")
        if not resolved:
            raise RuntimeError(
                "OpenAI 모델 ID 가 지정되지 않았다. "
                "환경변수 GEOLAB_OPENAI_MODEL 을 설정하거나 model= 로 넘긴다. "
                "어느 모델을 쟀는지가 측정 조건이므로 기본값을 두지 않는다."
            )

        self.name = "openai"
        self.model = resolved
        self.max_tokens = max_tokens
        self._client = OpenAI()

    def ask(self, prompt_id: str, text: str, run_index: int) -> Answer:
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": text}],
                max_completion_tokens=self.max_tokens,
            )
        except Exception as exc:
            return Answer.failure(self.name, self.model, prompt_id, run_index, f"{type(exc).__name__}: {exc}")

        body = response.choices[0].message.content or ""
        if not body.strip():
            return Answer.failure(self.name, self.model, prompt_id, run_index, "empty response")

        return Answer(
            engine=self.name,
            model=self.model,
            prompt_id=prompt_id,
            run_index=run_index,
            text=body,
            ok=True,
        )


def available_engines() -> list[str]:
    """환경변수 기준으로 지금 쓸 수 있는 엔진 이름.

    키가 없으면 조용히 건너뛰지 않고, 무엇이 빠졌는지 호출부가 알 수 있게 한다.
    """
    names = []
    if os.environ.get("ANTHROPIC_API_KEY"):
        names.append("anthropic")
    if os.environ.get("OPENAI_API_KEY"):
        names.append("openai")
    return names
