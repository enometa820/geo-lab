"""응답을 벡터로 바꿔 유형을 찾는다 — 탐색적 분석.

앞선 모듈들이 "무엇이 언급됐나"를 셌다면 여기서는 **"답변이 어떻게 생겼나"**를
본다. 같은 질문에 20번 답해도 어떤 때는 표로, 어떤 때는 문단으로, 어떤 때는
국내 도구 위주로 답한다. 그 유형이 나뉜다면 브랜드 노출도 유형을 탄다.

## 이 모듈의 결과는 근거가 아니라 단서다

다른 모듈과 성격이 다르므로 분명히 해 둔다.

    언급률·회귀    가설을 검증한다. 신뢰구간이 붙고 근거로 인용할 수 있다
    군집화         가설을 만든다. 정답 라벨이 없어 "맞았다"를 말할 수 없다

군집 개수를 몇 개로 잡느냐에 따라 그림이 달라지고, 실루엣 계수가 높다고 그
군집이 의미 있는 것도 아니다. 그래서 이 모듈은 **군집에 이름을 붙이지 않는다.**
각 군집이 어떤 단계·언어·브랜드로 채워졌는지 구성만 내놓고, 해석은 사람이 한다.

## 언어가 먼저 갈린다

한국어와 영어 응답을 함께 군집화하면 거의 확실히 **언어로 1차 분할**된다. 그건
이미 아는 사실이라 새 정보가 없다. `group_by="language"` 로 언어 안에서 따로
군집화하면 그 축을 지우고 나머지 구조를 볼 수 있다.

## 캐시

임베딩은 같은 텍스트에 같은 벡터를 준다. 다시 호출할 이유가 없으므로 텍스트
해시로 캐시한다. 측정 원자료와 같은 폴더에 두어 **결과와 그 재료가 같이 남게**
한다.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"

# 임베딩 API 는 한 번에 여러 개를 받는다. 너무 크면 요청이 거부되므로 나눠 보낸다.
BATCH_SIZE = 64


def text_key(text: str, model: str) -> str:
    """캐시 키. 모델이 다르면 벡터도 다르므로 모델명을 같이 해싱한다."""
    digest = hashlib.sha256(f"{model}\x00{text}".encode("utf-8")).hexdigest()
    return digest[:32]


@dataclass(frozen=True)
class EmbeddingSet:
    """텍스트와 벡터의 짝.

    `keys` 는 캐시 키이자 순서 보장 장치다 — 벡터 행렬의 i번째 행이 어느
    텍스트였는지를 잃으면 군집 결과를 원 응답에 되붙일 수 없다.
    """

    model: str
    keys: tuple[str, ...]
    vectors: "object"  # numpy.ndarray. 타입 힌트에 numpy 를 끌어오지 않는다

    @property
    def dimension(self) -> int:
        return int(self.vectors.shape[1])


def embed_texts(
    texts: Sequence[str],
    cache_path: str | Path,
    model: str = DEFAULT_EMBEDDING_MODEL,
    allow_api: bool = True,
) -> EmbeddingSet:
    """텍스트를 벡터로. 캐시에 있으면 호출하지 않는다.

    `allow_api=False` 면 캐시에 없는 텍스트를 만났을 때 호출하지 않고 실패한다.
    비용이 드는 호출이 예상치 못하게 일어나는 것을 막기 위한 장치다 — 리포트를
    다시 그릴 때는 항상 캐시만 써야 한다.
    """
    import numpy as np

    cache = Path(cache_path)
    stored: dict[str, "np.ndarray"] = {}
    if cache.exists():
        with np.load(cache, allow_pickle=False) as data:
            cached_keys = [str(k) for k in data["keys"]]
            stored = {k: v for k, v in zip(cached_keys, data["vectors"])}

    keys = [text_key(t, model) for t in texts]
    missing = [(k, t) for k, t in zip(keys, texts) if k not in stored]
    # 같은 텍스트가 두 번 들어와도 한 번만 호출한다
    unique_missing = list({k: t for k, t in missing}.items())

    if unique_missing:
        if not allow_api:
            raise RuntimeError(
                f"캐시에 없는 텍스트가 {len(unique_missing)}건 있는데 allow_api=False 다. "
                f"먼저 임베딩을 생성한다"
            )
        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY 가 없다. 새 셸에서 환경변수를 확인한다")

        from openai import OpenAI

        client = OpenAI()
        for start in range(0, len(unique_missing), BATCH_SIZE):
            chunk = unique_missing[start : start + BATCH_SIZE]
            response = client.embeddings.create(
                model=model, input=[t for _, t in chunk]
            )
            for (key, _), item in zip(chunk, response.data):
                stored[key] = np.asarray(item.embedding, dtype=np.float32)

        cache.parent.mkdir(parents=True, exist_ok=True)
        all_keys = list(stored)
        np.savez_compressed(
            cache,
            keys=np.array(all_keys),
            vectors=np.stack([stored[k] for k in all_keys]),
        )

    return EmbeddingSet(
        model=model,
        keys=tuple(keys),
        vectors=np.stack([stored[k] for k in keys]),
    )


@dataclass(frozen=True)
class ClusterChoice:
    """군집 개수 후보 하나의 성적.

    실루엣 계수는 -1~1 이고, **0 근처면 군집 구조가 사실상 없다는 뜻**이다.
    0.5를 넘으면 뚜렷하다고 보는 것이 통상이지만 절대 기준은 아니다.
    """

    k: int
    silhouette: float

    def describe(self) -> str:
        return f"k={self.k}: 실루엣 {self.silhouette:.3f}"


@dataclass(frozen=True)
class ClusterResult:
    """군집 결과와 그것을 믿을 정도.

    `weak_structure` 가 True 면 군집을 나눴다는 사실 자체를 근거로 쓰지 않는다.
    나누라고 했으니 나뉜 것이지 데이터에 경계가 있는 것이 아니다.
    """

    labels: tuple[int, ...]
    k: int
    silhouette: float
    candidates: tuple[ClusterChoice, ...]
    group: str | None = None

    # 이 아래로는 "구조가 있다"고 말하기 어렵다고 보는 관행적 경계
    WEAK_SILHOUETTE = 0.15

    @property
    def weak_structure(self) -> bool:
        return self.silhouette < self.WEAK_SILHOUETTE

    def describe(self) -> str:
        head = f"{'[' + self.group + '] ' if self.group else ''}k={self.k}, 실루엣 {self.silhouette:.3f}"
        if self.weak_structure:
            return head + " — 구조가 약하다. 군집 경계를 근거로 쓰지 않는다"
        return head


def choose_clusters(
    vectors: "object",
    k_range: Sequence[int] = (2, 3, 4, 5, 6),
    random_state: int = 0,
    group: str | None = None,
) -> ClusterResult:
    """실루엣 계수가 가장 높은 k 를 고른다.

    **후보 전부의 점수를 함께 돌려준다.** 최고점만 보고하면 "k=3이 가장 좋다"가
    "k=3이 좋다"로 읽힌다. 2등과 0.01 차이였다면 그 선택은 사실상 임의다.

    `random_state` 를 고정하는 이유는 KMeans 초기값이 난수라 실행마다 결과가
    달라지기 때문이다. 재현되지 않는 그림은 리포트에 실을 수 없다.
    """
    import numpy as np
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    matrix = np.asarray(vectors)
    n = matrix.shape[0]
    usable = [k for k in k_range if 2 <= k < n]
    if not usable:
        raise ValueError(f"표본 {n}건으로는 군집화할 수 없다: k_range={tuple(k_range)}")

    scored = []
    labels_by_k = {}
    for k in usable:
        model = KMeans(n_clusters=k, n_init=10, random_state=random_state)
        labels = model.fit_predict(matrix)
        score = float(silhouette_score(matrix, labels))
        scored.append(ClusterChoice(k=k, silhouette=score))
        labels_by_k[k] = labels

    best = max(scored, key=lambda c: c.silhouette)
    return ClusterResult(
        labels=tuple(int(x) for x in labels_by_k[best.k]),
        k=best.k,
        silhouette=best.silhouette,
        candidates=tuple(scored),
        group=group,
    )


@dataclass(frozen=True)
class ClusterProfile:
    """군집 하나의 구성.

    군집에 이름을 붙이는 대신 **무엇으로 채워졌는지**를 보여준다. 읽는 사람이
    스스로 판단할 수 있어야 하고, 그러려면 원재료가 보여야 한다.
    """

    cluster: int
    size: int
    stage_mix: dict[str, int]
    language_mix: dict[str, int]
    brand_share: dict[str, float]
    mean_length: float
    example_prompt_ids: tuple[str, ...]


def profile_clusters(
    rows: Sequence[dict],
    labels: Sequence[int],
    top_brands: int = 6,
) -> tuple[ClusterProfile, ...]:
    """각 군집이 무엇으로 이루어졌는지. 순수 함수.

    `rows` 는 `analysis.response_texts` 의 출력이고 `labels` 와 순서가 같아야
    한다. 길이가 다르면 조용히 어긋나므로 그 자리에서 실패시킨다.
    """
    if len(rows) != len(labels):
        raise ValueError(f"행 {len(rows)}개와 라벨 {len(labels)}개의 길이가 다르다")

    profiles = []
    for cluster in sorted(set(labels)):
        members = [r for r, lab in zip(rows, labels) if lab == cluster]
        size = len(members)

        stage_mix: dict[str, int] = {}
        language_mix: dict[str, int] = {}
        brand_hits: dict[str, int] = {}
        for r in members:
            stage_mix[str(r["stage"])] = stage_mix.get(str(r["stage"]), 0) + 1
            language_mix[str(r["language"])] = language_mix.get(str(r["language"]), 0) + 1
            for brand in r["mentions"]:
                brand_hits[brand] = brand_hits.get(brand, 0) + 1

        share = sorted(
            ((b, c / size) for b, c in brand_hits.items()), key=lambda kv: -kv[1]
        )[:top_brands]

        profiles.append(
            ClusterProfile(
                cluster=cluster,
                size=size,
                stage_mix=dict(sorted(stage_mix.items())),
                language_mix=dict(sorted(language_mix.items())),
                brand_share=dict(share),
                mean_length=sum(int(r["length"]) for r in members) / size,
                example_prompt_ids=tuple(
                    sorted({str(r["prompt_id"]) for r in members})[:5]
                ),
            )
        )
    return tuple(profiles)


def label_agreement(
    labels: Sequence[int],
    references: dict[str, Sequence[object]],
) -> dict[str, float]:
    """군집이 **이미 아는 라벨을 되찾은 것뿐인지** 재는 진단.

    조정 랜드 지수(adjusted Rand index)를 쓴다. 1이면 완전히 같은 분할이고,
    0이면 우연 수준이다.

    ## 왜 이 진단이 필요한가

    군집화는 항상 무언가를 내놓는다. k를 주면 반드시 k개로 나눈다. 그래서
    **"나뉘었다"는 사실만으로는 아무것도 알 수 없고**, 나뉜 결과가 새로운
    정보인지 아니면 이미 가진 변수의 재발견인지를 따로 확인해야 한다.

    지수가 높으면 그 군집은 새 발견이 아니다 — 예컨대 언어와의 일치도가 높으면
    "한국어 답변과 영어 답변이 다르게 생겼다"를 비싸게 다시 확인한 것이다.
    낮다고 해서 의미가 있다는 뜻도 아니다. 이 지수는 **과잉 해석을 막는
    장치이지 의미를 증명하는 장치가 아니다.**
    """
    from sklearn.metrics import adjusted_rand_score

    result = {}
    for name, reference in references.items():
        if len(reference) != len(labels):
            raise ValueError(
                f"{name}: 라벨 {len(labels)}개와 참조 {len(reference)}개의 길이가 다르다"
            )
        result[name] = float(adjusted_rand_score(list(reference), list(labels)))
    return dict(sorted(result.items(), key=lambda kv: -kv[1]))


def has_natural_k(candidates: Sequence[ClusterChoice]) -> bool:
    """최고점이 시도 범위의 **끝**에 있으면 자연스러운 군집 개수를 못 찾은 것이다.

    그 경우 "가장 좋은 k"는 데이터가 알려준 값이 아니라 우리가 어디서 멈췄는지를
    알려주는 값이다. 범위를 넓히면 최고점도 따라 움직인다.

    ## 왜 "한 번이라도 내려갔나"로 재면 안 되나

    처음에 그렇게 구현했다가 틀렸다. 실제 측정에서 실루엣이 k=2..12 내내 오르되
    중간에 한 칸(k=8→9)만 살짝 내려갔는데, 그 굴곡 하나 때문에 "정점을 찾았다"로
    판정됐다. 그림은 누가 봐도 우상향인데 본문이 반대로 말한 것이다.

    **국소적인 요철이 아니라 최고점의 위치가 판정 근거다.** 최저 k 에서 최고인
    경우는 경계지만 문제로 보지 않는다 — k=2 는 "나눌 수 있는 가장 적은 수"라
    그 자체로 답이 될 수 있다. 위험한 것은 위쪽 경계다.

    후보가 2개 이하면 판정하지 않는다 (증가·감소를 말할 자료가 없다).
    """
    if len(candidates) < 3:
        return True
    ordered = sorted(candidates, key=lambda c: c.k)
    best = max(ordered, key=lambda c: c.silhouette)
    return best.k != ordered[-1].k


def brand_by_cluster(
    rows: Sequence[dict],
    labels: Sequence[int],
    brand: str,
) -> dict[int, "object"]:
    """군집별로 이 브랜드가 몇 번 나왔나. Wilson 구간과 함께.

    "3번 군집에서만 우리가 나온다"는 관찰이 표본 4건에서 나온 것이면 아무 말도
    아니다. 군집 분석에서도 비율에는 구간을 붙인다.
    """
    from .stats import wilson_interval

    if len(rows) != len(labels):
        raise ValueError(f"행 {len(rows)}개와 라벨 {len(labels)}개의 길이가 다르다")

    result = {}
    for cluster in sorted(set(labels)):
        members = [r for r, lab in zip(rows, labels) if lab == cluster]
        hits = sum(1 for r in members if brand in r["mentions"])
        result[cluster] = wilson_interval(hits, len(members))
    return result
