---
title: "Dreamer 시리즈 아키텍처의 진화"
description: Dreamer V1에서 V4까지의 단계적 진화, Dreamer에서 인코더가 가교 역할을 하는 방식, 그리고 2강 전체 요약을 다룹니다.
lecture: 2
---

# Dreamer 시리즈 아키텍처의 진화

## Transformer 동역학: GRU에서 시퀀스 모델링으로

GRU의 핵심 한계는 정보 병목에서 비롯되는데, 모든 이력 정보를 고정 차원의 은닉 상태 $\mathbf{h}_t$ 하나에 압축해야 하기 때문입니다. 시퀀스가 길어질수록 초기 정보를 유지하기 어려워지고 장거리 의존성을 쉽게 잃어버리는데, 짧은 비디오 게임 프레임에서는 큰 문제가 아니지만 수십 스텝 전의 사건을 기억해야 올바른 결정을 내릴 수 있는 과제에서는 GRU의 기억 용량이 확실한 제약이 됩니다.

Transformer는 다른 접근을 취해, 하나의 은닉 상태로 이력을 요약하는 대신 잠재 상태의 전체 이력에 대해 직접 어텐션을 수행합니다. 그 덕분에 매 스텝의 예측이 어느 시점의 과거 상태든 "되돌아볼" 수 있어 정보 압축 병목이 없지만, 계산량이 문맥 길이에 따라 늘어나고 추론 시 메모리 사용량도 더 커진다는 대가가 따릅니다. Transformer 셀프 어텐션 메커니즘의 완전한 원리와 수식은 L03의 Transformer 아키텍처 절에서 다룹니다.

STORM(2023)은 RSSM 안의 GRU 백본을 Transformer로 교체해, 긴 시퀀스의 Atari 과제에서 예측 정확도와 정책 리턴 모두 눈에 띄게 향상시켰습니다. Dreamer V4(2025)도 같은 교체를 하고 오프라인 정책 학습과 결합해 장기 상상 궤적을 더 일관되고 신뢰할 수 있게 만들었습니다. L03에서는 RSSM을 기준선으로 삼아 이 두 백본 유형을 서로 다른 과제 제약 아래에서 나란히 비교합니다.


## Dreamer 시리즈의 아키텍처 반복 개선

RSSM은 Dreamer V1이 확립한 기본 아키텍처이며, 이후 세 버전은 그 위에서 단계적으로 진화했고 매 반복은 이전 버전의 구체적인 병목을 겨냥했습니다.

[**Dreamer V1(2019)**](https://arxiv.org/abs/1912.01603)은 이 강의 앞부분에서 설명한 RSSM과 잠재 공간 Actor-Critic의 전체 프레임워크를 확립했으며, 이후 모든 버전의 출발점이 되었습니다.

[**Dreamer V2(2020)**](https://arxiv.org/abs/2010.02193)는 연속적인 가우시안 $\mathbf{z}_t$를 **이산 카테고리컬 잠재 변수**(연속적인 실수 공간에서 샘플링하는 대신, 유한한 범주 중 하나를 선택하는 방식)로 대체하고, **STE**(straight-through estimator, 미분 불가능한 이산 샘플링 연산을 그라디언트가 "그대로 통과"하도록 만드는 기법으로, 순전파에서는 이산 샘플을 사용하고 역전파에서는 이 연산을 항등 함수처럼 취급해 그라디언트가 곧바로 흐르게 합니다)를 이용해 그라디언트를 전파했습니다. 이산 잠재 변수는 두 가지 효과를 낳았는데, 학습 곡선이 눈에 띄게 안정되었고 잠재 공간의 의미 구조도 더 명확해졌습니다. 동역학 백본은 여전히 GRU였고, 정책도 여전히 온라인으로 학습되었습니다.

[**Dreamer V3(2023)**](https://arxiv.org/abs/2301.04104)은 아키텍처가 아니라 학습 레시피를 바꿨습니다. 핵심 기법은 두 가지입니다. **대칭로그 변환**(symlog, 보상 값에 대칭적인 로그 압축을 적용하는 기법: $\text{symlog}(x) = \text{sign}(x) \cdot \ln(|x|+1)$, 크기가 크게 차이 나는 보상들을 비슷한 수치 범위로 압축해 극단적인 보상 값이 그라디언트를 좌우하지 못하게 합니다)은 극단적인 보상 값을 압축하고, **백분위수 정규화**(percentile normalization, 고정된 최댓값/최솟값 대신 보상 분포의 5번째와 95번째 백분위수를 기준으로 스케일을 조정하는 방법으로, 정규화를 이상치에 강건하게 만듭니다)는 보상 스케일을 단위 선택과 무관하게 만듭니다. 그 결과 단 하나의 하이퍼파라미터 세트로 Atari 전체, DMControl, Minecraft를 과제별 튜닝 없이 곧바로 실행할 수 있게 되었습니다. Minecraft에서 다이아몬드를 채굴할 수 있는 에이전트를 처음부터 학습시킨 것이 이 버전의 대표적인 성과이며, 충분히 견고한 학습 레시피만 있다면 GRU 백본에도 아직 활용되지 않은 잠재력이 남아 있음을 보여줍니다.

[**Dreamer V4(2025)**](https://arxiv.org/abs/2509.24527)는 레시피 조정이 아니라 질적인 아키텍처 변화입니다. 동역학 핵심이 GRU에서 **Transformer**로 바뀌면서 월드모델이 더 긴 문맥을 모델링할 수 있게 되었고, 장기 예측 정확도도 함께 향상되었습니다. 정책 학습 방식도 온라인 Actor-Critic에서 **오프라인 정책 학습**(offline policy learning, 정책이 실시간으로 환경과 상호작용할 필요 없이 미리 저장된 궤적 데이터만으로 전적으로 학습되는 방식으로, "온라인" 학습과의 차이는 온라인 학습은 상호작용하면서 갱신하는 반면 오프라인 학습은 고정된 데이터셋만 사용한다는 점입니다)으로 전환되었으며, 정책은 저장된 상상 궤적만으로 전적으로 학습되어 더 이상 온라인 롤아웃에 의존하지 않습니다. 이 설계는 L03에서 소개할 STORM([Zhang et al., 2023](https://arxiv.org/abs/2310.09615))이나 IRIS([Micheli et al., 2022](https://arxiv.org/abs/2209.00588))와 아키텍처 철학 면에서 매우 가까우며, 어떤 의미에서 Dreamer V4는 GRU 진영이 Transformer 진영으로 공식적으로 수렴한 것이라 할 수 있습니다.

| 버전 | 동역학 핵심 | 잠재 변수 유형 | 정책 학습 | 핵심 진전 |
|---------|--------------|---------------------|-----------------|-------------|
| V1 | GRU | 연속 가우시안 | 온라인 Actor-Critic | RSSM 아키텍처 확립 |
| V2 | GRU | 이산 카테고리컬 | 온라인 Actor-Critic | 이산 잠재 변수, 안정적인 학습 |
| V3 | GRU | 이산 카테고리컬 | 온라인 Actor-Critic | 도메인 전반의 단일 하이퍼파라미터, Minecraft 벤치마크 |
| V4 | Transformer | 이산 카테고리컬 | 오프라인 정책 학습 | 아키텍처 전환, 장기 추론 |

각 버전은 시스템 전체를 다시 설계하기보다 이전 버전의 구체적인 병목 하나를 겨냥합니다.

<figure>
<img src="/planet/rssm-diagnostics.png" alt="PlaNet의 개방 루프 상태 진단: 고정된 RSSM 잠재 상태로부터 실제 위치, 속도, 보상을 예측한다" style="width:90%;display:block;margin:0 auto">
<figcaption>Hafner et al.(2019)의 개방 루프 상태 진단 실험. RSSM 동역학 모델을 고정한 뒤, 학습된 잠재 상태로부터 시뮬레이터의 실제 위치, 속도, 보상을 예측하도록 작은 신경망을 학습시킵니다. 이 값들이 논문에서 사용한 계획 호라이즌(planning horizon)보다 더 먼 시점까지 정확하게 예측된다는 것은, 잠재 공간이 기저 시스템에 담긴 정보 대부분을 포착하고 있음을 확인해줍니다.</figcaption>
</figure>


## Dreamer에서 인코더가 가교 역할을 하는 방식

인코더는 단순한 압축 도구 이상으로, 픽셀 세계와 잠재 동역학 세계를 잇는 **가교**입니다. Dreamer 전체 파이프라인은 다음과 같습니다.

1. **인코딩**: $\mathbf{o}_t \xrightarrow{\text{encoder}} \mathbf{z}_t$
2. **동역학**: $(\mathbf{z}_t, \mathbf{a}_t) \xrightarrow{\text{RSSM}} \mathbf{z}_{t+1}, \mathbf{z}_{t+2}, \ldots$(순수 상상)
3. **정책 학습**: 실제 환경과 상호작용하지 않고 상상한 궤적으로 Actor-Critic을 학습
4. **실행**: 정책을 실제 환경에 적용해 소량의 새 샘플을 수집하고 이를 반복

인코더의 품질이 RSSM의 상한선을 직접 좌우하는데, 잠재 공간이 의미적으로 명확할수록 동역학 모델이 의미 있는 전이 패턴을 학습하기가 더 쉬워집니다.


## 요약

| 개념 | 역할 | 핵심 방정식/구조 |
|---------|------|--------------------------|
| VAE 인코더 | 픽셀을 $\mathbf{z}$로 압축 | ELBO = 재구성 손실 − KL 발산 |
| GRU 동역학 | 다음 상태를 결정론적으로 예측 | $\mathbf{z}_{t+1} = \text{GRU}(\mathbf{z}_t, \mathbf{a}_t)$ |
| MDN-RNN | 다봉 불확실성 모델링 | 가우시안 혼합 출력 분포 |
| RSSM | 결정론적/확률적 상태 분리 | $\mathbf{h}_t$(기억) + $\mathbf{z}_t$(지각) |
| Transformer 동역학 | 고정 은닉 상태 대신 전역 어텐션 | $\mathbf{h}_t = \text{Attention}(\mathbf{z}_{1:t}, \mathbf{a}_{1:t-1})$ |
| Dreamer 시리즈 | V1에서 V4까지의 단계적 진화 | GRU에서 Transformer로, 연속에서 이산 잠재 변수로, 온라인에서 오프라인 정책으로 |

좋은 월드모델은 좋은 인코더(지각 압축)와 좋은 동역학 모델(시간적 예측)을 합한 것이며, RSSM은 두 종류의 상태(결정론적 $h_t$와 확률적 $z_t$)를 분리함으로써 표현력과 연산 효율 사이에서 우아한 균형을 이룹니다. 네 개의 Dreamer 버전에 걸친 진화는 아키텍처 자체를 넘어 잠재 변수의 종류와 학습 레시피 역시 그에 못지않게 결정적인 요인임을 보여줍니다.


## 다음 강의

L03의 질문은 RSSM이 유일한 선택지는 아니라는 것입니다. Transformer를 백본으로 활용하는 월드모델(STORM, IRIS)은 긴 시퀀스 과제에서 어떤 성능을 보이며, Transformer로 전환한 뒤 Dreamer V4는 이들과 비교해 어디쯤 서 있을까요?

P01과 P02를 마치고 나면 여러분은 작동하는 RSSM 기준선을 갖게 되는데, L03은 이를 기준점으로 삼아 Transformer 동역학, 확산 모델, JEPA를 포함한 아홉 개 아키텍처 계열을 나란히 비교하고 Dreamer V4가 그 지도 위 어디에 위치하는지 설명합니다. 이 비교는 우열을 가리는 순위가 아니라, 서로 다른 과제 제약에서 각 아키텍처가 어디에 적용되는지를 보여주는 지도입니다.


## 더 읽을거리

- [Kingma & Welling (2014): Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114): VAE 원 논문, ELBO 유도와 재매개변수화 트릭
- [Ha & Schmidhuber (2018): World Models](https://arxiv.org/abs/1803.10122): MDN-RNN 동역학 모델과 꿈속 학습 프레임워크
- [Hafner et al. (2019): PlaNet / RSSM](https://arxiv.org/abs/1811.04551): 결정론적+확률적 이중 경로 잠재 동역학, RSSM 최초 제안
- [Hafner et al. (2019): Dream to Control (Dreamer V1)](https://arxiv.org/abs/1912.01603): RSSM과 잠재 Actor-Critic, 엔드투엔드 Dreamer 원 논문
- [Hafner et al. (2020): Mastering Atari with Discrete World Models (Dreamer V2)](https://arxiv.org/abs/2010.02193): 이산 잠재 변수와 STE(straight-through gradient estimator)
- [Hafner et al. (2023): Mastering Diverse Domains with World Models (Dreamer V3)](https://arxiv.org/abs/2301.04104): 과제 전반의 통일된 하이퍼파라미터, 안정적인 학습을 위한 대칭로그 변환
- [Hafner et al. (2025): Dreamer V4](https://arxiv.org/abs/2509.24527): GRU를 대체하는 Transformer 백본, 오프라인 데이터 사전학습
