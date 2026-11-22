---
title: "논문별 비교와 원문"
description: "두 논문을 펼쳐 미래 예측이 어느 시점에 제어에 반영되는지 비교해 보세요."
lecture: 6
---

# 논문별 비교와 원문

두 논문을 펼쳐 미래 예측이 어느 시점에 제어에 반영되는지 비교해 보세요.

## 두 방법 비교

<details>
<summary>Fast-WAM · Motus</summary>

Fast-WAM은 어텐션 마스크 덕분에 실행 시 미래 영상 입력을 생략합니다. Motus는 미래 영상을 순수 잡음으로 고정하고 행동의 잡음 제거만 진행해 직접 제어 명령을 생성할 수 있습니다.

</details>

<details>
<summary>Fast-WAM · LingBot-VA</summary>

Fast-WAM은 카메라 입력을 받은 뒤의 계산을 줄입니다. LingBot-VA는 실행 중인 행동과 다음 계산을 겹치고 피드백으로 다음 구간의 시작점을 보정합니다.

</details>

<details>
<summary>DreamZero · Fast-WAM</summary>

Fast-WAM의 행동은 시각 정보로 첫 프레임만 참조하므로 미래 영상 토큰을 생략할 수 있습니다. DreamZero는 매 반복에서 같은 구간의 영상과 행동을 함께 갱신합니다.

</details>

<details>
<summary>Cosmos Policy · Fast-WAM</summary>

Fast-WAM은 행동을 내기 전 미래 영상 계산을 생략합니다. Cosmos Policy는 선택지가 남아 있는 동안 여러 후보의 결과를 비교하는 데 계산을 쓸 수 있습니다.

</details>

<details>
<summary>LingBot-VA · Motus</summary>

Motus의 역동역학 모드에는 시각 시퀀스가 주어집니다. LingBot-VA는 다음 영상 구간도 직접 예측하며 구간 사이에 실제 피드백을 넣어 제어 루프를 닫습니다.

</details>

<details>
<summary>DreamZero · Motus</summary>

Motus는 라벨 없는 영상의 광류에서 잠재 행동을 추출합니다. DreamZero는 다른 로봇이나 사람의 영상을 시각 예측에 쓰고 목표 로봇의 궤적으로 행동을 지도합니다.

</details>

<details>
<summary>Cosmos Policy · Motus</summary>

둘 다 주어진 행동의 결과를 예측할 수 있습니다. Cosmos Policy는 종점을 가치 함수에 입력해 같은 관측에서 나온 여러 후보를 비교합니다.

</details>

<details>
<summary>DreamZero · LingBot-VA</summary>

두 모델 모두 카메라 관측으로 문맥을 갱신할 수 있습니다. LingBot-VA는 영상 변화를 먼저 예측하고 제어를 추론합니다. DreamZero는 한 반복 안에서 영상과 행동을 함께 갱신합니다.

</details>

<details>
<summary>Cosmos Policy · LingBot-VA</summary>

LingBot-VA는 피드백으로 현재 실행 중인 동작의 종점을 추정합니다. Cosmos Policy는 실행할 후보를 선택하기 전에 여러 동작의 종점을 예측합니다.

</details>

<details>
<summary>Cosmos Policy · DreamZero</summary>

DreamZero는 영상과 함께 한 번에 생성한 행동을 실행합니다. Cosmos Policy의 계획 모드는 여러 행동을 제안하고 각각의 결과를 예측해 가치 순으로 정렬합니다.

</details>

## 논문 원문

- [Fast-WAM: Do World Action Models Need Test-time Future Imagination?](https://arxiv.org/abs/2603.16666)
- [Motus: A Unified Latent Action World Model](https://arxiv.org/abs/2512.13030)
- [Causal World Modeling for Robot Control](https://arxiv.org/abs/2601.21998)
- [World Action Models are Zero-shot Policies](https://arxiv.org/abs/2602.15922)
- [Cosmos Policy: Fine-Tuning Video Models for Visuomotor Control and Planning](https://arxiv.org/abs/2601.16163)

글의 구성은 다음 해설을 참고했습니다. [Nemo’s Diffusion RL guide](https://knightnemo.github.io/blog/posts/diffusion-rl-zh/).
