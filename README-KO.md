<div align="center">
  <img src="./docs/public/preface.png" width="100%" alt="Learn World Models Banner">
  <br>

[English](./README.md) · [中文](./README-CN.md) · [한국어](./README-KO.md)

# Learn World Models（⚠️ 알파 프리뷰）

[![GitHub Stars](https://img.shields.io/github/stars/datawhalechina/learn-world-model?style=for-the-badge&logo=github)](https://github.com/datawhalechina/learn-world-model/stargazers)
[![라이선스: MIT](https://img.shields.io/badge/라이선스-MIT-yellow?style=for-the-badge)](https://github.com/datawhalechina/learn-world-model/blob/main/LICENSE)

> **잠재 동역학에 대한 직관에서 출발해 실제로 작동하는 시뮬레이션, 계획, 평가 시스템에 이르기까지, 월드모델을 직접 만들어보며 배웁니다.**

### 📖 [**온라인으로 강의 읽기 →**](https://datawhalechina.github.io/learn-world-model)

</div>

> [!CAUTION]
> ⚠️ **알파 프리뷰**입니다. 아직 초기 빌드 단계라 콘텐츠가 계속 보완되고 수정되는 중이며, 섹션, 예시, 문구가 앞으로도 바뀔 수 있습니다. Issue를 통한 피드백을 환영합니다.

---

## ✨ 미리보기

### 🏠 강좌 홈
> 강의와 프로젝트 카드로 구성된 체계적인 학습 경로.

![강좌 홈](./docs/public/screenshots/readme/en-home.png)

### 📖 강의 페이지
> 딥러닝 지식이 있는 독자를 위해 개념부터 설명하며, mermaid 다이어그램과 함께 필요한 배경 지식을 짧게 짚어주는 박스를 곳곳에 넣었습니다.

![강의 페이지](./docs/public/screenshots/readme/en-lecture-01.png)

### 🗂️ 아키텍처 심층 탐구
> 아홉 개 아키텍처 계열, 세 가지 계획 메커니즘, 그리고 나란히 비교할 수 있는 표를 담았습니다.

![아키텍처 강의](./docs/public/screenshots/readme/en-lecture-03.png)

---

## 이 강좌가 다루는 내용

다섯 개의 강의와 여섯 개의 프로젝트를 통해, 월드모델에 대한 직관에서 출발해 현대적인 월드모델 시스템을 학습시키고 평가하며 인과적으로 검증하는 데까지 이릅니다.

| # | 유형 | 제목 | 핵심 주제 |
|---|------|------|-----------|
| L01 | 강의 | 내부 시뮬레이션과 역사적 맥락 | Craik의 심적 모델, 예측 부호화, 월드모델 진화의 네 시대 |
| L02 | 강의 | 관측 인코딩과 잠재 동역학 | VAE, CNN 인코더, ELBO, GRU → MDN-RNN → RSSM |
| L03 | 강의 | 아키텍처 패턴, 학습 패러다임과 계획 | 계획과 제어, 백본 선택, 아홉 개 아키텍처 계열, 최신 연구 동향 정리(선택) |
| L04 | 강의 | 월드모델 진단하기 | 표현, 동역학, 롤아웃, 과제 신호, 계획, 배포의 진단 |
| L05 | 강의 | 최전선 논쟁 | 언어 대 물리 세계 이해, Bitter Lesson, 연구 목표로서의 AGI |
| P01 | 프로젝트 | VAE 인코더 학습 | 64×64 픽셀에 대한 소형 CNN VAE, ELBO 손실 곡선, 잠재 벡터 슬라이더 시각화 |
| P02 | 프로젝트 | RSSM 동역학 모델 구축 | GRU, MDN-RNN, RSSM 비교, 사전·사후 롤아웃 그래프 |
| P03 | 프로젝트 | Dreamer 에이전트 학습 | 인코더 + RSSM + 잠재 Actor-Critic의 전체 학습 루프, 소형 픽셀 환경에서 실행 |
| P04 | 프로젝트 | 동역학 백본 교체 | RSSM을 소형 인과적 Transformer(STORM 방식)로 교체, 아키텍처 비교 |
| P05 | 프로젝트 | 월드모델 평가 대시보드 | FID, 보상 상관관계, PSNR, 잠재 드리프트를 모델별로 나란히 비교 |
| P06 | 프로젝트 | 동작 조건화 반사실적 월드모델 | 개입적/반사실적 롤아웃, 역동역학 정규화, 동작 영향력 지표 |

---

## 커리큘럼 흐름

| 단계 | 먼저 읽기 | 그다음 실습 |
| --- | --- | --- |
| 기초 | L01 | 공통 용어와 역량 계단 구축 |
| 표현 | L02: 관측 인코딩 | P01: VAE 인코더 학습 |
| 동역학 | L02: 잠재 동역학 | P02: RSSM 동역학 모델 구축 |
| 제어 | L03: 계획과 제어 | P03: Dreamer 에이전트 학습 |
| 백본 선택 | L03: 백본 선택 | P04: 동역학 백본 교체 |
| 연구 방향 | L03: 최신 연구 동향(선택) | 선택 읽기, 프로젝트 선행 조건 아님 |
| 진단 | L04: 월드모델 진단하기 | P05: 평가 대시보드, P06: 반사실적 정확도 |
| 열린 질문 | L05 | 아직 해결되지 않은 논쟁 종합 |

먼저 L01과 L02를 P01, P02와 번갈아 학습하고, 이어서 L03을 P03, P04와 함께 익힙니다. 여유가 되면 L03의 최신 연구 동향도 살펴본 뒤, P05와 P06을 진행하며 L04를 활용하고, 마지막으로 L05로 마무리합니다.

모든 이론을 다 읽은 뒤에 프로젝트를 시작할 필요는 없습니다. 먼저 만들어보고, 질문을 안고 다시 돌아오면 됩니다.

---

## 빠른 시작

```sh
npm install
npm run docs:dev        # 핫 리로드가 적용된 개발 서버
npm run docs:build      # 프로덕션 빌드
npm run docs:preview    # 빌드된 사이트 미리보기
```

빌드 후 README 스크린샷을 새로고침하려면 다음을 실행합니다.

```sh
npm run docs:build
npm run screenshots:readme
```

---

## 저장소 구조

```
learn-world-model/
├── docs/                                  # VitePress 문서 사이트
│   ├── .vitepress/config.mts             # 내비게이션과 사이드바(EN + ZH + KO)
│   ├── en/lectures/                       # 영어 강의 모듈 5개
│   ├── zh/lectures/                       # 중국어 강의 모듈 5개
│   ├── ko/lectures/                       # 한국어 강의 모듈 5개
│   ├── en/projects/                       # 영어 프로젝트 페이지 6개
│   ├── zh/projects/                       # 중국어 프로젝트 페이지 6개
│   └── ko/projects/                       # 한국어 프로젝트 페이지 6개
├── external/world-model-tutorial/         # 프로젝트가 참조하는 PyTorch 소스 코드
│   └── references.md                      # 네 시대 역사와 아키텍처 개관
├── scripts/                               # 빌드 유틸리티(스크린샷, PDF)
└── package.json
```

---

## 커뮤니티

QR 코드를 스캔해 위챗 대화방(微信交流群)에 참여하세요.

<div align="center">
  <img src="./docs/public/wechat.png" width="300" alt="위챗 대화방 QR 코드">
</div>

---

## 기여하기

기여를 환영합니다. Pull Request를 제출하기 전에 [CLAUDE.md](./CLAUDE.md)를 먼저 읽어주세요. 여기에는 모든 강의와 프로젝트 파일에 적용되는 작성 규칙(em dash 금지, 단순 선형 mermaid 다이어그램 금지, 화살표 나열식 문장 금지, EN/ZH/KO 동기화 등)이 정리되어 있습니다. 이 규칙을 따르지 않으면 병합 전에 수정을 요청드립니다.

---

## 기여자

| 이름 | 역할 | 소속 | GitHub |
| ---- | ---- | ----------- | ------ |
| Zhimin Zhao | 프로젝트 리드 | Queen's University | [@zhimin-z](https://github.com/zhimin-z) |
| Qi Wang | 프로젝트 리드 | Chinese Academy of Sciences | [@qiwang067](https://github.com/qiwang067) |
| Xun Wang | 기여자 |  | [@wangxunx](https://github.com/wangxunx) |
