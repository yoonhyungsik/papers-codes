# 📘 Improving Image Synthesis with Diffusion-Negative Sampling (DNP)

## 1. 개요 (Overview)

- **제목**: Improving Image Synthesis with Diffusion-Negative Sampling
- **저자**: Alakh Desai, Nuno Vasconcelos
- **소속**: UC San Diego (SVCL)
- **학회**: ECCV 2024
- **링크**: [arXiv](https://arxiv.org) / [arXiv HTML](https://arxiv.org) / [Papers with Code](https://paperswithcode.com)

**선정 이유**: Negative prompt가 효과적이지만 "좋은 negative를 사람이 찾기 어렵다"는 근본 문제를 **semantic gap(인간 vs DM 내부 의미공간)**으로 해석하고, **모델이 직접 만든 '진짜 negative 샘플'(least-compliant)**을 텍스트로 변환해 negative prompt로 쓰는 DNP 파이프라인을 제안한다. 즉, "사람이 negative를 잘 쓰게 훈련시키기"가 아니라 "모델이 납득하는 negative를 먼저 뽑고 인간 언어로 번역"하는 방향.

---

## 2. 문제 정의 (Problem Formulation)

### 문제 및 기존 한계
일반적인 Negative Prompting은 효과가 있으나, 어떤 negative가 좋은지 탐색이 어렵고 프롬프트/시드에 따라 불안정함. 그 이유를 "사람이 생각하는 negative"와 "디퓨전 모델이 내부적으로 분리하는 개념" 사이의 **semantic gap**으로 설명:
- 사람이 직관적으로 쓴 negative는 DM 내부에서 '진짜로' 확률을 낮추는 방향이 아닐 수 있음

### 제안 방식
1. **DNS (Diffusion-Negative Sampling)**: 주어진 positive prompt $p$에 대해 가장 덜 따르는(least-compliant) 이미지 $x^-$를 샘플링하는 절차를 정의
2. **DNP (Diffusion-Negative Prompting)**:
   - DNS로 $x^-$ 생성
   - $x^-$를 사람이(또는 캡셔너가) 설명해 negative prompt $n^*$ 생성
   - 최종 생성은 $(p, n^*)$로 수행

### 핵심 개념 정의
- **Negative Prompting(NP)**: "이미지에 포함되면 안 되는 내용"을 텍스트로 넣어 생성 방향을 제어
- **DNS**: "텍스트를 잘 따르는 샘플"이 아니라, 텍스트를 가장 덜 따르는 샘플을 일부러 뽑는 sampling
- **DNP**: DNS로 얻은 "모델 기준 negative 이미지"를 텍스트로 번역해, 그 텍스트를 negative prompt로 사용

---

## 3. 모델 구조 (Architecture)

### 전체 구조
1. **입력**: positive prompt $p$
2. **DNS Chain**: $p$에 대해 least-compliant 이미지 $x^-$ 생성
3. **Captioning**: $x^- \rightarrow n^*$
4. **NP Chain**: $(p, n^*)$로 최종 이미지 생성

### 💠 핵심 모듈 또는 구성 요소

#### 📌 DNS (Diffusion-Negative Sampling)
- **목표**: $p$에 대한 prompt adherence가 낮은 샘플을 의도적으로 생성
- **직관**: CFG가 "$p$를 더 잘 따르도록" 확률 질량을 이동시키는 반면, DNS는 "$p$를 덜 따르는 쪽"을 강조
- **실무적 관점**: NP를 지원하는 모델에서 (positive를 비우고, negative에 $p$를 넣는 형태로) DNS 효과를 유도할 수 있다고 설명

#### 📌 DNP / auto-DNP
- $x^-$를 사람이 설명하면 $n^*$가 되고, 그게 DM 내부 의미공간에서 더 "진짜 negative"로 작동
- **auto-DNP**: 캡셔너(예: BLIP2)로 $x^- \rightarrow n^*$를 자동화

---

## ⚖️ 기존 모델과의 비교

| 항목 | 본 논문(DNP) | 기존 방법1(CFG) | 기존 방법2(수동 Negative Prompting) |
|------|-------------|----------------|-----------------------------------|
| "네거티브" 원천 | 모델이 만든 least-compliant 샘플에서 유도 | 없음 | 사람이 직접 작성 |
| 추가 모듈 | (auto-DNP면) 캡셔너 필요 | 없음 | 없음 |
| 장점 | DM 내부 의미공간에 맞는 negative를 제공 | 단순/빠름 | 케이스에 따라 강력 |
| 한계 | 캡셔너 오류/정보 손실 가능 | adherence 한계 | 탐색 난이도/불안정 |

---

## 📉 실험 및 결과

### 데이터셋/벤치
논문에서 다루는 범주: 복잡 프롬프트(관계/배치), 인체/손(H&H 류), 다양한 텍스트 벤치

### 비교 모델
- SD/SDXL baseline(CFG)
- 수동 NP
- Attend-and-Excite류 결합 등

### 주요 성능 지표
- CLIP score
- Inception Score(IS)
- Human preference(정확성/품질 선호)

| 모델 | Accuracy | F1 | BLEU | 기타 |
|------|----------|-------|------|------|
| 본 논문(DNP/auto-DNP) | - | - | - | CLIP/IS/사람 선호 개선(보고) |
| 기존 SOTA | - | - | - | baseline/기존 기법 대비 |

### 실험 요약
DNS로 뽑은 negative 샘플을 언어로 번역해 negative로 쓰면, "사람이 떠올린 negative"보다 prompt adherence/품질이 더 안정적으로 개선된다는 것을 정성/정량/사람평가로 보여줌.

---

## ✅ 장점 및 한계

### 장점
- Training-free (학습 없이 적용)
- "좋은 negative를 찾기 어려움"을 모델 내부에서 negative를 찾는 방식으로 구조적으로 해결
- 여러 diffusion 변형/가이던스 기법과 조합 가능

### 한계 및 개선 가능성
- auto-DNP는 이미지→텍스트 변환(캡셔닝) 손실/오류 가능
- DNS + 최종 생성으로 추가 sampling 비용 발생
- negative prompt가 "step 전체에 하나로 고정"되는 구조라 step-wise 최적성은 보장 어려움 (→ ANSWER가 여기를 직접 공격)

---

## 🧠 TL;DR – 한눈에 요약

> 모델이 생각하는 '진짜 negative'(least-compliant 이미지)를 먼저 뽑고, 그걸 텍스트로 바꿔 negative prompt로 써서 생성 품질/정합성을 올린다.

| 구성 요소 | 설명 |
|----------|------|
| 핵심 모듈 | DNS(least-compliant sampling) + DNP(캡션 기반 negative 생성) |
| 학습 전략 | 학습 없음(training-free) |
| 전이 방식 | NP/CFG 지원 DM에 쉽게 결합 |
| 성능/효율성 | 개선 보고, 대신 캡셔너/추가 샘플링 비용 및 캡션 손실 리스크 |

---

## 🔗 참고 링크 (References)

- 📄 [arXiv 논문](https://arxiv.org)
- 📄 [arXiv HTML](https://arxiv.org)
- 📈 [Papers with Code](https://paperswithcode.com)

---

**다음 논문**: ANSWER – DNP의 "캡션 손실/외부 리소스/단일 negative 가정"을 제거하고, 샘플링 루프 내부에서 step-wise negative를 직접 구성하는 방향.
