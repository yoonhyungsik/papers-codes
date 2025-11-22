# 📘 VIEScore: Towards Explainable Metrics for Conditional Image Synthesis Evaluation

## 1. 개요 (Overview)

* **제목**: VIEScore: Towards Explainable Metrics for Conditional Image Synthesis Evaluation
* **저자**: Max Ku, Dongfu Jiang, Cong Wei, Xiang Yue, Wenhu Chen
* **소속**: University of Waterloo, IN.AI Research
* **학회**: ACL 2024 (62nd Annual Meeting of the ACL, Long Papers)
* **링크**: [arXiv](https://arxiv.org/abs/2312.14867) / [GitHub](https://github.com/TIGER-AI-Lab/VIEScore) / [Project Page](https://tiger-ai-lab.github.io/VIEScore/)

> **논문 선정 이유 및 간단한 도입부**  
> * 텍스트-투-이미지, 이미지 편집, 컨트롤, subject 기반 생성 등 **다양한 조건부 이미지 생성 태스크**를 한 번에 평가할 수 있는 프레임워크를 제안한다.  
> * 기존 FID, IS, CLIP-Score, LPIPS, DINO 등은 **스칼라 점수만 주는 블랙박스 metric**이며, “왜 점수가 낮은지/무엇이 문제인지”에 대한 **설명 가능한 평가(explainable evaluation)**가 부족하다.  
> * VIEScore는 GPT-4o 같은 **MLLM을 평가자(metric)**로 사용해, 이미지에 대해 **자연어 rationale + 점수(SC/PQ)를 동시에 출력**하면서, ImagenHub에서 인간 평가자와 비슷한 수준의 상관을 달성한다.

---

## 2. 문제 정의 (Problem Formulation)

### 문제 및 기존 한계

* **자동 평가지표의 한계**
  * FID, IS, CLIP-Score, LPIPS, DINO 등은  
    - 태스크 비특이적(task-agnostic): 텍스트-투-이미지, 편집, subject 유지, 컨트롤 등 서로 다른 요구 조건을 구분하지 못함  
    - 스칼라 점수만 제공: 어떤 부분이 잘못되었는지, 무엇이 개선되어야 하는지에 대한 설명이 없음
* **인간 평가의 한계**
  * ImagenHub 기준, 인간–인간(Human–Human) Spearman 상관이 약 0.5 수준 → 사람끼리도 완전히 일관적이지 않음  
  * 대규모 실험에 사람 평가를 쓰기에는 **비용·시간·주관성 문제**가 큼

### 제안 방식

* **VIEScore (Visual Instruction-guided Explainable Score)** 프레임워크 제안
  * 평가 함수를 다음과 같이 정의:
  
    $f_{\text{VIE}}(I, O, C^\*) = (\text{rationale}, \text{score})$
  
    - $I$: evaluation instruction (어떤 기준으로 평가할지, SC/PQ 정의, 점수 범위 등)  
    - $O$: 생성 또는 편집된 이미지  
    - $C^\*$: 조건 집합 (텍스트 프롬프트, subject 이미지, 원본 이미지, 마스크, 컨트롤 이미지 등)
  * 이 $f_{\text{VIE}}$를 GPT-4o, GPT-4v, Gemini, LLaVA 등의 **MLLM으로 구현**하고, 프롬프트 설계를 통해 SC/PQ에 대한 평가를 유도
* **SC/PQ 분리 + 집계**
  * **SC (Semantic Consistency)**: 텍스트 프롬프트, subject 이미지, 컨트롤 조건 등을 얼마나 잘 따랐는지
  * **PQ (Perceptual Quality)**: 노이즈, 블러, 아티팩트, 이상한 손/텍스트, 전반적 자연스러움 등 이미지 품질
  * SC와 PQ 각각에 대해 여러 개의 서브스코어(0–10)를 MLLM이 출력하도록 유도하고,  
    이를 [0,1]로 정규화 후 **min pooling + 기하 평균(geometric mean)**으로 최종 overall score를 계산

> ※ **핵심 개념 정의**
> * **VIEScore**: 조건부 이미지 생성 결과를 **자연어 rationale + SC/PQ 점수**로 동시에 평가하는 MLLM 기반 metric 프레임워크  
> * **SC (Semantic Consistency)**: 조건(텍스트, subject, 컨트롤 등)을 얼마나 잘 만족시키는지  
> * **PQ (Perceptual Quality)**: 이미지 자체의 시각적 품질  
> * **ImagenHub**: 7개 조건부 생성/편집 태스크(텍스트 생성, 편집, subject, 컨트롤 등)에 대해 인간 레이팅과 모델 출력을 모아둔 벤치마크  
> * **M-H correlation**: metric이 낸 점수와 human rating 간의 Spearman 상관

---

## 3. 모델 구조 (Architecture)

### 전체 구조

![모델 구조](경로)

* **입력**
  * 태스크별로 다른 조건들을 모두 포함:
    - 텍스트 프롬프트, subject 이미지, 원본 이미지, 마스크, 컨트롤 맵 등
  * 함께 주어지는 **인스트럭션 텍스트**에
    - SC/PQ를 어떻게 정의할지
    - 0–10 범위 점수로 SC/PQ의 서브 항목들을 평가하도록 할지
    - 마지막에 JSON 또는 일정한 포맷으로 점수를 출력하라고 명시
* **MLLM 평가자**
  * MLLM은 위 입력을 보고 먼저 **자연어 rationale**(각 항목에 대한 평가 이유)을 생성하고,  
    이어서 SC와 PQ의 서브스코어들을 0–10 범위로 출력
* **후처리**
  * 파이프라인이 MLLM의 출력에서 서브스코어들을 파싱하고 [0,1]로 정규화  
  * SC와 PQ를 각각 min pooling으로 합치고, 최종 overall은 SC와 PQ의 기하 평균 형태로 집계

---

### 💠 핵심 모듈 또는 구성 요소

#### 📌 VIEScore Scorer $f_{\text{VIE}}$: MLLM 기반 평가 함수

* **작동 방식**
  * 입력: $I$ (instruction), $O$ (output image), $C^\*$ (조건 묶음)  
  * 출력:  
    - 자연어 rationale  
    - SC 관련 서브스코어 $\{\alpha_i\}_{i=1}^m$ (0–10)  
    - PQ 관련 서브스코어 $\{\beta_j\}_{j=1}^n$ (0–10)
* **수식적 개념**
  
  $f_{\text{VIE}}(I, O, C^\*) = \big(\text{rationale}, \{\alpha_i\}_{i=1}^m, \{\beta_j\}_{j=1}^n\big)$
  
  * 이후 파이프라인에서 $\alpha_i, \beta_j$를 [0,1] 범위로 정규화하고 집계

#### 📌 SC/PQ Decomposition & Aggregation

* **역할**
  * SC/PQ 각각에 대해 여러 관점의 질문을 던져 **서브스코어를 분해**:
    - SC 예: “프롬프트에 언급된 객체들이 모두 존재하는가?”, “subject identity가 유지되는가?”, “multi-concept이 잘 조합되었는가?”  
    - PQ 예: “이상한 손/텍스트가 있는가?”, “노이즈/블러 수준은 어떠한가?”, “전반적으로 자연스럽게 보이는가?”
* **집계 방식 (bottleneck-style)**
  * 정규화된 서브스코어에 대해:
  
    
    $\text{SC} = \min_i \alpha_i, \quad$
    $\text{PQ} = \min_j \beta_j$
    
  
  * 최종 overall score는 대략
  
    $O = \sqrt{\text{SC} \cdot \text{PQ}}$
  
    와 같은 기하 평균 구조를 사용  
  * **의미**: 한 영역(SC나 PQ)에서 큰 실패가 나면 전체 점수가 함께 내려가는 **병목(bottleneck)** 스타일 metric
* **태스크별 템플릿**
  * 7개 ImagenHub 태스크마다 서로 다른 **프롬프트 템플릿**을 사용하여  
    각 태스크의 핵심 포인트(예: subject 유지, multi-concept 조합, 컨트롤 이미지의 준수 정도 등)를 강조

---

## ⚖️ 기존 모델과의 비교

| 항목      | 본 논문 (VIEScore)                                                                 | 기존 방법1 (CLIP-Score)                            | 기존 방법2 (LPIPS / DINO 등)                                      |
|---------|----------------------------------------------------------------------------------|---------------------------------------------------|------------------------------------------------------------------|
| 구조      | MLLM(텍스트+이미지) + 인스트럭션 기반 SC/PQ 다중 스코어 + 자연어 rationale 출력         | CLIP 텍스트/이미지 인코더 간 코사인 유사도                      | 이미지 피처 거리(Perceptual distance) 또는 subject embedding 유사도 기반 |
| 학습 방식   | 추가 학습/파인튜닝 없이, 프롬프트 설계만으로 사용 가능                                      | 대규모 텍스트-이미지 데이터로 사전학습 후 고정                       | 별도 사전학습 후 고정, 평가 시에는 forward만 사용                               |
| 목적      | 조건부 이미지 생성 전반(생성+편집)에 대한 **task-aware + explainable metric**        | 주로 텍스트-이미지 정렬 정도 측정 (텍스트-투-이미지 중심)          | 이미지 품질/유사도(편집, subject 유지, 컨트롤 등 특정 태스크)에 특화             |

---

## 📉 실험 및 결과

* **데이터셋**:
  * **ImagenHub**: 7개 조건부 이미지 태스크
    - Text-guided Image Generation  
    - Mask-guided Image Editing  
    - Text-guided Image Editing  
    - Subject-driven Image Generation  
    - Subject-driven Image Editing  
    - Multi-concept Image Composition  
    - Control-guided Image Generation
* **비교 모델**:
  * **VIEScore + MLLM 백본**
    - GPT-4o, GPT-4v (0-shot / 1-shot 설정)
    - Gemini-1.5-pro, LLaVA 등 오픈소스/클로즈드 MLLM
  * **기존 metric**
    - CLIP-Score, LPIPS, DINO, CLIP-I 등 태스크별 SOTA 지표들
* **주요 성능 지표 및 결과 (Metric–Human Spearman 상관 기준)**:

| 모델                       | SC corr. | PQ corr. | Overall corr. | 기타 설명                                  |
|--------------------------|----------|----------|---------------|-----------------------------------------|
| 본 논문 (VIEScore, GPT-4o) | ≈ 0.45   | ≈ 0.36   | ≈ 0.40        | Human–Human ≈ 0.45에 근접               |
| Human Raters (H–H)      | ≈ 0.50   | ≈ 0.36   | ≈ 0.47        | 상한선(upper bound) 역할                 |
| CLIP-Score              | ~0 또는 음수 | ~0 또는 음수 | ~0 또는 음수     | Text-guided Generation에서 특히 취약   |
| DINO / LPIPS 등          | 태스크별로 상이 | 태스크별로 상이 | 태스크 의존적       | subject 유지 / control에 강점           |

> **실험 결과 요약 및 해석**
> * GPT-4o/4v 기반 VIEScore는 **텍스트-투-이미지 생성 태스크**에서 인간 상관과 거의 동급 수준의 성능을 보이며,  
>   기존 자동 지표(CLIP-Score, LPIPS 등)를 대부분 상회한다.  
> * **편집(Editing) 태스크**에서는 전반적으로 상관이 낮아지고, 특히 미세한 변경(over-editing, subtle edit)을 잡는 데 어려움을 보인다.  
> * 오픈소스 MLLM(LLaVA 등)은 이 세팅에서 상관이 0에 가깝거나 음수인 경우도 많아,  
>   **“metric으로 쓰기 위해서는 백본 퀄리티가 매우 중요하다”**는 점을 보여준다.

---

## ✅ 장점 및 한계

### **장점**:

* **설명 가능한(Explainable) 평가**
  * 스칼라 점수만 제공하는 기존 metric과 달리, MLLM이 **자연어 rationale**을 함께 제공하여  
    어떤 조건을 잘 따랐고, 어떤 부분이 문제인지 직관적으로 파악 가능
* **태스크 인지 + 조건 인지**
  * 텍스트, subject 이미지, 마스크, 컨트롤 이미지 등을 모두 입력으로 고려하며,  
    태스크별로 다른 인스트럭션 템플릿을 통해 **task-aware metric**을 구현
* **추가 학습 불필요**
  * MLLM을 백본으로 사용하되, **별도 fine-tuning 없이 프롬프트 설계만으로** 새로운 태스크에 바로 적용 가능
* **인간 수준에 가까운 상관**
  * GPT-4o 기반 VIEScore는 전체 태스크 평균에서 M-H 상관 ≈ 0.40을 달성하여,  
    H–H ≈ 0.45에 근접한 성능을 보인다 → 대규모 벤치마크에서 사람 평가를 부분적으로 대체할 가능성

### **한계 및 개선 가능성**:

* **클로즈드소스 MLLM 의존**
  * 좋은 성능은 GPT-4o/4v, Gemini 등 상업용 모델에 의존하며,  
    LLaVA 등 오픈소스 MLLM은 현 시점에서 metric으로 쓰기엔 성능이 부족
* **편집 태스크에 취약**
  * 작은 지역 편집 여부, over-editing, subtle edit 판단에 약해,  
    픽셀 단위/지역 단위에 민감한 DINO, LPIPS 같은 지표와의 **하이브리드 설계**가 필요해 보임
* **프롬프트 민감도 및 비용**
  * 인스트럭션 설계에 따라 성능이 민감하게 변할 수 있고,  
  * 대규모 실험에 적용 시 API 비용과 latency가 기존 metric보다 훨씬 큼
* **Bias / 안전성 문제**
  * 평가의 기준이 MLLM에 내재된 지식과 편향에 의존하기 때문에,  
    LLM의 편향과 환각이 metric에도 그대로 반영될 위험이 있음

---

## 🧠 TL;DR – 한눈에 요약

> **MLLM(예: GPT-4o)을 “평가자”로 사용해,  
> > 조건부 이미지 생성 결과에 대해 자연어 rationale과 SC/PQ 점수를 동시에 내는 explainable metric 프레임워크.  
> > ImagenHub에서 인간 평가자에 가까운 상관(≈0.4)을 달성하지만, 편집 태스크와 오픈소스 MLLM 기반 설정에서는 한계가 뚜렷하다.**

| 구성 요소    | 설명 |
|------------|-----|
| 핵심 모듈    | MLLM 기반 VIEScore Scorer $f_{\text{VIE}}$ + SC/PQ 서브스코어 분해 & bottleneck-style 집계 |
| 학습 전략    | 별도 학습 없이 zero/few-shot 인스트럭션 튜닝으로 사용, ImagenHub는 평가용 레퍼런스 데이터로만 활용 |
| 전이 방식    | 텍스트-투-이미지, subject-driven, 컨트롤, 편집 등 7개 조건부 태스크를 하나의 프레임워크로 통합, 새로운 태스크도 조건/인스트럭션만 정의하면 적용 가능 |
| 성능/효율성  | GPT-4o 기준 M-H 상관 ≈ 0.40 (H–H ≈ 0.45), 기존 CLIP-Score/LPIPS보다 전반적으로 우수하지만, 비용·지연·클로즈드 모델 의존성 존재 |

---

## 🔗 참고 링크 (References)

* [📄 arXiv 논문](https://arxiv.org/abs/2312.14867)
* [💻 GitHub](https://github.com/TIGER-AI-Lab/VIEScore)
* [🌐 Project Page & Leaderboard](https://tiger-ai-lab.github.io/VIEScore/)

## 다음 논문:
