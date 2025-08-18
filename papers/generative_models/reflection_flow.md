# 📘 ReflectionFlow: Scaling Inference-Time Self-Refinement in Text-to-Image Diffusion Models

## 1. 개요 (Overview)

* **제목**: From Reflection to Perfection: Scaling Inference-Time Optimization for Text-to-Image Diffusion Models via Reflection Tuning (ReflectionFlow)  
* **저자**: Le Zhuo, Liangbing Zhao, Sayak Paul, Yue Liao, Renrui Zhang, Yi Xin, Peng Gao, Mohamed Elhoseiny, Hongsheng Li  
* **소속**:  
  - CUHK MMLab (Hongsheng Li, Le Zhuo, Yue Liao, Renrui Zhang)  
  - KAUST (Liangbing Zhao, Mohamed Elhoseiny)  
  - Hugging Face (Sayak Paul)  
  - Shanghai AI Lab (Yi Xin, Peng Gao)  
* **학회**: ICCV 2025 (Accepted)  
* **링크**:  
  - [arXiv](https://arxiv.org/abs/2504.16080)  
  - [GitHub](https://github.com/Diffusion-CoT/ReflectionFlow)  
  - [Project Page](https://diffusion-cot.github.io/reflection2perfection/)  
  - [Papers with Code](https://paperswithcode.com/paper/from-reflection-to-perfection-scaling)  

> **논문 선정 이유**  
> 최근 Text-to-Image Diffusion 모델은 빠르게 발전하고 있지만, **추론 단계에서의 품질 향상 전략**은 상대적으로 덜 연구되어 옴.  
> ReflectionFlow는 **inference-time self-refinement**를 제안하며, 단순히 모델 학습을 잘하는 것뿐 아니라, **추론 과정에서 자기 점검(self-reflection)과 보정을 반복**하여 결과를 개선하는 새로운 패러다임을 제시.  
> 특히 **Noise-level / Prompt-level / Reflection-level scaling**이라는 세 가지 축을 통해 구조적이고 체계적으로 품질을 향상시키는 점, 그리고 **1M triplets의 GenRef 데이터셋**과 **227K CoT annotation**을 공개한 점이 주목할 만함.  

---

## 2. 문제 정의 (Problem Formulation)

**문제 및 기존 한계**:

* 기존 Text-to-Image Diffusion 모델은 **고정된 추론 예산(inference budget)** 안에서 이미지를 생성하기 때문에,  
  품질이 기대에 못 미칠 경우 **추론 도중 결과를 보정하거나 개선할 방법이 없음**.
* 단순히 **샘플링 스텝(step) 수를 늘리는 방식**은 연산량만 증가시키고, 의미적 정합성(semantic alignment)이나 세부 디테일 보완에는 한계가 있음.
* 즉, **추론 과정에서의 “자기 점검(self-evaluation)” 및 “자기 개선(self-refinement)”** 메커니즘이 부재.

---

**제안 방식 (ReflectionFlow)**:

* 추론 과정에서 생성된 결과를 **반영(reflection)** → **문제점 식별** → **개선 지침 생성** → **다시 이미지 보정**  
  이라는 루프를 반복하는 **inference-time self-refinement framework**를 제안.
* 이를 위해 세 가지 **scaling 축**을 정의:
  1. **Noise-level scaling**: 초기 노이즈 샘플링 다양화 → 보다 좋은 시작점 탐색  
  2. **Prompt-level scaling**: 입력 프롬프트를 semantically precise하게 보정/강화  
  3. **Reflection-level scaling**: 모델 스스로 생성 결과에 대한 **reflection(피드백)**을 생성 → 개선된 조건으로 다시 생성

---

> ※ **핵심 개념 정의**  
> - **Reflection**: 모델이 스스로 “생성 이미지의 문제점”을 설명하고, 이를 개선하기 위한 지침을 생성하는 과정  
> - **Reflection Tuning**: 이러한 reflection을 학습에 통합하여 모델이 추론 시 reflection을 효과적으로 활용할 수 있도록 하는 학습 전략  
> - **GenRef Dataset**: (flawed image, reflection, enhanced image) 3요소로 구성된 1M triplet 데이터셋 + 227K CoT reflection annotation  

---

## 3. 모델 구조 (Architecture)

### 전체 구조

![reflection_flow](..\images\reflection_flow.png)

**파이프라인 한눈에 보기**

1) **Generator (기본 T2I 모델)**  
   - FLUX.1-dev(12B) 기반으로 초기 후보 이미지를 생성. 필요 시 LoRA 가중치 오프로딩 형태로 불러 사용. 

2) **Verifier & Reflector (평가·피드백 모듈)**  
   - **Verifier**: NVILA-Lite-2B(SANA 1.5의 Verifier)를 사용해 후보 이미지들을 다각도로 채점/순위화.  
   - **Reflector**: Qwen2.5-VL-7B(논문에서 파인튜닝)로부터 **reflection 텍스트**와 **프롬프트 보강안**을 생성. 필요 시 GPT-4o 등 대체도 비교. 

3) **Corrector (Reflection-Tuned DiT)**  
   - 이전 단계의 “이미지 + 원 프롬프트 + 리플렉션 프롬프트”를 **하나의 멀티모달 시퀀스로 결합**하여 **공동 멀티모달 어텐션**(MMDiT)을 수행, 이미지를 **수정/개선**.  
   - 별도 어댑터(예: ControlNet, IP-Adapter) 없이 **텍스트·이미지 토큰을 단일 시퀀스로 결합**하여 처리.

4) **Iterative Refinement**  
   - 위 절차를 **반복**(reflection depth)하며 동시에 **병렬 체인 수**(search width)를 조절해 **추론 예산 ↔ 품질**을 트레이드오프. 마지막에 Best-of-N 선택.

> 핵심 아이디어: **Noise-level / Prompt-level / Reflection-level**의 **3축 추론-시간 스케일링**을 **단일 루프** 안에 통합해, “생성→평가→반영→수정”을 반복함.

---

### 💠 핵심 모듈 또는 구성 요소

#### 📌 Generator (T2I Diffusion Transformer)

* **역할**: 초기 후보 이미지를 샘플링(노이즈 초기화 다양화 포함).  
* **특징**: FLUX.1-dev(Flow-based DiT)를 기본 제너레이터로 채택. 실험 설정에서 **LoRA 오프로딩**으로 초기 샘플 생성 파이프라인을 구성.  
* **포인트**: **Noise-level scaling**으로 다양한 초기 latent에서 출발해 **좋은 시작점**을 탐색.
#### 📌 Verifier (평가자)

* **작동 방식**: 각 후보 이미지를 **다차원 기준**으로 스코어링(정합성, 품질 등) → **랭킹** 및 **체인 선택**에 사용.  
* **구현**: 논문 메인 실험에서 **NVILA-Lite-2B Verifier(SANA 1.5)** 사용, 추가로 자체 Verifier 및 **GPT-4o**도 비교. **Verifier 품질이 최종 성능에 중요**. 

#### 📌 Reflector (피드백 생성기)

* **작동 방식**: (원 프롬프트, 이전 단계 이미지, Verifier 점수) → **Reflection 텍스트** 생성 + **프롬프트 개선**(prompt expansion/rewriting).  
* **구현**: **Qwen2.5-VL-7B**를 GenRef-CoT 등으로 파인튜닝하여 사용; 필요 시 GPT-4o 프롬프트도 제공.
  
#### 📌 Corrector (Reflection-Tuned DiT)

* **역할**: Reflector가 만든 **reflection 텍스트**와 **이전 이미지**를 받아 **수정된 이미지**를 생성.  
* **핵심 설계**:  
  - **공동 멀티모달 어텐션(MMDiT)**: `텍스트(원 프롬프트 ⊕ 리플렉션) ⊕ 이미지 토큰(이전 이미지) ⊕ 타깃(고품질) 컨텍스트`를 **단일 시퀀스**로 결합해 **특화 모듈 없이** 처리.  
  - 수식 개념(간단 표기):  
    - 입력 결합:  `X = concat(tokens(text_orig ⊕ text_reflect), tokens(img_prev), tokens(img_target?))`  
    - 프로젝션:  `Q, K, V = W · X`  
    - 어텐션:  `Attn(X) = softmax(QKᵀ/√d) V`  
    - **의미**: 텍스트와 이미지 간 **양방향 정보 교환**으로 수정 포인트를 정확히 반영.  
  - **LoRA 기반 효율 미세튜닝**: Corrector는 **LoRA (rank=256)** 로 튜닝되어, 대형 DiT를 **추가 모듈 없이** 반영 학습에 적응. 

---

### 🔁 Iterative Refinement 루프 (Test-Time Scaling)

* **Step-by-Step**  
  1. **초기 생성**: Generator가 후보 이미지 세트 생성(Noise-level scaling).  
  2. **평가**: Verifier가 후보들을 채점/정렬.  
  3. **반영/프롬프트 보강**: Reflector가 (문제 진단 + 수정 지시) **reflection**과 **개선 프롬프트** 생성(Prompt-level scaling).  
  4. **수정 생성**: Corrector가 reflection/보강 프롬프트/이전 이미지를 입력으로 **개선 이미지** 생성(Reflection-level scaling).  
  5. **반복**: 필요한 **reflection depth**만큼 2–4를 반복, 마지막에 Best-of-N 선택. **search width**와 **depth**로 예산을 조절. 

---

### 📦 학습·데이터 포인트

* **GenRef-1M**: *(flawed, enhanced, reflection)* **100만 트리플렛** + **GenRef-CoT 227K**(GPT-4o/Gemini 2.0 기반 프로그레시브 반영). **완전 자동 파이프라인(4 소스)** 로 수집.   
* **Reflection Tuning 전략**:  
  - **편집 문제(Formulation as editing)**로 정식화 → **멀티모달 공동 어텐션**만으로 반영 이해/적용이 가능하도록 튜닝.  
  - **추가 모듈 불필요**, **LoRA-256**으로 비용 효율적. 

---

### ✅ 설계 상의 차별점 요약

* (기존) 노이즈 공간 탐색만 확대 vs. (본 논문) **노이즈·프롬프트·리플렉션 3축**을 **단일 루프**로 통합.   
* (기존) 외부 편집 어댑터 의존 vs. (본 논문) **MMDiT 단일 시퀀스 결합**으로 **모듈 추가 없이** 반영-튜닝.   
* (기존) 한 번 생성으로 종료 vs. (본 논문) **반복적 자기 보정**(self-refinement)으로 난이도 높은 프롬프트에서 큰 이득. 

---

## ⚖️ 기존 모델과의 비교

| 항목    | 본 논문 (ReflectionFlow) | 기존 방법1 (Noise-scale Only, 예: CFG-guided SD) | 기존 방법2 (External Editor, 예: ControlNet/IP-Adapter) |
| ----- | ------------------------ | ----------------------------------------- | --------------------------------------------------- |
| 구조    | Generator + Verifier + Reflector + Corrector의 **루프 구조** (self-refinement loop) | 기본 Diffusion 모델에 **샘플링 스텝/노이즈 다양화**만 적용 | 외부 편집 모듈(예: ControlNet)을 붙여 **조건 제어** |
| 학습 방식 | **Reflection Tuning**: (flawed, reflection, enhanced) triplet 데이터 기반 멀티모달 공동 어텐션 학습 | 기존 모델 그대로 사용, 학습 개입 없음 | 별도 어댑터 학습 필요, 멀티모달 공동 학습 아님 |
| 목적    | 추론 시 **자기 반영·보정**을 통한 품질 향상 (노이즈+프롬프트+리플렉션 3축 스케일링) | 더 많은 후보/스텝으로 **샘플 다양성 증가** | 특정 task(구도/포즈/세그멘테이션 등)에 대한 제어 |

---

## 📉 실험 및 결과

* **데이터셋**:
  - **GenRef-1M**: flawed image / reflection / enhanced image 100만 triplets  
  - **GenRef-CoT-227K**: GPT-4o/Gemini 기반 progressive reflection annotation  
  - 평가 데이터: COCO, PartiPrompts, DrawBench 등 표준 T2I 벤치마크  

* **비교 모델**:
  - FLUX.1-dev baseline  
  - 기존 noise-level scaling (샘플링만 증가)  
  - ControlNet, IP-Adapter 등 외부 편집 기반 방법  

* **주요 성능 지표 및 결과**:

| 모델      | Accuracy/정합성 | FID | CLIP-Score | 기타 (주관적 평가) |
| ------- | --------------- | --- | ---------- | ---------------- |
| 본 논문 (ReflectionFlow) | 기존 대비 +5~10% 개선 | 낮음 (better) | ↑ (텍스트-이미지 정합성 강화) | 사람 평가에서도 더 나은 선호도 |
| 기존 SOTA | 기준점 | 상대적 높음 | 기준점 | 일부 실패 케이스 다수 |

> **실험 결과 요약 및 해석**  
> - ReflectionFlow는 단순 노이즈 스케일링보다 **정합성(semantic alignment)**과 **세부 디테일 보정 능력**에서 크게 우세.  
> - 외부 편집 모듈 대비, **추가 네트워크 없이** 동일 또는 더 나은 성능을 보이며 **추론 예산 대비 효율성**이 높음.  
> - 반복(reflection depth ↑) 및 폭(search width ↑)을 조절해 **연산-품질 트레이드오프**를 세밀히 제어 가능.  

---

## ✅ 장점 및 한계

## **장점**:

* **학습 불필요한 기본 모델에도 적용 가능** → 추론 단계에서 바로 성능 향상  
* **Reflection-level scaling** 도입으로, 단순 샘플링 확장 이상의 개선 효과  
* **GenRef-1M** 공개 → T2I self-refinement 연구의 기반 제공  
* **효율성**: LoRA 기반 tuning으로 대규모 모델에 경제적 적용 가능  

## **한계 및 개선 가능성**:

* 반복(reflection depth) 횟수가 많아지면 추론 속도 느려짐  
* Reflection의 품질은 Verifier/Reflector 모델의 성능에 크게 의존  
* 특정 모달리티(예: 영상, 오디오) 확장은 아직 미검증  
* 인간이 기대하는 고차원적 미적 품질은 여전히 한계 존재  

---

## 🧠 TL;DR – 한눈에 요약

> **ReflectionFlow는 T2I diffusion 모델에서 “추론 시 자기 반영(self-reflection)과 수정”을 반복하는 프레임워크를 제안하여, 노이즈·프롬프트·리플렉션의 3축 scaling을 통합적으로 활용함으로써, 기존 대비 효율적이고 강력한 품질 향상을 달성했다.**

| 구성 요소  | 설명 |
| ------ | -- |
| 핵심 모듈  | Generator / Verifier / Reflector / Corrector 루프 |
| 학습 전략  | GenRef-1M triplet 기반 Reflection Tuning (LoRA 효율 미세튜닝) |
| 전이 방식  | 멀티모달 공동 어텐션 (텍스트+이미지 단일 시퀀스) |
| 성능/효율성 | 기존 대비 정합성↑, 세부 디테일 개선, 모듈 추가 없이 효율적 적용 |

---

## 🔗 참고 링크 (References)

* [📄 arXiv 논문](https://arxiv.org/abs/2504.16080)  
* [💻 GitHub](https://github.com/Diffusion-CoT/ReflectionFlow)  
* [📈 Papers with Code](https://paperswithcode.com/paper/from-reflection-to-perfection-scaling)  

---
