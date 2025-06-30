# 📘 [What’s in the Image? A Deep‑Dive into the Vision of Vision Language Models]

## 1. 개요 (Overview)

- **제목**: What’s in the Image? A Deep‑Dive into the Vision of Vision Language Models  
- **저자**: Yaniv Kaduri, Hila Chefer, Idan Schwartz, Shai Bagon, Tali Dekel  
- **소속**: Weizmann Institute of Science, NVIDIA Research, Bar-Ilan University  
- **학회**: arXiv preprint, submitted November 2024  
- **링크**: [arXiv](https://arxiv.org/abs/2411.17491) / [GitHub](https://github.com/yanivkaz/vlm_vision_analysis) / [Papers with Code](https://paperswithcode.com/paper/what-s-in-the-image-a-deep-dive-into-the)

> **논문 선정 이유 및 간단한 도입부**  
> 최근 Vision-Language Model(VLM)의 활용이 급격히 확산되면서, 이러한 모델들이 **이미지를 어떻게 해석하고 언어로 전환하는지**에 대한 근본적인 질문이 중요해졌다.  
> 이 논문은 VLM 내부의 **어텐션 구조와 정보 흐름을 정량적으로 분석**하여, 이미지 정보가 텍스트 응답에 어떤 방식으로 영향을 주는지를 심층적으로 파헤친다.  
> 시각적 정보를 모델이 어디서, 어떻게 처리하는지 명확히 이해하고자 이 논문을 선정했다.


## 2. 문제 정의 (Problem Formulation)

**문제 및 기존 한계**:

- 기존 Vision-Language Models (VLMs)는 뛰어난 성능을 보이고 있으나, **시각 정보가 내부적으로 어떻게 언어 정보로 변환되는지**는 명확하게 분석되지 않았다.
- 대부분의 연구는 VLM의 결과물(텍스트 응답)에 초점을 맞추며, **내부 메커니즘—특히 어떤 층에서 시각 정보가 통합되는지**에 대한 실증적인 분석은 부족했다.
- 또한, 이미지와 텍스트 간의 정보 흐름이 **어느 층을 통해 어떤 방식으로 이동하는지**, **어떤 토큰이 시각 정보를 주로 활용하는지** 등에 대한 구체적인 설명이 없었다.

**제안 방식**:

- 본 논문은 **Transformer 내부의 attention 흐름을 정량적으로 분석**하여 시각 정보가 어떻게 언어 응답에 반영되는지를 조사한다.
- 이를 위해 **Layer-wise Attention Blocking** 기법을 도입하여, 특정 층에서 vision token을 차단한 뒤 언어 생성 결과를 비교 분석한다.
- 다양한 프롬프트 유형(예: open-ended description, object-attribute 추론)에 대해 **어떤 토큰이 어느 위치의 이미지 정보를 주로 활용하는지**를 평가한다.

> ※ **핵심 개념 정의**:
> - **Vision Token**: 이미지 입력이 patch 단위로 분할되어 transformer에 입력되는 시각 정보 단위.
> - **Query Token**: "이미지를 설명해줘", "이 장면은 어디냐?" 등의 프롬프트에 포함된 언어 토큰으로, 시각 정보 요약과 전달의 핵심 역할을 한다.
> - **Attention Blocking**: 특정 층에서 이미지 토큰에 대한 어텐션 접근을 강제로 막아, 정보 흐름의 위치와 역할을 평가하는 기법.

---

## 3. 모델 구조 (Architecture)

### 전체 구조

![모델 구조](경로)

- 본 논문은 새로운 모델을 제안하는 것이 아니라, 기존 VLM 아키텍처(ex: BLIP-2, LLaVA 등)의 내부 동작을 **분석 및 계층별 시각 정보 흐름 측면에서 해석**하는 실험 중심 구조를 갖는다.
- 실험 대상이 된 VLM은 일반적으로 다음과 같은 구조를 따른다:

```text
[Image Encoder (e.g. ViT)] → [Query/Prompt Token Embedding] → [Transformer Decoder (Language Model)] → [Text Output]
```

- **입력 흐름**:
  - 이미지 → Patch → Vision Encoder → Vision Token (V)
  - 프롬프트 → Tokenizer → Text Token (T)
- **출력 흐름**:
  - Language Transformer → Predict next token → Generate caption or response

- 이 구조 위에 저자들은 특정 실험 기법을 삽입하여 내부 정보 흐름을 추적한다.

---

### 💠 핵심 모듈 또는 구성 요소

#### 📌 Attention Blocking Layer (ABL)

- **작동 방식**:  
  특정 transformer 층 `L_i`에서 **image token → text token 간 attention을 0으로 마스킹**.  
  즉, 해당 층에서 이미지 정보가 텍스트 토큰으로 전달되지 못하도록 차단함.
  
- **수식적으로 표현하면**:  
  Attention matrix `A`에서 vision token 관련 attention weight `A[v, t]`를 `0`으로 설정  
$A[v, t] = 0 \quad (v: \text{vision token}, \ t: \text{text token})$

- **사용 목적**:
- 어떤 층이 이미지 정보를 전달하는지 분석
- 특정 층만 시각 정보 접근 가능하게 만들고 나머지는 차단

---

#### 📌 Vision-Query Token Dependency Analyzer

- **역할**:  
생성된 텍스트 토큰(예: "red bike")이 **어떤 vision token에 attention을 집중하는지**를 분석  
→ 이를 통해 어떤 텍스트가 어떤 이미지 위치에 대응되는지 파악 가능

- **기존 방식과의 차별점**:
- 단순 attention 시각화가 아니라, **단어 단위로 image-patch 연결을 정량 분석**함
- "object-attribute grounding"이라는 측면에서 fine-grained하게 평가

---

#### 📌 Prompt Ablation Experiment

- **작동 방식**:  
프롬프트의 일부(예: “in this image,” “describe”)를 제거하고 성능을 비교  
→ **query token 자체가 시각 정보를 내포하는지** 검증

- **의의**:  
- VLM이 프롬프트 구조 자체에 학습되어, 시각적 정보를 해당 토큰에 암묵적으로 인코딩하고 있음  
- 즉, **텍스트 토큰이 이미지 요약 정보를 보유**한다는 것을 실험적으로 입증

---

### 🔍 어텐션 구조 분석 (Detailed Attention Flow Analysis)

#### 📌 1. Cross-Modal Attention 정의

VLM의 핵심은 Vision Token ($V$)과 Text Token ($T$) 간의 **Cross-Attention**입니다.  
Transformer의 self-attention은 원래 동종 토큰끼리 정보를 주고받지만, VLM에서는 서로 다른 modality 간 attention이 존재합니다.

- 일반 self-attention: $A[t_i, t_j]$
- cross-modal attention: $A[t_i, v_j]$ 또는 $A[v_i, t_j]$

#### 📌 2. 실험 설계: Layer-wise Attention Blocking (ABL)

저자들은 각 Transformer Layer $L_i$에서 다음 중 하나를 선택적으로 차단합니다:

1. **Text-to-Vision 차단**:\
   $A[t_i, v_j] = 0 \quad \text{for all } i, j$
   → 텍스트 토큰이 비전 토큰에서 정보를 가져오지 못하게 함

3. **Vision-to-Text 차단**:  
   $A[v_i, t_j] = 0 \quad \text{for all } i, j$
   → 비전 토큰이 텍스트로 정보 전달하지 못하게 함

4. **Vision-to-Vision 유지 / Text-to-Text 유지**는 그대로 두고 cross-modal 경로만 막음

- 이를 통해 **어느 층에서 이미지 정보가 효과적으로 언어로 전달되는지 계층별로 분석**함

#### 📌 3. 주요 관찰 결과

- **중간층 (~25%~50%)**에서 cross-modal attention이 가장 활발하며, 실제로 이 층에서만 vision access를 허용해도 성능 손실이 거의 없음.
- 초/후반층의 attention은 vision token에 대한 집중도가 낮고, 거의 대부분의 시각 정보가 **중간 층에서 집중적으로 흘러들어감**.
- 프롬프트 token (e.g., “describe”, “in this image”)에 해당하는 text token은 거의 모든 vision token에 **low-entropy attention**을 분산적으로 보내며, **전역적 시각 요약을 수행**하는 것으로 보임.

#### 📌 4. Attention Map 시각화

저자들은 실제 attention map을 다음과 같이 분석:

- 특정 단어(예: “red” 혹은 “bike”)를 생성하는 token이 어떤 image patch (vision token)에 attention을 집중했는지 추적
- 해당 token의 attention vector에서 가장 높은 weight를 갖는 top-k vision token을 추출하여 **spatial grounding** 수행

→ 이는 object attribute grounding 분석에 활용됨 (예: “blue shirt”라는 표현이 실제로 파란 셔츠 위치에 attention 집중)

#### 📌 5. 수식 기반 정리

- Attention Weight:  
  $\alpha_{t_i, v_j} = \frac{\exp(q_{t_i} \cdot k_{v_j})}{\sum_{j'} \exp(q_{t_i} \cdot k_{v_{j'}})}$
  - $q_{t_i}$: 텍스트 토큰의 쿼리 벡터
  - $k_{v_j}$: 이미지 토큰의 키 벡터

- Blocking은 이 $\alpha_{t_i, v_j}$ 값을 직접 0으로 만드는 방식으로 수행됨.

---
## ⚖️ 기존 모델과의 비교

| 항목    | 본 논문                                         | 기존 방법1 (LLaVA)                        | 기존 방법2 (BLIP-2)                          |
| ------- | ---------------------------------------------- | ---------------------------------------- | ------------------------------------------- |
| 구조    | 기존 VLM 위에 attention-blocking 실험 설계 추가 | Vision encoder + LLM                     | Q-former + Vision encoder + LLM              |
| 학습 방식 | 기존 모델 기반 zero-shot or finetuned 사용       | Stage-wise pretraining                   | Pretrained Q-former + Frozen ViT + LLM       |
| 목적    | 시각 정보가 어느 층, 어떤 토큰에 반영되는지 분석     | 멀티모달 문장 생성 (이미지 설명, QA 등)      | 시각 정보 기반 질문응답 및 caption 생성        |

---

> **실험 결과 요약 및 해석**  
> - Vision token access를 오직 중간층에만 허용해도 성능이 거의 유지됨 → **중간층이 주요 정보 전달 경로**  
> - Query token (e.g., “describe”)은 이미지 전체에 대한 low-entropy attention을 보내며, **전역적 시각 요약 담당**  
> - 특정 텍스트 단어는 해당 object 위치의 patch에 attention 집중 → **object-attribute grounding 확실**

---

## ✅ 장점 및 한계

### **장점**:

- VLM 내부의 정보 흐름을 **실험적으로, 계층적으로 검증**한 최초의 정량적 연구 중 하나
- Query token이 실제로 전역 이미지 정보를 요약한다는 **새로운 시사점**
- 향후 경량화 또는 pruning 시 **중요한 층만 남기는 구조 최적화 방향 제시 가능**

### **한계 및 개선 가능성**:

- 학습 자체를 다루지 않고 **기존 모델에 분석 실험을 추가**한 구조 → 구조 자체의 혁신은 아님
- 실험 범위가 제한적이며, multimodal generation 전반으로 확장 필요
- attention weight만으로 정보 흐름을 추론하는 데 한계가 존재할 수 있음

---

## 🧠 TL;DR – 한눈에 요약

> 이 논문은 Vision-Language Models 내부에서 **시각 정보가 언어로 어떻게 전달되는지를 계층적으로 분석**한다.  
> 특히 중간 Transformer 층에서 vision token의 정보가 query token에 집중 전달되며, object-level grounding은 텍스트 생성에 강한 공간적 연결성을 갖는다.  
> 이를 통해 **VLM 해석 가능성, 최적화 가능성, 전역 vs. 국부 attention 구조**에 대한 이해를 제공한다.

| 구성 요소     | 설명 |
| ------------ | ---- |
| 핵심 모듈     | Attention Blocking Layer, Prompt Ablation, Attention Map 분석 |
| 학습 전략     | 기존 pretrained 모델 활용, 추가 학습 없이 실험적 구조 적용 |
| 전이 방식     | Vision-to-Text 정보 흐름 추적 (특히 Query token 중심으로) |
| 성능/효율성  | 중간층만 활용해도 성능 유지 → 계산량 줄이면서 구조 경량화 가능성 있음 |

---

## 🔗 참고 링크 (References)

* [📄 arXiv 논문](https://arxiv.org/abs/2411.17491)
* [💻 GitHub](https://github.com/yanivkaz/vlm_vision_analysis)
* [📈 Papers with Code](https://paperswithcode.com/paper/what-s-in-the-image-a-deep-dive-into-the)

---

## 다음 논문:
-"LLaVA: Large Language and Vision Assistant" (2023)
