# 📘 LoRA: Low-Rank Adaptation of Large Language Models

---

## 1. 개요 (Overview)

- **제목 (Title)**: LoRA: Low-Rank Adaptation of Large Language Models  
- **저자 (Authors)**: Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Weizhu Chen  
- **소속 (Affiliations)**: Microsoft Research, Carnegie Mellon University  
- **학회/학술지 (Conference / Journal)**: ICLR 2022 (International Conference on Learning Representations)  
- **링크**:  
  - [arXiv](https://arxiv.org/abs/2106.09685)  
  - [GitHub](https://github.com/microsoft/LoRA)  
  - [Papers with Code](https://paperswithcode.com/paper/lora-low-rank-adaptation-of-large-language)

### 📝 논문 선정 이유 및 간단한 도입

최근 대규모 언어 모델(LLM)의 파인튜닝 비용이 급격히 증가하면서, 전체 파라미터를 업데이트하지 않고도 모델 성능을 개선할 수 있는 **PEFT (Parameter-Efficient Fine-Tuning)** 기법이 주목받고 있다.  
**LoRA(Low-Rank Adaptation)**는 이러한 흐름 속에서 등장한 대표적인 접근으로, 기존 weight 행렬을 고정한 채 **저랭크 행렬을 추가 학습**하여 성능을 유지하면서도 **메모리·계산 효율성을 획기적으로 향상**시키는 방법이다.

본 논문은 PEFT 기법 중에서도 가장 단순하고 범용성이 높은 방식으로 평가받는 LoRA를 구조적으로 이해하고, 실험 성능 및 후속연구로의 확장 가능성을 탐색하기 위해 선정하였다.

## 🔧 PEFT란? (Parameter-Efficient Fine-Tuning)

PEFT는 대규모 사전학습 모델의 전체 파라미터를 수정하지 않고,  
**일부 작은 파라미터만 학습해 성능을 개선하는 기법**을 말합니다.

### ✅ 장점
- 전체 모델을 다시 학습할 필요 없음
- 메모리와 계산 비용 절감
- 하나의 기반 모델로 다양한 태스크 적용 가능

### 🔍 대표 기법
- **LoRA**: 저랭크 행렬만 추가 학습
- **Adapter**: 중간에 작은 모듈 삽입
- **Prefix/Prompt Tuning**: 입력 앞에 학습 가능한 벡터 추가

---

## 2. 문제 정의 (Problem Formulation)

### ❗ 문제 및 기존 한계

대규모 사전학습 언어모델(LLM)은 뛰어난 성능을 보이지만,  
**전체 파라미터를 파인튜닝하려면 엄청난 연산 자원과 저장 공간이 필요**합니다.

- 모든 downstream task에 대해 full fine-tuning 시,  
  매번 **수백~수천 MB 크기의 모델 사본을 저장**해야 함
- 연산 비용과 GPU 메모리 요구도 매우 큼
- 특정 task에만 필요한 지식도 전체 모델이 학습하게 됨 → **비효율적**

### 💡 제안 방식: LoRA

LoRA는 기존 weight 행렬을 고정한 채,  
**저랭크 행렬을 학습해 성능을 유지하며 연산·메모리 비용을 대폭 줄이는 방법**입니다.

- 기존 weight W를 수정하지 않고, 추가 행렬 `ΔW = A @ B`만 학습
- 학습 파라미터 수는 기존 대비 **수천 배 감소**
- 성능은 full fine-tuning과 거의 동등하거나 더 우수

### 📌 핵심 개념 정의

- **Low-Rank Adaptation**: 큰 행렬을 저랭크 행렬의 곱으로 근사하여 학습 파라미터 수를 줄이는 방식  
- **Frozen Pretrained Weights**: 사전학습된 모델의 기존 weight는 고정하고 추가 모듈만 학습

---

## 3. 모델 구조 (Architecture)

---

### 🧱 전반적인 구조 개요

LoRA는 기존 Transformer 기반 모델의 파라미터를 **전혀 변경하지 않고**,  
일부 weight 행렬에만 **저랭크 행렬(ΔW = A @ B)을 추가로 학습**하는 방식입니다.

- 기존 모델 구조나 연산 흐름을 **변형하지 않음**
- 학습 시에만 LoRA 모듈이 개입되며, **추론 시에는 제거 또는 병합 가능**
- 모델 크기는 그대로 유지하면서도 **효율적인 task-specific tuning이 가능**

---

### 🔄 입력-출력 흐름 정리

```text
[입력 텍스트 토큰]
      ↓
[Embedding Layer]
      ↓
[Transformer Layer]
      ├── Self-Attention
      │    ├── Wq (LoRA 적용) → A_q @ B_q
      │    └── Wv (LoRA 적용) → A_v @ B_v
      └── FFN (선택적 적용)
      ↓
[Output Hidden State]
      ↓
[Decoder or Task Head]
```
### 💠 핵심 구성 요소 상세

#### 📌 LoRA 모듈: Low-Rank Adaptation

기존의 선형 계층에서는:

$$
y = Wx
$$

LoRA를 적용하면, weight 행렬에 저랭크 행렬 \( \Delta W = BA \)를 더해:

$$
y = (W + \Delta W)x = Wx + BAx
$$

여기서:

- $A \in \mathbb{R}^{r \times d}$
- $B \in \mathbb{R}^{d \times r}$
- $\Delta W = BA$, 단 $r \ll d$

> 보통 rank $r$은 1~8 수준의 소수이며,  
> 이로 인해 LoRA는 **저랭크 근사**를 통해 파라미터 수를 크게 줄일 수 있습니다.

- 원래의 $W$는 **동결(frozen)**되어 학습되지 않고,  
- $A$, $B$만 학습됩니다.

ℹ️ 학습 시에는 $\Delta W$만 업데이트되고,  
추론 시에는 $W + \Delta W$를 사전에 병합(merge)하여 연산 효율을 유지할 수 있습니다.


---

#### 📌 적용 위치

LoRA는 Transformer 내부의 **Self-Attention** 블록에서 다음 projection 계층에 주로 적용됩니다:

- Query Projection: \( W_q \)
- Value Projection: \( W_v \)

선택적으로 Feedforward Layer나 다른 Linear Layer에도 적용 가능하지만,  
논문에서는 성능과 효율 균형을 위해 **\( W_q \), \( W_v \)** 에만 적용하는 것이 일반적입니다.

---

### 🧪 파라미터 수 비교

| 방식              | 학습 파라미터 수            | 설명                          |
|-------------------|------------------------------|-------------------------------|
| Full Fine-Tuning  | 전체 파라미터 수 (예: 300M) | 모든 weight를 학습함         |
| LoRA (r=8 기준)   | 전체의 약 0.1% ~ 1% 수준     | A, B 저랭크 행렬만 학습      |

예시: GPT-2 (124M 파라미터) → LoRA 적용 시 약 0.2M 파라미터만 학습

---

### ⚙️ LoRA 구현 예시 (PyTorch)

```python
class LoRALinear(nn.Module):
    def __init__(self, in_dim, out_dim, r):
        super().__init__()
        self.W = nn.Linear(in_dim, out_dim, bias=False)
        self.A = nn.Linear(in_dim, r, bias=False)
        self.B = nn.Linear(r, out_dim, bias=False)

    def forward(self, x):
        return self.W(x) + self.B(self.A(x))
```
---

### 🔧 LoRA 초기화 전략

논문에서는 $A$, $B$ 행렬의 **초기값을 0으로 설정**하여  
학습 초반에는 $\Delta W = 0$ 이 되도록 설계합니다.

> 즉, 초기에는 기존 weight $W$만 사용되며,  
> 학습이 진행되면서 LoRA 모듈이 점차 영향을 미치게 됩니다.
---

### ✅ 기존 방식과의 구조적 차이점

| 항목               | Full Fine-Tuning | Adapter                        | LoRA                            |
|--------------------|------------------|---------------------------------|----------------------------------|
| 원래 weight 수정    | ✅ O              | ❌ X                             | ❌ X                             |
| 파라미터 수         | 매우 많음         | 적음                            | 매우 적음 (~0.1%)               |
| 연산 병목           | 있음              | 증가 가능                        | 없음 (추론 시 병합 가능)        |
| 구조 변화           | 없음              | 있음 (중간 모듈 삽입 필요)       | 없음 (기존 weight에 덧붙임)     |

---

### 🔄 요약

LoRA는 기존 모델의 구조를 유지한 채,  
특정 선형 계층에 **저랭크 행렬을 삽입**하여  
**파인튜닝의 연산·메모리 효율을 극대화**한 기법입니다.

특히 **Self-Attention의 Query 및 Value projection**에 적용할 경우,  
**성능을 유지하면서도 수백~수천 배 적은 파라미터만 학습**하여  
효율적이고 범용적인 fine-tuning이 가능합니다.

---

---

## 📉 실험 및 결과

### 📊 데이터셋

LoRA는 다양한 자연어 처리(NLP) 태스크에서 실험되었으며, 대표적으로 다음과 같은 데이터셋이 사용되었습니다:

- **MNLI** (Multi-Genre Natural Language Inference)
- **RTE** (Recognizing Textual Entailment)
- **CoLA** (Corpus of Linguistic Acceptability)
- **SQuAD 2.0** (Question Answering)
- **WikiSQL** (Semantic Parsing)
- **LAMBADA** (Language Modeling)
- **OpenWebText2** (Pretraining)

---

### 🤖 비교 모델

- **Full Fine-Tuning**: 전체 weight를 학습
- **Adapter** (Houlsby et al., 2019)
- **Prompt Tuning**, **Prefix Tuning**
- **LoRA (본 논문 제안)**

---

### 📈 주요 성능 지표 및 결과

| 모델       | GLUE 평균 점수 | LAMBADA Perplexity ↓ | SQuAD EM / F1 | 파라미터 수 (백만) |
|------------|----------------|----------------------|----------------|--------------------|
| Full FT    | 89.6           | 41.9                 | 79.3 / 86.8    | 330M               |
| Adapter    | 88.5           | 45.2                 | 77.8 / 85.1    | +7M                |
| LoRA (r=8) | 89.5           | 42.3                 | 79.1 / 86.4    | **+0.3M**           |

> 💡 **LoRA는 파라미터 수를 극단적으로 줄이면서도 Full FT에 가까운 성능을 달성**합니다.

---

### 🧠 실험 결과 요약 및 해석

- **성능 유지**: 대부분의 태스크에서 Full Fine-Tuning과 거의 동일한 성능을 달성
- **압도적인 효율**: Adapter 대비 10~30배 적은 파라미터 수로도 동등한 성능
- **모델 재사용 가능**: Base 모델은 고정되어 하나의 모델로 다양한 태스크를 처리 가능

---

## ✅ 장점 및 한계

### 🌟 장점

- ✅ **파라미터 효율성**: 전체 모델의 0.1~0.5%만 학습
- ✅ **모델 재사용**: 여러 task에 대해 동일한 base model 사용 가능
- ✅ **낮은 연산비용**: GPU 메모리 및 학습 시간 대폭 절감
- ✅ **기존 구조 유지**: Transformer 구조 수정 없음

---

### ⚠️ 한계 및 개선 가능성

- ❌ **적용 위치 선택이 수동**: 어느 계층에 적용할지 사전에 선택해야 함
- ❌ **초기에는 단순한 구조**: Adapter 등보다 구조적 유연성이 낮음
- ❌ **범용성 연구 부족**: 비언어 영역 (예: 비전, 멀티모달) 적용은 이후 연구 필요

> 이후 연구에서는 ViT, Diffusion 등 다양한 분야로 LoRA가 확장되며 이 한계를 극복 중입니다.

---

## 🧠 TL;DR – 한눈에 요약

### 📌 핵심 아이디어 요약

> LoRA는 대규모 사전학습 모델의 선형 계층을 저랭크 근사 행렬로 대체하여,  
> **전체 파라미터를 학습하지 않고도 성능을 거의 유지하는 초경량 파인튜닝 기법**이다.

Transformer 구조는 그대로 유지하면서,  
Query 및 Value 프로젝션에 **\( \Delta W = BA \)** 형태의 저랭크 행렬을 추가하여 학습합니다.  
이로 인해 파라미터 수는 1000분의 1 수준으로 감소하며,  
Full Fine-Tuning에 가까운 성능을 훨씬 적은 비용으로 달성할 수 있습니다.

---

### 🧩 구성 요소별 요약

| 구성 요소       | 설명                                                                 |
|----------------|----------------------------------------------------------------------|
| **핵심 모듈**    | LoRA 모듈: 기존 선형 weight에 \( \Delta W = BA \)를 덧붙이는 구조      |
| **학습 전략**    | 기존 weight는 고정 (Frozen), A, B만 학습 (초기값 0으로 설정)           |
| **전이 방식**    | 원 모델 재사용 + LoRA 모듈만 별도 적용 가능 (Task-specific)            |
| **성능/효율성** | Full FT와 유사한 성능, 파라미터 수는 약 0.1%, 연산량/메모리도 대폭 감소 |

---

## 🔗 참고 링크 (References)

- 📄 **arXiv 논문**: [https://arxiv.org/abs/2106.09685](https://arxiv.org/abs/2106.09685)  
- 💻 **GitHub 구현**: [https://github.com/microsoft/LoRA](https://github.com/microsoft/LoRA)  
- 📈 **Papers with Code**: [https://paperswithcode.com/paper/lora-low-rank-adaptation-of-large-language](https://paperswithcode.com/paper/lora-low-rank-adaptation-of-large-language)

---

## 🔜 다음 논문: **QLoRA**



