# 📘 BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding

## 1. 개요 (Overview)

* **제목**: BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding  
* **저자**: Jacob Devlin, Ming-Wei Chang, Kenton Lee, Kristina Toutanova  
* **소속**: Google AI Language  
* **학회**: arXiv (2018)  
* **링크**: [arXiv](https://arxiv.org/abs/1810.04805) / [GitHub](https://github.com/google-research/bert) / [Papers with Code](https://paperswithcode.com/paper/bert-pre-training-of-deep-bidirectional)

> **논문 선정 이유**:  
> ChatGPT를 포함한 다양한 최신 언어 모델들이 BERT 구조를 기반으로 하고 있다는 점에서 흥미를 느꼈고,  
> 또한 현재의 대형 언어 모델(LLM) 발전의 출발점이 된 모델이라는 점에서 본 논문을 선정하게 되었다.

---

## 2. 문제 정의 (Problem Formulation)

**문제 및 기존 한계**:
* 기존 언어 표현 사전학습 방식은 **feature-based 방식(예: Word2Vec, ELMo)**과 **fine-tuning 방식(예: GPT)**으로 나뉜다.
* Feature-based 방식은 사전학습된 표현을 고정된 입력으로 사용하고, fine-tuning 방식은 사전학습 모델 전체를 미세 조정한다.
* 두 방식 모두 양방향 문맥 표현의 한계가 있으며, 이로 인해 복잡한 문맥 이해가 필요한 작업에서 성능이 제한된다.

**제안 방식**:
* 본 논문은 Transformer 기반 양방향 인코더 구조를 사전학습하여, 문맥의 좌우 정보를 동시에 반영할 수 있는 BERT(Bidirectional Encoder Representations from Transformers)를 제안함.
* 사전학습 단계에서는 문맥 이해 능력을 학습하기 위해 **Masked Language Modeling(MLM)**과 **Next Sentence Prediction(NSP)**이라는 두 가지 목표 함수를 함께 사용.

> **핵심 개념 정의**

> **Masked Language Modeling (MLM)**:  
> 전체 입력 문장에서 무작위로 선택된 일부 토큰(약 15%)을 `[MASK]`로 가리고,  
> 문장의 좌우 문맥을 함께 고려해 해당 단어를 예측하도록 학습하는 방식.  
> 이는 일종의 "빈칸 채우기" 문제이며, 불용어도 포함됨으로써 문법 구조까지 학습 가능.

> **Next Sentence Prediction (NSP)**:  
> 두 문장이 주어졌을 때, 두 번째 문장이 첫 번째 문장의 실제 다음 문장인지 아닌지를 맞추는 이진 분류 과제.  
> 문장 간 연속성 학습을 위한 목적이지만, 후속 연구에서는 표면적 통계에 의존한다는 비판을 받았고  

---

## 3. 모델 구조 (Architecture)

![BERT Architecture](/papers/images/bert_architecture.png)

### 전체 구조

![Input Format](/papers/images/bert_input.png)

* BERT는 **Transformer Encoder만으로 구성된 다층 구조**로,  
  입력 문장을 처리하여 **양방향 문맥 정보를 반영한 토큰 표현**을 생성한다.  
* 입력은 `[CLS]` + 문장 A + `[SEP]` + 문장 B + `[SEP]` 형태로 구성된다.

  ex) [CLS] The man went to the store [SEP] He bought some apples [SEP]

- NSP (Next Sentence Prediction): 두 문장이 실제 이어지는지 분류  
- MLM (Masked Language Modeling): `He bought [MASK] apples` → "some" 예측

* 각 토큰은 다음 세 가지 임베딩의 합으로 표현된다:  
**Token + Segment + Position Embedding**

* 출력은 두 가지로 나뉜다:
- `[MASK]` 위치에서의 단어 예측 → **MLM**
- `[CLS]` 토큰의 출력 벡터를 통한 문장 관계 분류 → **NSP**

---

### 💠 핵심 모듈 또는 구성 요소

---

#### 🔹 Input Representation

## 🔹 입력 임베딩 구성

BERT의 입력 임베딩 \( E_i \)는 세 가지 임베딩의 합으로 구성됩니다:

$$
E_i = E_i^{\text{token}} + E_i^{\text{segment}} + E_i^{\text{position}}
$$

- $E_i^{\text{token}}$: 각 토큰의 단어 임베딩  
- $E_i^{\text{segment}}$: 문장 구분 임베딩 (Segment A/B)  
- $E_i^{\text{position}}$: 위치 정보를 나타내는 포지셔널 임베딩

입력 시퀀스 전체는 다음과 같은 임베딩 행렬로 표현됩니다:

$$
X \in \mathbb{R}^{n \times d}
$$
- \( n \): 시퀀스 길이 (토큰 수)  
- \( d \): 임베딩 차원


입력 시퀀스는 $$\( X \in \mathbb{R}^{n \times d} \)$$의 임베딩 행렬로 구성됨.

---

## 🔹 Transformer Encoder (총 $L$층)

각 레이어는 다음 두 모듈로 구성됩니다:

### 1. Multi-Head Self-Attention

$$
\text{Attention}(Q, K, V) = \text{softmax} \left( \frac{QK^\top}{\sqrt{d_k}} \right)V
$$

> 각 토큰이 문맥 내 다른 모든 토큰과의 관계를 학습

### 2. Position-wise Feed-Forward Network

$$
\text{FFN}(x) = \text{ReLU}(xW_1 + b_1)W_2 + b_2
$$

모든 연산은 **Residual Connection**과 **Layer Normalization**을 포함한 다음 구조로 이루어집니다:

$$
H^{(l+1)} = \text{LayerNorm}\left(H^{(l)} + \text{FFN}\left(\text{Attention}(H^{(l)})\right)\right)
$$

---

## 🔹 Masked Language Modeling (MLM)

- 전체 입력 토큰 중 15%를 무작위로 선택하여 `[MASK]`, 랜덤 토큰, 또는 그대로 유지
- 목표: 마스킹된 위치 \( i \)에서의 단어 \( x_i \)를 양방향 문맥 기반으로 예측

$$
\max_\theta \sum_{i \in \mathcal{M}} \log P_\theta(x_i \mid x_{\setminus i})
$$

> 여기서 \( \mathcal{M} \)은 마스킹된 위치들의 집합이며,  
> \( x_{\setminus i} \)는 \( x_i \)를 제외한 나머지 입력을 의미

- 이를 통해 **양방향 contextual embedding**을 학습 가능

---

#### 🔹 Next Sentence Prediction (NSP)

- 입력 문장 쌍 (A, B)에 대해, B가 실제 다음 문장인지 분류하는 이진 과제  
- [CLS] 토큰의 출력 벡터 \( h_{\text{[CLS]}} \)를 기반으로 softmax 분류 수행

$$
P(y \mid A, B) = \text{softmax}(W h_{\text{[CLS]}} + b)
$$

- NSP는 문장 간 의미적 연결 학습을 유도하나,  
**후속 연구(RoBERTa 등)**에서는 해당 방식의 효과성에 의문을 제기함

---



---

## ⚖️ 기존 모델과의 비교

| 항목         | BERT                        | GPT                         | ELMo                      |
|--------------|-----------------------------|-----------------------------|---------------------------|
| 구조         | Transformer (Encoder Only)  | Transformer (Decoder)       | BiLSTM                    |
| 문맥 방향성  | 양방향 (Bidirectional)      | 단방향 (Left-to-Right)      | 양방향 (Separate LSTM)    |
| 파인튜닝 방식 | 전체 모델 파인튜닝          | 전체 모델 파인튜닝          | Feature-based (고정 임베딩) |
| 사전학습 과제 | MLM + NSP                   | Language Modeling            | Language Modeling         |

---

## 📉 실험 및 결과

* **데이터셋**: GLUE, SQuAD, MNLI 등  
* **비교 모델**: GPT, ELMo, 기존 BiLSTM 기반 모델 등  
* **주요 성능 지표 및 결과**:

| 모델      | GLUE 평균 | SQuAD v1.1 | SQuAD v2.0 |
|-----------|-----------|------------|------------|
| BERT-Base | 80.5      | 88.5       | 76.3       |
| BERT-Large| 82.1      | 90.9       | 80.5       |
| GPT       | 75.1      | 85.4       | 72.0       |
| ELMo      | 72.1      | 79.8       | -          |

> BERT는 당시 기준 거의 모든 자연어 처리 태스크에서 SOTA 성능을 달성함.

---

## ✅ 장점 및 한계

### 장점:
* 양방향 문맥 정보 활용 → 문장 이해 성능 향상  
* 다양한 태스크에 동일 구조 적용 가능 (범용성)  
* 사전학습 + 파인튜닝 전략의 대중화

### 한계 및 개선 가능성:
* NSP의 효과성에 대한 논란  
* [MASK] 토큰이 실제 입력에서는 존재하지 않음 → 파인튜닝 시 mismatch 발생  
* 연산량이 크고 학습 비용이 높음

---

## 🧠 TL;DR – 한눈에 요약

> **BERT는 Transformer Encoder만을 이용해 양방향 문맥을 동시에 고려하는 사전학습 언어 모델이다.**  
> 입력 문장에서 일부 토큰을 `[MASK]`로 가리고, 좌우 문맥을 모두 활용해 해당 단어를 예측하는 Masked Language Modeling(MLM)을 통해 학습된다.  
> 또한 두 문장의 연결 관계를 예측하는 Next Sentence Prediction(NSP) 태스크를 병행하여 문장 간 의미 관계도 학습한다.  
> 사전학습 이후에는 전체 모델을 다양한 다운스트림 태스크에 맞게 fine-tuning할 수 있으며,  
> 문장 분류, 개체명 인식, 질의응답 등 거의 모든 NLU 태스크에서 당시 최고 성능(SOTA)을 달성했다.  
> 이후 등장한 RoBERTa, ALBERT, DistilBERT 등 수많은 모델들의 기반이 된 **역사적 전환점**으로 평가된다.

---

| 구성 요소        | 설명 |
|------------------|------|
| **핵심 구조**     | Transformer Encoder × N (Base: 12층, Large: 24층) |
| **입력 형식**     | `[CLS]` + 문장 A + `[SEP]` + 문장 B + `[SEP]` |
| **임베딩 방식**   | Token + Segment + Position Embedding |
| **학습 목표**     | Masked Language Modeling (MLM), Next Sentence Prediction (NSP) |
| **전이 전략**     | 사전학습 후 전체 모델 fine-tuning |
| **대표 성과**     | GLUE, SQuAD 등 다양한 NLU 벤치마크에서 SOTA 달성 |
| **의의**         | 사전학습 + 문맥 양방향성 기반 범용 구조 제안 → LLM 시대의 기초 모델 |


---

## 🔗 참고 링크 (References)

* [📄 arXiv 논문](https://arxiv.org/abs/1810.04805)  
* [💻 GitHub](https://github.com/google-research/bert)  
* [📈 Papers with Code](https://paperswithcode.com/paper/bert-pre-training-of-deep-bidirectional)

---

## ⏭️ 다음 논문 예고: Swin Transformer

> "Shifted Windows를 활용한 Hierarchical Vision Transformer"  
> ViT의 전역성 + CNN의 지역성/계층적 구조를 결합한 구조로, 이미지 인식, 분할 등에서 강력한 성능을 보임

