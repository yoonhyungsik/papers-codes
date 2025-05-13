# 📘 Vision Transformer (ViT): An Image is Worth 16x16 Words

## 1. 개요 (Overview)

* **제목**: An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale  
* **저자**: Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, et al.  
* **소속**: Google Research, Brain Team  
* **학회**: ICLR 2021  
* **링크**: [arXiv](https://arxiv.org/abs/2010.11929) / [GitHub](https://github.com/google-research/vision_transformer) / [Papers with Code](https://paperswithcode.com/paper/an-image-is-worth-16x16-words-transformers)

## 2. 논문 선정 이유 및 도입부

기존 컴퓨터 비전 분야에서는 CNN이 압도적인 성능을 보여주었지만, NLP에서 성공한 Transformer 구조를 이미지 인식에 도입하고자 하는 시도가 이어져 왔음. 본 논문은 Transformer를 순수하게 이미지 분류에 적용한 최초의 연구로, 

**이미지를 패치 단위로 나누어 토큰처럼 처리**하고, 이를 **기존 NLP용 Transformer에 입력**하여 학습하는 새로운 접근을 제안.

특히, 대규모 데이터셋과 사전학습(pretraining)을 통해 CNN 기반 모델과 **동등하거나 그 이상의 성능**을 달성할 수 있다는 점에서 큰 의의가 있다. ViT는 이후 Swin Transformer, DeiT, BEiT 등의 다양한 비전 트랜스포머 연구의 기반이 되었기 때문에, 가장 먼저 읽고 이해해야 할 핵심 논문이라 생각.

---


## 🧠 문제 정의 (Problem Definition)

자연어 처리(NLP) 분야에서는 Self-Attention 기반의 **Transformer** 구조가 표준 모델로 자리잡았으며, 대규모 말뭉치에 대한 사전학습 후 소규모 태스크에 파인튜닝하는 방식이 널리 사용되고 있다 (Devlin et al., 2019).  
이러한 Transformer 기반 모델은 **100B 이상의 파라미터**를 가진 대형 모델도 효율적으로 학습 가능하며, 성능의 정체 현상도 아직 보이지 않는다.

반면, **컴퓨터 비전(CV)** 분야에서는 여전히 **CNN 기반 구조**(LeNet, AlexNet, ResNet 등)가 지배적이다. Transformer 구조를 비전에 적용하려는 시도는 있었지만 다음과 같은 한계가 존재했다:

- CNN이 가지는 **지역성(Locality)**과 **변환 불변성(Translation Equivariance)**과 같은 inductive bias가 없음
- 일부 모델은 복잡한 attention 패턴을 사용하여 **하드웨어 가속기와의 호환성**이 떨어짐
- **중간 규모 데이터셋**(예: ImageNet)에서는 CNN에 비해 성능이 낮음

---

## 💡 제안 방식 (Proposed Method: Vision Transformer)

ViT는 **최대한 단순한 형태의 Transformer**를 이미지에 직접 적용하는 접근을 제안한다.

### 🔹 핵심 아이디어

- 입력 이미지를 **고정 크기 패치(patch)**로 나눈 후, 각 패치를 **선형 임베딩(linear projection)**하여 Transformer의 입력 시퀀스로 사용
- 각 이미지 패치는 NLP에서의 단어 토큰(token)처럼 처리됨
- 특별한 구조적 변경 없이 **기존 NLP용 Transformer 아키텍처를 그대로 활용**
- **Supervised Learning** 방식으로 이미지 분류 학습
---

## 3. 모델 구조 (Architecture)

### 🔷 전체 구조

![모델 구조](path/to/vit_architecture.png)

Vision Transformer (ViT)는 NLP에서 사용되는 표준 Transformer 구조를 이미지에 그대로 적용한 모델로, CNN 없이 순수 Transformer만으로 이미지 인식 문제를 해결하고자 한다.

- 입력 이미지: $x \in \mathbb{R}^{H \times W \times C}$
- 이미지를 $P \times P$ 크기의 패치로 나눈 뒤, 각 패치를 펼쳐 벡터화 (Flatten)
- 총 패치 수: $N = \frac{HW}{P^2}$
- 각 패치를 선형 투영(linear projection)을 통해 Transformer 입력 차원 $D$로 임베딩
- $\texttt{[CLS]}$ 토큰을 첫 위치에 삽입하고, 포지션 임베딩(Position Embedding)을 추가하여 Transformer Encoder에 입력
- 마지막 $\texttt{[CLS]}$ 토큰 출력을 통해 전체 이미지 표현을 얻고, 이를 분류기에 전달

---

### 💠 핵심 모듈 구성

#### 📌 Patch Embedding

- 입력 이미지 $x \in \mathbb{R}^{H \times W \times C}$를 $P \times P$ 크기의 패치로 분할 후 펼침
- 각 패치 $x_p \in \mathbb{R}^{P^2 \cdot C}$를 선형 투영하여 차원 $D$로 변환

$$
x_p \mapsto x_pE, \quad E \in \mathbb{R}^{(P^2 \cdot C) \times D}
$$

- 포지션 임베딩 $E_{\text{pos}} \in \mathbb{R}^{(N+1) \times D}$를 더함
- [CLS] 토큰 $x_{\text{class}}$를 앞에 붙여 최종 입력 시퀀스 구성

$$
z_0 = [x_{\text{class}}; x_1E; x_2E; \cdots; x_N E] + E_{\text{pos}}
$$

---

#### 📌 Transformer Encoder

**Transformer Encoder Layer 구성:**

- Step 1: Multi-Head Self-Attention + Residual

$$
z'_{\ell} = \text{MSA}(\text{LN}(z_{\ell - 1})) + z_{\ell - 1}, \quad \ell = 1, \dots, L
$$

- Step 2: MLP + Residual

  $$
  z_{\ell} = \text{MLP}(\text{LN}(z'_{\ell})) + z'_{\ell}, \quad \ell = 1, \dots, L
  $$


- 최종 출력 정규화:

  $$
  y = \text{LN}(z_0^L)
  $$



---

#### 📌 Classification Head

- 마지막 [CLS] 토큰의 출력 $z_0^L$에 LayerNorm을 적용 후, 분류 헤드에 입력:

$$
y = \text{LN}(z_0^L)
$$

- Pre-training 시: MLP with 1 hidden layer
- Fine-tuning 시: Linear layer (zero-initialized)

---

### 🧩 위치 정보 처리 (Positional Encoding)

- $1D$ Learnable positional embedding 사용
- 고해상도 입력을 처리할 때는 **2D interpolation**으로 기존 포지션 임베딩을 조정

---

### 🌀 Hybrid Architecture (선택적 변형)

- 순수 패치 대신 CNN feature map으로부터 패치 추출 가능
- 이 경우, feature map을 flatten하고 선형 투영하여 Transformer 입력 시퀀스로 사용

---

### ⚠️ Inductive Bias 비교

- CNN은 **locality, translation equivariance, 2D 구조**를 내재적으로 포함
- ViT는 이러한 inductive bias가 거의 없으며, 대부분의 공간 정보는 학습을 통해 내재화되어야 함


---

## ⚖️ 기존 모델과의 비교

| 항목    | 본 논문 | 기존 방법1 | 기존 방법2 |
| ----- | ---- | ------ | ------ |
| 구조    |      |        |        |
| 학습 방식 |      |        |        |
| 목적    |      |        |        |

---

## 📉 실험 및 결과

* **데이터셋**:
* **비교 모델**:
* **주요 성능 지표 및 결과**:

| 모델      | Accuracy | F1 | BLEU | 기타 |
| ------- | -------- | -- | ---- | -- |
| 본 논문    |          |    |      |    |
| 기존 SOTA |          |    |      |    |

> 실험 결과 요약 및 해석

---

## ✅ 장점 및 한계

## **장점**:

*

## **한계 및 개선 가능성**:

*

---

## 🧠 TL;DR – 한눈에 요약

> 핵심 아이디어 요약 + 이 논문의 기여를 한 줄로 요약

| 구성 요소  | 설명 |
| ------ | -- |
| 핵심 모듈  |    |
| 학습 전략  |    |
| 전이 방식  |    |
| 성능/효율성 |    |

---

## 🔗 참고 링크 (References)

* [📄 arXiv 논문](https://arxiv.org/)
* [💻 GitHub](https://github.com/)
* [📈 Papers with Code](https://paperswithcode.com/)

## 다음 논문:
