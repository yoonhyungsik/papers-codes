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

- 입력 이미지를 **고정된 크기의 비중첩 패치(patch)** 단위로 분할한 후, 각 패치를 **선형 임베딩(linear projection)**하여 Transformer의 입력 시퀀스로 활용한다.
- 이렇게 생성된 패치 시퀀스는 NLP에서의 단어 토큰(token) 시퀀스와 유사한 방식으로 처리된다.
- 기존 자연어 처리용 **Transformer 아키텍처를 최소한의 수정만으로 적용**하며, convolution 연산은 전혀 사용하지 않는다.
- 학습은 **전통적인 감독 학습(supervised learning)** 방식으로 수행되며, 대규모 데이터셋을 활용한 사전학습과 소규모 다운스트림 태스크에의 파인튜닝을 통해 전이 학습이 가능하다.

---

## 3. 모델 구조 (Architecture)

### 🔷 전체 구조

![모델 구조](path/to/vit_architecture.png)

Vision Transformer(ViT)는 기존 Transformer 구조를 시각 입력에 직접 적용한 최초의 순수 Transformer 기반 모델 중 하나로, convolution 연산 없이 이미지 분류 문제를 해결하는 데 초점을 맞춘다. 모델의 전반적 구조는 다음과 같다:

- 입력 이미지 $x \in \mathbb{R}^{H \times W \times C}$는 고정 크기 $(P \times P)$ 패치로 분할되며, 총 패치 수는 $N = \frac{HW}{P^2}$이다.
- 각 패치는 펼쳐진 후(linearly flattened) 차원 $D$의 공간으로 선형 임베딩된다.
- 임베딩된 패치 시퀀스 앞에 학습 가능한 [CLS] 토큰을 prepend하고, 위치 정보를 보존하기 위해 learnable positional embedding을 element-wise로 더한다.
- 결과 시퀀스는 표준 Transformer Encoder에 입력되며, 최종적으로 [CLS] 토큰의 출력 벡터는 분류 목적의 표현으로 사용된다.

---

### 💠 핵심 구성 모듈

#### 📌 Patch Embedding

- 입력 이미지를 $P \times P$ 크기의 non-overlapping patch로 분할한 후, 각 패치 $x_p \in \mathbb{R}^{P^2 \cdot C}$는 선형 임베딩 계층을 통해 $D$ 차원으로 투영된다:

$$
x_p \mapsto x_pE, \quad E \in \mathbb{R}^{(P^2 \cdot C) \times D}
$$

- 전체 시퀀스에는 위치 임베딩 $E_{\text{pos}} \in \mathbb{R}^{(N+1) \times D}$이 추가되며, 학습 가능한 [CLS] 토큰 $x_{\text{class}}$가 시퀀스의 첫 번째 위치에 삽입된다:

$$
z_0 = [x_{\text{class}}; x_1E; x_2E; \cdots; x_N E] + E_{\text{pos}}
$$

---

#### 📌 Transformer Encoder

Transformer Encoder는 $L$개의 동일한 레이어로 구성되며, 각 레이어는 Multi-Head Self-Attention(MSA) 모듈과 Position-wise Feedforward Network(MLP)로 이루어진다. 각 서브블록 앞에는 Layer Normalization이 적용되며, residual connection이 후속 연산과 함께 사용된다.

**Step 1: Multi-Head Self-Attention (MSA) + Residual connection**

$$
\mathbf{Z}'_\ell = \mathrm{MSA}(\mathrm{LN}(\mathbf{Z}_{\ell - 1})) + \mathbf{Z}_{\ell - 1}
$$

**Step 2: Feedforward (MLP) + Residual connection**

$$
\mathbf{Z}_\ell = \mathrm{MLP}(\mathrm{LN}(\mathbf{Z}'_\ell)) + \mathbf{Z}'_\ell
$$

**최종 출력 (분류를 위한 표현):**

$$
\mathbf{y} = \mathrm{LN}(\mathbf{Z}_0^L)
$$

---

#### 📌 Classification Head

- Encoder의 최종 출력 시퀀스 중 [CLS] 토큰에 해당하는 벡터 $z_0^L$에 LayerNorm을 적용한 후, 분류 헤드에 입력된다:

$$
y = \mathrm{LN}(z_0^L)
$$

- Pre-training 시: 1 hidden layer를 갖는 MLP로 구성
- Fine-tuning 시: zero-initialized linear projection만을 사용하여 효율적으로 학습

---

### 🧩 위치 정보 처리 (Positional Encoding)

- 패치 간의 상대적/절대적 순서를 인코딩하기 위해 learnable한 **1차원 위치 임베딩(positional embedding)**을 사용한다.
- 파인튜닝 과정에서 해상도를 변경하는 경우, 사전 학습된 위치 임베딩을 입력 해상도에 맞게 **2D 선형 보간(interpolation)**하여 적용한다.

---

### 🌀 Hybrid Architecture (선택적 변형)

- 순수 이미지 패치 대신, 기존 CNN 백본에서 추출한 feature map을 patch로 사용하는 hybrid 구조도 제안된다.
- 이 경우, CNN 출력 feature를 flatten한 후 동일한 선형 임베딩 계층을 적용한다. 이러한 구조는 convolution의 inductive bias를 유지하면서 Transformer의 전역 표현 능력을 활용할 수 있는 절충안이다.

---

### ⚠️ Inductive Bias 비교

- Convolutional Neural Networks(CNN)는 **local connectivity, translation equivariance, shift invariance** 등 시각적 inductive bias를 아키텍처 수준에서 강제한다.
- 반면, ViT는 이러한 inductive bias가 없으며, 대신 **더 많은 데이터와 더 긴 학습**을 통해 공간 관계 및 위치 정보를 학습해야 한다.
- 그 결과 ViT는 **데이터 크기와 연산 자원이 충분한 경우 더 강력한 표현력**을 보이지만, **소규모 데이터 환경에서는 일반화 성능이 제한적일 수 있음**을 시사한다.


## ⚖️ 기존 모델과의 비교

| 항목        | 본 논문 (ViT)                                               | 기존 방법1 (ResNet)                             | 기존 방법2 (CNN+Self-Attention, e.g. Bottleneck Attention) |
|-------------|--------------------------------------------------------------|--------------------------------------------------|-------------------------------------------------------------|
| 구조        | Pure Transformer (patch + position embedding + encoder)     | Convolutional layers with residual blocks        | CNN + attention module (local or hybrid attention)         |
| 학습 방식   | 대규모 사전학습 후 파인튜닝 (Supervised, scale up 중요)     | End-to-end supervised 학습                       | End-to-end supervised 학습                                 |
| 목적        | CNN 없이 이미지 인식 가능 여부 실험 + 대규모 학습의 효과 검증 | 효과적인 이미지 분류 (특히 적은 계산량에서 우수) | CNN의 inductive bias를 유지하면서 attention 효과 결합       |

---

## 📉 실험 및 결과

- **데이터셋**:
  - ImageNet-1k
  - ImageNet-21k (pretraining)
  - JFT-300M (in-house, pretraining)
  - CIFAR-100, VTAB

- **비교 모델**:
  - ResNet-152
  - EfficientNet
  - Big Transfer (BiT)
  - Hybrid CNN+Transformer 모델

- **주요 성능 지표 및 결과**:

| 모델             | Top-1 Accuracy | CIFAR-100 | VTAB (19 tasks) | 기타 |
|------------------|----------------|-----------|------------------|------|
| ViT-L/16 (JFT-300M 사전학습) | 88.55%        | 94.55%    | 77.63%           | ImageNet-ReaL: 90.72% |
| ResNet-152       | ~78.5%         | 낮음       | 낮음             | -    |
| BiT-L (Big Transfer) | ~87.5%     | 93% 이상   | 75% 내외         | -    |

> 🔍 **실험 결과 요약**: 대규모 데이터셋(JFT-300M 등)을 활용한 사전학습이 없으면 성능이 떨어지지만, 충분히 사전학습하면 ViT는 CNN 기반 모델을 능가하는 성능을 보여줌.

---

## ✅ 장점 및 한계

### ✅ 장점:

- **심플한 구조**: 이미지 처리에 convolution 없이도 효과적인 성능 달성
- **확장성 우수**: 데이터와 모델 규모를 키울수록 성능이 더 좋아짐
- **전이학습 친화적**: 대규모 데이터로 학습 후 다양한 다운스트림 태스크에 전이 가능

### ⚠️ 한계 및 개선 가능성:

- **대규모 학습 필요**: 작은 데이터셋에서는 inductive bias 부족으로 일반화 성능 저하
- **데이터 효율성 낮음**: ResNet보다 작은 데이터에서 비효율적
- **위치 정보 학습 필요**: CNN과 달리 위치/로컬 정보를 직접 학습해야 함

---

## 🧠 TL;DR – 한눈에 요약

> **"Convolution 없이 순수 Transformer 구조만으로도 이미지 인식이 가능하며, 충분히 대규모의 데이터셋과 연산 자원을 사용할 경우 기존 CNN 기반 모델을 능가할 수 있다."**

Vision Transformer(ViT)는 자연어 처리(NLP) 분야에서 성공을 거둔 Transformer 아키텍처를 이미지 분류 태스크에 직접 적용한 첫 번째 시도 중 하나다. 이 논문은 Convolution 연산 없이, 이미지를 고정 크기의 패치로 분할한 후 각 패치를 단어 토큰처럼 처리하여 Transformer에 입력하는 구조를 제안한다. 이를 통해 ViT는 CNN과 달리 inductive bias(예: locality, translation equivariance)에 의존하지 않고 전역적 관계를 학습할 수 있다.

모델 구조는 기존의 NLP용 Transformer를 거의 그대로 따르며, 입력 이미지는 패치 → 임베딩 → 위치 인코딩을 거쳐 Transformer Encoder에 전달된다. 출력은 [CLS] 토큰의 최종 벡터를 통해 분류된다.

논문에서 저자들은 ImageNet-21k 및 JFT-300M과 같은 **대규모 사전학습 데이터셋**을 통해 ViT를 학습시키고, 다양한 다운스트림 태스크 (ImageNet, CIFAR-100, VTAB 등)로 전이시켰을 때 **ResNet, EfficientNet, BiT 등 기존 SOTA CNN 모델들을 능가하는 성능**을 입증한다.

하지만 ViT는 작은 데이터셋이나 사전학습 없이 학습될 경우 CNN에 비해 성능이 낮고, inductive bias가 없기 때문에 일반화 성능이 부족해질 수 있다. 이를 보완하기 위해 대규모 학습과 정규화 기법이 필요하다.

### 📌 이 논문의 핵심 기여

- CNN 구조 없이 **Transformer만으로 이미지 분류를 가능하게 하는 새로운 패러다임** 제시
- 이미지 → 패치 → 시퀀스로 변환하여 **비전 데이터를 NLP 방식으로 처리**
- 대규모 데이터 사전학습이 **inductive bias보다 더 중요할 수 있음**을 실험적으로 검증
- Vision 분야에서 이후 DeiT, Swin, MAE 등 다양한 Transformer 기반 모델들의 출발점이 된 **기초적인 연구**

---

## 🔗 참고 링크 (References)

* [📄 arXiv 논문 (An Image is Worth 16x16 Words)](https://arxiv.org/abs/2010.11929)
* [💻 GitHub (google-research/vision_transformer)](https://github.com/google-research/vision_transformer)
* [📈 Papers with Code (ViT)](https://paperswithcode.com/paper/an-image-is-worth-16x16-words-transformers)


## 다음 논문:
