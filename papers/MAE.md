# 📘 Masked Autoencoders Are Scalable Vision Learners

## 1. 개요 (Overview)

- **제목**: Masked Autoencoders Are Scalable Vision Learners  
- **저자**: Kaiming He, Xinlei Chen, Saining Xie, Yanghao Li, Piotr Dollár, Ross Girshick  
- **소속**: Meta AI (Facebook AI Research, FAIR)  
- **학회**: CVPR 2022  
- **링크**:  
  - [arXiv](https://arxiv.org/abs/2111.06377)  
  - [GitHub](https://github.com/facebookresearch/mae)  
  - [Papers with Code](https://paperswithcode.com/paper/masked-autoencoders-are-scalable-vision)

> 논문 선정 이유 및 간단한 도입부 작성

최근 딥러닝 분야는 모델 규모와 성능이 폭발적으로 성장하며, 수억 개의 라벨이 필요한 대규모 데이터셋에 대한 의존도가 증가하고 있다. 자연어 처리(NLP) 분야에서는 이러한 문제를 자기지도학습(self-supervised learning)을 통해 성공적으로 해결했다. 

대표적으로 GPT의 오토레그레시브 학습과 BERT의 마스킹 기반 오토인코딩(Masked Autoencoding)은 데이터의 일부를 제거한 후 이를 복원하는 방식으로 효과적인 표현 학습을 가능하게 했다. 이러한 방식은 현재 수백억 파라미터 규모의 범용 언어 모델 학습에도 적용되고 있다.

이 논문에서는 **Masked Autoencoders (MAE)** 라는 방식으로, NLP에서 검증된 마스킹 기반 사전학습을 **Vision Transformer (ViT)** 구조에 효과적으로 적용한다. 특히 이미지의 공간적 중복성과 낮은 정보 밀도를 고려하여, 전체 패치 중 75% 이상의 높은 비율을 마스킹하고, 남은 소수의 패치만으로 복원을 수행하는 방식으로 고난이도의 자가학습 과제를 만든다.

또한, MAE는 **비대칭 인코더-디코더 구조**를 통해 계산 효율성까지 확보한다. 인코더는 가시 패치(visible patches)에만 작동하며, 마스크 토큰(mask token)은 디코더에서만 처리한다. 이로 인해 **사전학습 시간과 메모리 사용량을 3배 이상 절감**하면서도 높은 성능을 유지할 수 있다.

### 📌 논문 선정 이유
- **BERT의 마스킹 전략을 Vision 도메인에 성공적으로 이식**
- **75% 이상 마스킹**이라는 도전적 학습 과제를 통해 고차원 표현 학습 유도
- **비대칭 설계**를 통해 계산 효율성과 모델 확장성 동시에 확보
- 단일 ImageNet-1K 데이터셋만으로도 SOTA 달성
- 객체 탐지, 분할 등 다양한 다운스트림 비전 태스크에서 성능 향상 입증

---
## 2. 문제 정의 (Problem Formulation)

### 🔹 문제 및 기존 한계

* 최근 Vision Transformer(ViT)의 부상과 함께, 대용량 이미지 모델을 학습하기 위한 효과적인 **사전학습 전략**의 필요성이 대두되고 있음
* 기존의 감독학습(supervised learning)은 수억 개 수준의 **라벨링된 데이터에 대한 의존도**가 높아, 일반적인 사용에 한계가 존재
* 자연어 처리(NLP)에서는 마스킹 기반의 자기지도학습 방식(BERT, GPT)이 대규모 데이터 없이도 **범용 표현 학습에 성공**하였으나,
  - 컴퓨터 비전에서는 아직 **마스킹 기반 학습이 충분히 확장되지 못함**
  - 이는 구조적 한계(과거에는 CNN 기반), 정보 밀도 차이(이미지 vs. 텍스트), 디코더 역할의 차이 등이 주된 원인

### 🔹 제안 방식 (Masked Autoencoder, MAE)

* MAE는 이미지의 대부분(예: 75%)을 무작위로 **마스킹한 후**, 나머지 일부 패치만을 인코더에 입력
* **비대칭 인코더-디코더 구조**를 통해 효율성 확보
  - 인코더: 가시 패치(visible patches)만 처리 → 계산량 절감
  - 디코더: 마스크된 위치를 포함한 전체 패치를 복원
* 복원 대상은 **픽셀 단위의 원본 이미지**로, 낮은 수준의 재구성이지만 학습된 표현은 고수준 인식 태스크에 효과적
* 마스킹 비율을 높게 설정함으로써, 단순한 복원이 아닌 **전체 이미지에 대한 이해와 추론**을 요구

> ※ **핵심 개념 정의**
>
> - **Masked Autoencoding (MAE)**: 입력 이미지의 일부 패치를 마스킹하고, 나머지 정보를 통해 원본을 복원하는 자기지도 학습 방식  
> - **Visible Patches**: 마스킹되지 않은 입력 패치로, 인코더의 입력이 됨  
> - **Mask Tokens**: 마스킹된 위치에 삽입되어 디코더가 복원 작업을 수행할 수 있게 해주는 특수 토큰  
> - **Asymmetric Encoder-Decoder**: 계산 효율성을 위해 encoder와 decoder의 역할/크기를 구분하는 설계 전략  


---

## 3. 모델 구조 (Architecture)

### 🔷 전체 구조

![모델 구조](/papers/images/mae_architecture.png)

MAE (Masked Autoencoders)는 **비대칭적 인코더-디코더 구조(asymmetric encoder-decoder architecture)**를 기반으로 하는 자기지도 학습(self-supervised learning) 프레임워크로, **입력 이미지의 일부만을 인코더에 제공하고, 마스킹된 정보를 디코더가 복원**하도록 학습된다. 이 설계는 학습 효율성과 표현력 확보라는 두 가지 목표를 동시에 달성하는 데 초점을 맞추고 있다.

입력 이미지 $x \in \mathbb{R}^{H \times W \times C}$는 고정 크기의 패치(patch)로 분할되며, 각 패치는 선형 임베딩 및 위치 인코딩을 통해 토큰화된다. 전체 패치 중 **75% 이상을 마스킹**하고, 나머지 25%의 **가시 패치(visible patches)**만 인코더에 입력된다. 이후, **마스크 토큰(mask token)**을 삽입하여 전체 시퀀스를 구성하고, 디코더가 픽셀 복원을 수행한다.

---

### 💠 핵심 모듈 및 구성 요소 상세 분석

#### 📌 Patch Embedding Module

* 입력 이미지는 $P \times P$ 크기의 non-overlapping 패치로 분할되며, 총 패치 수는 다음과 같다:
  
  $$N = \frac{H \cdot W}{P^2}$$

* 각 패치 $x_i$는 플래튼된 후, 고정 차원 $D$의 latent space로 임베딩된다:
  
  $$z_i = E_{\text{patch}}(x_i) = W_e \cdot \text{Flatten}(x_i) + p_i$$

  - 여기서 $W_e$는 학습 가능한 선형 임베딩 가중치이고, $p_i$는 learnable positional embedding이다.

* 전체 패치 중 무작위로 선택된 75%는 제거되며, 나머지 $25\%$만 인코더에 입력됨.

#### 📌 Encoder (Representation Learner)

* Transformer 기반의 인코더는 가시 패치에 대해서만 작동하며, **마스크된 위치에 대한 정보는 입력되지 않음**. 이는 계산량을 줄이고 효율적인 표현 학습을 유도함.
  
* 인코더는 $L$개의 트랜스포머 블록으로 구성되며, 각 블록은 multi-head self-attention과 MLP로 구성된다.

* 인코더 출력은 잠재 표현 $z_{\text{vis}} \in \mathbb{R}^{n \times D}$로 구성되며, 이는 디코더의 입력 중 일부가 됨.

#### 📌 Mask Token 및 Sequence Reassembly

* 디코더에 입력하기 위해, 마스킹된 위치에는 **공통적인 learnable mask token** $m \in \mathbb{R}^D$이 삽입된다.
  
* 위치 정보 보존을 위해 positional encoding을 다시 부여하여 **full-length sequence**를 재구성한다:

  $$\tilde{z} = \text{Reassemble}(z_{\text{vis}}, m) + p$$

* 이 시퀀스 $\tilde{z}$는 디코더의 입력이 된다.

#### 📌 Decoder (Pixel Reconstruction Network)

* 디코더는 상대적으로 얕은 구조 (예: 4-layer transformer)로 구성되며, 전체 패치 위치에 대한 복원 작업을 수행한다.

* 복원 대상은 **픽셀 레벨의 원본 이미지 조각**이며, 디코더의 출력 $z_j$는 다음과 같이 다시 픽셀 공간으로 투영된다:

  $$\hat{x}_j = W_d \cdot z_j, \quad \text{for masked patch } j$$

  - 여기서 $W_d$는 선형 복원 매핑에 해당한다.
  - 손실 함수는 마스킹된 위치에 대해서만 계산되며, 보통 MSE loss가 사용됨:
  
  $$ \mathcal{L}_{\text{MAE}} = \frac{1}{|\mathcal{M}|} \sum_{j \in \mathcal{M}} \left\| \hat{x}_j - x_j \right\|_2^2 $$

  - $\mathcal{M}$: 마스킹된 패치 인덱스 집합

---

### 🔍 설계 상의 핵심 차별점

| 항목 | MAE | BERT-style Vision Models |
|------|-----|---------------------------|
| 입력 토큰 | visible patch only | masked + visible patches |
| 인코더 처리 | partial input only | full sequence |
| 마스크 토큰 | 디코더에서만 사용 | 인코더 입력에 포함됨 |
| 복원 대상 | RGB 이미지 픽셀 | discrete token (e.g., VQ-VAE) |
| 목적 | reconstruction loss 기반 representation 학습 | classification-oriented 또는 디스크리트 복원 |

---

### 📌 요약

* MAE는 복원 중심의 자기지도 학습을 통해 고차원 표현 학습을 가능하게 하는 효율적인 비전 프레임워크
* **인코더는 가시 패치에만 집중**하여 연산을 줄이고 학습 표현의 효율성을 극대화함
* **디코더는 마스크 위치 정보를 보강하고 원본을 복원**함으로써 학습 목표를 완성
* 높은 마스킹 비율과 비대칭 구조 설계를 통해 **일반화 성능과 계산 자원 간의 균형**을 성공적으로 달성



---

## ⚖️ 기존 모델과의 비교

| 항목        | 본 논문 (MAE)                            | BEiT (Bao et al., 2021)                        | SimMIM (Xie et al., 2021)                     |
|-------------|-------------------------------------------|------------------------------------------------|-----------------------------------------------|
| 구조        | 비대칭 encoder-decoder 구조               | Transformer 기반 encoder-only                 | encoder-only 구조                              |
| 학습 방식   | 75% 마스킹 후 픽셀 복원                   | discrete token 복원 (VQ tokenizer 필요)        | 픽셀 복원                                     |
| 목적        | 효율적 representation 학습 및 전이        | BERT-style 사전학습 → downstream 전이          | 이미지 복원 기반 pretraining                 |

---

## 📉 실험 및 결과

### 🧪 **데이터셋**
- Pretraining: ImageNet-1K (100만 이미지, 라벨 불사용)
- Finetuning/Transfer: ImageNet-1K, ADE20K, COCO

### 🤖 **비교 모델**
- ViT-B / ViT-L / ViT-H 모델 기반
- Supervised Pretraining, MoCo-v3, BEiT 등과 비교

### 📊 **주요 성능 지표 및 결과**

| 모델             | Top-1 Accuracy (ImageNet-1K) | 기타 성능 |
|------------------|------------------------------|------------|
| MAE (ViT-B)      | 83.6%                        | -          |
| MAE (ViT-L)      | 85.9%                        | -          |
| MAE (ViT-H)      | **87.8%**                    | -          |
| BEiT (ViT-B)     | 83.2%                        | -          |
| SimMIM (ViT-B)   | 83.8%                        | -          |
| Supervised (ViT-H) | 85.1%                      | -          |

> **해석**: MAE는 **ImageNet-1K만으로 사전학습**하고도 기존 supervised 및 self-supervised 방법보다 높은 정확도를 기록. 특히 ViT-H 모델에서 **SOTA 달성**.

---

## ✅ 장점 및 한계

### ✅ **장점**
- 매우 높은 masking ratio (75%)에도 성능 유지 → **효율적인 pretraining**
- 인코더가 전체 토큰을 처리하지 않아도 되어 **메모리 및 계산 비용 절감**
- 추가적인 토크나이저(VQ 등) 없이 **픽셀 복원만으로 강한 표현 학습**
- 다양한 downstream task (분류, 검출, 분할)에서 강력한 성능

### ⚠️ **한계 및 개선 가능성**
- 복원 대상이 픽셀이기 때문에 **semantic richness 부족**
- 학습 초기에는 복원 과제가 너무 단순하거나 너무 어려울 수 있음
- encoder에만 학습 집중 → decoder의 역할이 제한적

---

## 🧠 TL;DR – 한눈에 이해하는 MAE

> **MAE (Masked Autoencoders)**는 입력 이미지의 75% 이상을 무작위로 마스킹하고, 남은 일부 패치만을 인코더로 처리하여 전체 이미지를 복원하는 자기지도 학습 방식이다.  
> **비대칭적인 encoder-decoder 구조**를 통해 계산 효율성과 표현력 학습을 동시에 달성하며, ImageNet-1K만을 이용한 사전학습으로도 기존 SOTA를 능가하는 결과를 보여준다.

| 구성 요소    | 설명 |
|--------------|------|
| 🔧 **핵심 아이디어** | NLP의 Masked Language Modeling을 차용해, 이미지에서도 마스킹된 부분을 복원하는 방식으로 시각 표현 학습을 수행함 |
| 🧠 **학습 전략** | 전체 패치 중 75%를 무작위 마스킹 → visible patch만 인코더에 입력 → 마스크 토큰을 포함한 전체 시퀀스를 디코더가 복원 |
| ⚙️ **모델 구조** | 가벼운 디코더와 깊은 인코더를 분리한 **비대칭 구조** 채택 → 연산량 3배 이상 감소, 학습 시간 단축 |
| 🎯 **복원 과제** | 복원 대상은 discrete token이 아닌 **픽셀 레벨 이미지** → 추가 토크나이저 없이 간결하고 직관적인 학습이 가능 |
| 📈 **전이 성능** | 사전학습된 MAE + ViT는 분류, 객체 탐지, 세그멘테이션 등 다양한 다운스트림 작업에서 기존 supervised 및 self-supervised 방법보다 우수한 성능을 보임 |
| 📊 **성능 지표** | ViT-H 모델 기준, ImageNet-1K Top-1 accuracy: **87.8%** (사전학습에 ImageNet-1K만 사용) |
| 🧩 **비교 우위** | BEiT, SimMIM 등 기존 마스킹 기반 방법들과 비교해 **추가 토크나이저 필요 없음**, **더 높은 masking ratio 지원**, **더 빠르고 단순한 구조** |

> ✅ MAE는 단순한 이미지 복원 과제를 통해 효율적이고 확장성 있는 비전 표현 학습을 가능하게 하며, BERT 이후 가장 성공적인 self-supervised 프레임워크 중 하나로 자리 잡음.

---

## 🔗 참고 링크 (References)

* [📄 arXiv 논문](https://arxiv.org/abs/2111.06377)
* [💻 GitHub](https://github.com/facebookresearch/mae)
* [📈 Papers with Code](https://paperswithcode.com/paper/masked-autoencoders-are-scalable-vision)


## 다음 논문: SegFormer
