# 📘 Swin Transformer: Hierarchical Vision Transformer using Shifted Windows

## 1. 개요 (Overview)

- **제목**: Swin Transformer: Hierarchical Vision Transformer using Shifted Windows  
- **저자**: Ze Liu, Yutong Lin, Yue Cao, Han Hu, Yixuan Wei, Zheng Zhang, Stephen Lin, Baining Guo  
- **소속**: Microsoft Research Asia  
- **학회**: ICCV 2021 (International Conference on Computer Vision)  
- **링크**: [arXiv](https://arxiv.org/abs/2103.14030) / [GitHub](https://github.com/microsoft/Swin-Transformer) / [Papers with Code](https://paperswithcode.com/paper/swin-transformer-hierarchical-vision)


## 2. 논문 선정 이유 및 도입부

자연어 처리(NLP) 분야에서 Transformer는 뛰어난 성능으로 주목받아왔으며, BERT, GPT 등의 모델이 그 가능성을 입증해왔다. 이러한 Transformer 구조를 이미지 처리에 적용한 ViT(Vision Transformer)는 CNN 없이도 이미지 분류에서 경쟁력 있는 성능을 보임.

그러나 ViT는 다음과 같은 한계를 갖고 있었다:

- **Hierarchical feature extraction이 불가능**하여 다양한 비전 과제에 활용하기 어렵고,
- **Global self-attention 구조**로 인해 **고해상도 입력에 비효율적**이며,
- **Local information learning이 부족**하여 세밀한 객체 인식에 약한 경향이 있음.

이에 따라, 해당 논문인 **Swin Transformer**는 위 문제를 해결하기 위해 제안된 모델로, 다음과 같은 이유에서 흥미를 끌었다:

- Transformer 구조를 **local window-based attention**으로 변형하여 CNN처럼 계층적이고 지역적인 정보를 학습 가능하게 함
- **Shifted window mechanism**을 통해 정보 흐름을 효율적으로 확장
- ViT보다 범용성이 뛰어나 **객체 검출, 분할, 추론 등 다양한 비전 작업에 유연하게 적용 가능**

이처럼 Swin Transformer는 ViT의 구조적 한계를 해결하며, 비전 분야에서 Transformer 계열 모델의 실용성을 한 단계 끌어올렸다는 점에서 꼭 읽어야할 논문이라 판단했다.

---

## 2. 문제 정의 (Problem Formulation)

### 📌 문제 및 기존 한계

- 기존 컴퓨터 비전 모델은 대부분 **CNN 기반 구조**로 발전해 왔으며, AlexNet 이후 다양한 변형이 등장하며 성능을 향상시켜 왔음.
- 최근 NLP에서는 **Transformer**가 주류 아키텍처로 자리 잡았으며, 이를 Vision 분야로 확장하려는 시도들이 이루어짐 (예: ViT).
- 그러나 Vision에 Transformer를 직접 적용하는 데에는 다음과 같은 **구조적 한계**가 존재:

  1. **Token 크기 고정 문제**  
     - NLP에서는 word token이 고정된 단위인 반면, Vision에서는 객체의 **크기가 다양**함.  
     - ViT는 고정된 크기의 patch만 사용하므로, **다중 스케일 객체 표현에 부적합**.

  2. **높은 해상도 처리의 비효율성**  
     - Vision에서는 고해상도 이미지가 일반적이며, semantic segmentation처럼 **dense prediction**이 필요한 과제도 많음.  
     - 기존 Vision Transformer는 **global self-attention**을 사용하므로 **계산 복잡도 $O(N^2)$**로 비효율적임.

  3. **계층적 표현 부족**  
     - 기존 Vision Transformer는 **단일 해상도의 feature map**만 생성함.  
     - CNN처럼 **계층적으로 정보 추출하는 구조가 부재**하여, FPN이나 U-Net 등의 기존 vision 기법과의 연계가 어려움.

---

### 💡 제안 방식 (Swin Transformer)

Swin Transformer는 다음과 같은 방식으로 기존 한계를 극복함:

1. **Hierarchical Feature Map 구성**  
   - 입력 이미지를 작은 patch로 분할 후, 각 계층에서 **인접 patch를 병합**하여 점진적으로 해상도를 낮춤.
   - CNN처럼 계층적 표현이 가능하여, **다양한 비전 작업 (분류, 탐지, 분할)**에 효과적.

2. **Local Window 기반 Self-Attention**  
   - Global이 아닌 **local window 단위**로 self-attention 수행 → **계산 복잡도 $O(N)$로 감소**.
   - 각 window는 고정된 크기로 나뉘며, 병렬 처리와 latency 측면에서도 효율적.

3. **Shifted Window Mechanism**  
   - 인접 계층 간의 window 경계를 **한 칸씩 밀어** 배치함으로써, **cross-window 연결**을 형성.  
   - 정보를 전역적으로 전달하면서도 효율적인 구조 유지.

4. **범용 백본으로서의 활용성 확보**  
   - FPN, U-Net과 연계 가능한 구조로 **dense prediction** 작업에 적합.  
   - Classification, Detection, Segmentation 모두에서 **SOTA 수준 성능**을 달성.

---

### 🧠 핵심 개념 정의

- **Hierarchical Feature Map**
  
  ![swin_vit_comp](/papers/images/swin_vit_comp.png)
  
  ViT와 달리, 여러 단계의 resolution을 가지는 feature map을 생성하여 다양한 스케일의 시각 정보를 효과적으로 처리함.
 

- **Window-based Multi-head Self-Attention (W-MSA)**
  
  ![batch_computation](/papers/images/batch_computation.png)
  
  이미지 전체가 아닌 **작은 창(window)** 단위에서만 self-attention 수행 → 선형 계산 복잡도 확보.
  

- **Shifted Window**
  
  ![shift_window](/papers/images/shifted_window.png)
  
  window 경계를 다음 layer에서 한 칸씩 이동시켜 **cross-window dependency**를 형성.  
  모델의 표현력을 증가시키면서도 연산 효율성 유지.

- **General-purpose Backbone**  
  이미지 분류뿐 아니라 객체 탐지, 시맨틱 분할 등 **다양한 CV 과제에서 활용 가능한 백본 구조**로 설계됨.

---

## 3. 모델 구조 (Architecture)

### 🏗️ 전체 구조

![swin_architecture](/papers/images/swin_architecture.png)

Swin Transformer는 전체적으로 CNN과 유사한 **계층적 아키텍처(hierarchical architecture)**를 따르며, 입력 이미지를 작은 패치로 분할하고, 이후 여러 단계에 걸쳐 feature resolution을 점진적으로 축소시키면서 채널 수를 증가시킨다.

- 입력 이미지는 $H \times W \times 3$ 크기의 RGB 이미지
- $4 \times 4$ patch 분할 → 각 patch는 flatten되어 임베딩됨
- 총 4단계 (Stage 1 ~ 4)로 구성되며, 각 단계는 다음과 같은 연산 흐름을 따른다:

Patch Partition → Linear Embedding → Swin Transformer Blocks → Patch Merging → ...

각 단계에서는 Swin Transformer Block을 여러 개 쌓아서 local-context 기반의 표현을 학습하며, 다음 단계로 넘어갈 때 **Patch Merging**을 통해 spatial resolution을 절반으로 줄이고 channel 수를 증가시킴.

---

### 💠 핵심 모듈 또는 구성 요소

#### 📌 Patch Partition & Linear Embedding

- 입력 이미지를 $4 \times 4$ 비중첩(non-overlapping) patch로 분할
- 각 patch는 flatten되어 길이 $4 \times 4 \times 3 = 48$ 벡터가 되고, Linear Projection을 통해 $C$차원으로 매핑됨

> 수식 표현:  
> 이미지 $x \in \mathbb{R}^{H \times W \times 3}$ → patch sequence $z_0 \in \mathbb{R}^{\frac{HW}{16} \times C}$

---

#### 📌 Swin Transformer Block

각 Stage는 Swin Transformer Block으로 구성되며, 이는 다음의 두 Attention 모듈 쌍으로 구성됨:

1. **Window-based Multi-head Self-Attention (W-MSA)**
2. **Shifted Window-based Multi-head Self-Attention (SW-MSA)**

각 블록은 PreNorm 구조로 LayerNorm, MLP, Residual Connection으로 구성된다.

> 수식 흐름 (한 block 내 W-MSA와 SW-MSA 쌍):

$$
\begin{aligned}
\hat{z}^{l} &= \text{W-MSA}(\text{LN}(z^{l-1})) + z^{l-1} \\
\tilde{z}^{l} &= \text{MLP}(\text{LN}(\hat{z}^{l})) + \hat{z}^{l} \\
\hat{z}^{l+1} &= \text{SW-MSA}(\text{LN}(\tilde{z}^{l})) + \tilde{z}^{l} \\
z^{l+1} &= \text{MLP}(\text{LN}(\hat{z}^{l+1})) + \hat{z}^{l+1}
\end{aligned}
$$

다음 block에서는 W-MSA 대신 SW-MSA가 사용되며, 동일한 구조에서 윈도우 위치만 이동시킨다.

---

#### 📌 Window-based Multi-head Self-Attention (W-MSA)

- 입력 feature map을 $M \times M$ 크기의 local window로 분할하여 **각 window 내에서만 Self-Attention 수행**
- 연산 복잡도는 $O(HW)$로 감소하며, 전체 이미지에 대한 global attention을 피함

> 기본 Attention 수식:

$$
\text{Attention}(Q, K, V) = \text{Softmax}\left(\frac{QK^T}{\sqrt{d}}\right)V
$$

- $Q, K, V$는 각 window 단위에서 계산됨

---

#### 📌 Shifted Window Multi-head Self-Attention (SW-MSA)

- 다음 블록에서는 기존 window를 **$\lfloor \frac{M}{2} \rfloor$ 만큼 shift**하여 새로운 윈도우 생성
- 이를 통해 이전 window 간 **cross-window dependency**를 학습 가능
- Shift로 인한 연산 문제는 **masking과 cyclic shift** 기법으로 해결

> 효과:
> - Window 간 정보 연결 (global receptive field 보완)
> - 효율성 유지하면서 표현력 향상

> 수식적 차이는 없으며, 단지 window partition 위치만 달라짐

---

#### 📌 Patch Merging Layer

- 다음 계층으로 넘어갈 때, $2 \times 2$ 인접한 patch들을 병합하여 resolution을 절반으로 줄이고, 채널 수는 2~4배로 증가
- CNN의 Downsampling과 유사한 역할 수행
- 이 과정을 통해 **계층적으로 표현 추출 가능**

> 수식 표현:  
> $z^{(i)} \in \mathbb{R}^{\frac{H}{2^i} \times \frac{W}{2^i} \times C_i}$  
> → patch merging →  
> $z^{(i+1)} \in \mathbb{R}^{\frac{H}{2^{i+1}} \times \frac{W}{2^{i+1}} \times C_{i+1}}$  
> where $C_{i+1} = 2 \times C_i$ or $4 \times C_i$

---

#### 📌 MLP & Normalization

- 각 Swin Block 내에는 Feed-Forward Network (MLP)와 LayerNorm이 포함됨
- Residual 구조와 함께 안정적인 학습 보장

> MLP 구성:
> LayerNorm → Linear (C → 4C) → GELU → Linear (4C → C)

> 수식:  
\( \text{MLP}(x) = W_2 \cdot \text{GELU}(W_1 x) \)

---

### 🔄 Stage 구조 요약

| Stage | Resolution        | Patch 수        | Channel 수 | Block 수 |
|-------|-------------------|------------------|-------------|-----------|
| 1     | H/4 × W/4         | HW/16            | 96          | 2         |
| 2     | H/8 × W/8         | HW/64            | 192         | 2         |
| 3     | H/16 × W/16       | HW/256           | 384         | 6         |
| 4     | H/32 × W/32       | HW/1024          | 768         | 2         |


---

### 🔄 아키텍처의 반복 구조

- Swin-T, Swin-S, Swin-B는 block 수와 hidden dim만 다름
- 예:  
  - **Swin-Tiny**: $(2, 2, 6, 2)$ blocks per stage  
  - **Swin-Small**: $(2, 2, 18, 2)$  
  - **Swin-Base**: $(2, 2, 18, 2)$ with larger embedding

---
## ⚖️ 기존 모델과의 비교

| 항목        | Swin Transformer (본 논문) | ViT (Vision Transformer) | CNN 기반 모델 (ResNet 등) |
| ----------- | -------------------------- | ------------------------- | -------------------------- |
| 구조        | Hierarchical Transformer 구조, Local-Global 윈도우 기반 Attention | 단일 해상도, Global Self-Attention | Convolution 계층 기반 피처 추출 |
| 학습 방식   | Local window-based attention → Shifted window → 계층적 학습 | 전체 patch 간 global attention | 커널 기반 지역 receptive field 확대 |
| 목적        | 범용 비전 백본 (Classification, Detection, Segmentation 등) | 이미지 분류 중심 | 이미지 분류, 탐지 (task-specific tuning) |

---

## 📉 실험 및 결과

- **데이터셋**:
  - ImageNet-1K (Classification)
  - COCO (Object Detection, Instance Segmentation)
  - ADE20K (Semantic Segmentation)

- **비교 모델**:
  - ViT / DeiT (Transformer 계열)
  - ResNet, ResNeXt (CNN 계열)
  - SETR, DetectoRS 등 (SOTA 모델들)

- **주요 성능 지표 및 결과 요약**:

| 모델               | Top-1 Accuracy (ImageNet) | Box AP (COCO) | Mask AP (COCO) | mIoU (ADE20K) |
| ------------------ | -------------------------- | ------------- | -------------- | ------------- |
| **Swin-T (ours)**   | 81.3%                      | 50.5          | 43.7           | 44.5          |
| ViT-B/16           | 77.9%                      | -             | -              | -             |
| ResNet-50          | 76.2%                      | 38.0          | 33.2           | 36.7          |
| DetectoRS          | -                          | 55.7          | 48.4           | -             |
| SETR-PUP           | -                          | -             | -              | 50.3          |
| **Swin-L (ours)**   | **87.3%**                  | **58.7**      | **51.1**       | **53.5**      |

> 🔍 **실험 결과 요약**:
> - Swin Transformer는 다양한 비전 작업에 대해 기존 SOTA 대비 **1~3% 이상의 성능 향상**을 보임
> - 특히 COCO와 ADE20K에서 **객체 검출 및 분할 성능이 크게 향상**
> - 효율성과 정확도 모두를 만족시키는 범용 백본으로서의 가능성을 입증

---

## ✅ 장점 및 한계

### **장점**:

- ✅ **선형적 연산 복잡도**: Local Window 기반 Self-Attention 덕분에 고해상도 이미지에서도 효율적
- ✅ **Hierarchical 구조**: CNN처럼 다단계 feature 추출이 가능하여 다양한 downstream task에 적합
- ✅ **범용성**: Classification, Detection, Segmentation 등 다양한 Task에 모두 활용 가능
- ✅ **SOTA 성능**: 기존 ViT, ResNet 계열 대비 뛰어난 성능 확보

### **한계 및 개선 가능성**:

- ⚠️ **Shifted Window 구현 복잡성**: HW 병렬화를 고려한 최적 구현이 필요
- ⚠️ **Local 정보 중심의 한계**: 완전한 전역 관계 파악에는 제한적일 수 있음
- ⚠️ **대규모 사전 학습 필요**: 성능을 최대로 끌어올리기 위해선 많은 데이터와 연산 자원이 요구됨
- ⚠️ **Self-attention 창 크기 고정**: 다양한 scale에 완전히 유연하진 않음

---


## 🧠 TL;DR – 한눈에 요약

> **Swin Transformer는 계층적 구조와 Shifted Window 기반 Self-Attention을 도입하여, 고해상도 이미지 처리에 효율적이고 다양한 컴퓨터 비전 과제에 범용적으로 활용 가능한 Transformer 기반 백본을 제안한 논문이다.**

| 구성 요소     | 설명 |
| ------------ | ----------------------------------------------------------------------------- |
| 핵심 모듈     | **Window-based Multi-head Self-Attention (W-MSA)** + **Shifted Window (SW-MSA)**를 조합한 블록 구조 |
| 학습 전략     | **Hierarchical learning** – Patch → Block 반복 + Patch Merging을 통해 다중 해상도 feature 학습 |
| 전이 방식     | 다양한 Downstream Task (Classification, Detection, Segmentation)에 FPN, U-Net 등의 구조와 연계 |
| 성능/효율성  | 기존 ViT/ResNet 대비 **Top-1 정확도 및 COCO/AP/mIoU 모두에서 SOTA 달성**, 계산량은 선형 수준 유지 |

---

## 🔗 참고 링크 (References)

* [📄 arXiv 논문](https://arxiv.org/abs/2103.14030)
* [💻 GitHub](https://github.com/microsoft/Swin-Transformer)
* [📈 Papers with Code](https://paperswithcode.com/paper/swin-transformer-hierarchical-vision)

---

## 다음 논문:
👉Masked AutoEncoder (MAE) – ViT 기반 자기지도 학습
