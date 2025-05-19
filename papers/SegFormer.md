# 📘 [SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers]

## 1. 개요 (Overview)

* **제목**: SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers  
* **저자**: Enze Xie, Wenhai Wang, Zhiding Yu, Anima Anandkumar, Jose M. Alvarez, Ping Luo  
* **소속**: The Chinese University of Hong Kong, NVIDIA, Caltech  
* **학회**: NeurIPS 2021  
* **링크**: [arXiv](https://arxiv.org/abs/2105.15203) / [GitHub](https://github.com/NVlabs/SegFormer) / [Papers with Code](https://paperswithcode.com/paper/segformer-simple-and-efficient-design-for)


> 논문 선정 이유 및 간단한 도입부 작성

최근 Vision Transformer(ViT)의 등장은 이미지 분류를 넘어 다양한 컴퓨터 비전 분야로 확장되고 있다. 그러나 semantic segmentation과 같은 dense prediction task에 Transformer를 적용하는 데에는 여전히 계산량, 구조 복잡성, positional encoding 의존성 등의 여러 제약이 존재한다.

이 논문은 이러한 한계를 극복하고자 **효율성과 성능을 모두 고려한 새로운 구조의 segmentation 모델, SegFormer를 제안**하였다. 특히 CNN 기반 백본 없이도 계층적 feature 표현을 Transformer로 직접 구성하며, MLP 기반의 단순한 decoder 구조로도 높은 정확도를 달성하는 점이 흥미롭다.

### 📌 선정 이유
- **Transformer 기반 구조**임에도 **계층적 특성과 연산 효율성**을 동시에 달성
- 복잡한 decoder 없이도 SOTA 수준의 성능을 보이는 **단순한 MLP decoder 설계**
- **Positional Encoding 없이도** 강력한 성능 달성 → ViT의 구조적 한계를 넘는 방식으로 주목

SegFormer는 Vision Transformer의 구조적 유연성을 semantic segmentation에 성공적으로 확장한 사례로, **향후 경량화된 Transformer 모델 설계에 있어 중요한 참고가 될 수 있다.**

---

## 2. 문제 정의 (Problem Formulation)

**문제 및 기존 한계**:

**ViT 기반 segmentation 모델의 문제점**
  - ViT는 입력을 고정된 크기의 patch로 나누고, 단일 해상도의 feature map만 생성함.
  - **단일 스케일 출력**으로 인해 다양한 해상도 정보 부족 (fine + coarse feature 병합 불가).
  - 높은 연산 비용 → 대규모 입력 이미지에 비효율적.
  - **Positional Encoding**에 의존 → 입력 해상도가 변경되면 성능 저하 발생.
**Decoder 설계의 소홀함**
   - 대부분의 연구가 encoder 구조에 집중.
   - 복잡하거나 무거운 decoder 구조 또는 CNN decoder를 그대로 사용하는 경우가 많음.
   - encoder가 생성한 feature를 효과적으로 활용하지 못함.

**제안 방식**:
**Positional Encoding 없이도 성능이 좋은 계층적 Transformer Encoder**
   - positional embedding 제거 → 다양한 해상도에서도 안정적 동작
   - 로컬-글로벌 정보를 모두 포착할 수 있도록 **계층적 (pyramid-like) 구조** 채택

**경량화된 All-MLP Decoder**
   - 복잡한 모듈 없이 multi-scale feature를 간단한 MLP로 결합
   - Transformer의 다양한 layer에서 얻은 local-global attention 정보를 효과적으로 통합

> ※ **핵심 개념 정의 (예: Masked LM, Next Sentence Prediction 등)**
| 개념 | 설명 |
|------|------|
| **Hierarchical Transformer Encoder** | 고해상도에서 저해상도로 이어지는 multi-scale 구조. 각 단계는 Transformer layer로 구성되어 있고, positional encoding이 없음. |
| **All-MLP Decoder** | multi-scale feature들을 각 해상도에 맞게 upsample 후, MLP를 통해 병합하는 간단한 구조. 복잡한 convolution 모듈 없이도 성능 확보. |
| **Positional Encoding Free** | 위치 정보를 학습에 직접적으로 입력하지 않으며, 구조적으로 local-global 정보를 병합함으로써 이를 대체. |
---

## 3. 모델 구조 (Architecture)

### 전체 구조

![segformer_architecture](/papers/images/segformer_architecture.png)

SegFormer는 Vision Transformer의 강점을 활용하되, 기존의 ViT 기반 segmentation 모델들이 가진 한계를 극복하기 위해 **계층적 multi-scale encoder**와 **경량화된 MLP 기반 decoder**로 구성된 구조를 제안한다.

---

### 💠 핵심 구성 요소 및 작동 방식

#### 📌 Hierarchical Transformer Encoder

SegFormer의 encoder는 ViT 구조를 계층화(hierarchical)한 형태로, 입력 이미지를 단계별로 downsampling 하며 **multi-scale token feature**를 생성한다. 각 단계는 다음과 같은 과정을 포함한다:

- **Overlapping Patch Embedding**  
  각 입력 이미지를 kernel size $3 \times 3$, stride $2$, padding $1$인 convolution으로 분할하고, Linear projection을 통해 token으로 변환한다.

  $$z^0 = \text{Conv}_{3\times3}(x)$$

- **Transformer Block 반복**  
  각 스테이지마다 $L_i$개의 Transformer block이 존재하며, 각 block은 다음 연산으로 구성된다:

$$
\begin{aligned}
z' &= \text{MSA}(\text{LN}(z)) + z \\
z^{\text{out}} &= \text{MLP}(\text{LN}(z')) + z'
\end{aligned}
$$


  여기서 MSA는 Multi-head Self-Attention, MLP는 두 개의 Linear layer와 GELU 활성화 함수로 구성된다.

- **Downsampling between stages**  
  각 스테이지 사이에는 resolution을 줄이고 채널 수를 늘리기 위한 **patch merging** 연산이 수행된다.

**총 4개의 스테이지**에서 각각 다음과 같은 해상도와 채널 수를 생성:

| Stage | Resolution                | Channels |
|-------|---------------------------|----------|
| 1     | $\frac{H}{4} \times \frac{W}{4}$   | 64       |
| 2     | $\frac{H}{8} \times \frac{W}{8}$   | 128      |
| 3     | $\frac{H}{16} \times \frac{W}{16}$ | 320      |
| 4     | $\frac{H}{32} \times \frac{W}{32}$ | 512      |

**Positional Encoding 제거**:  
SegFormer는 positional embedding을 제거하고, self-attention 구조 자체에서 공간 관계를 학습하게 함으로써 **해상도 독립적** 성능을 보장한다.

---

#### 📌 All-MLP Decoder

SegFormer의 decoder는 convolution 또는 attention 연산 없이, 각 스테이지의 feature를 단순 upsample하고 MLP로 결합하는 **매우 단순한 구조**이다.

##### 구성 단계:

1. **Feature Upsampling**:  
   각 스테이지에서 나온 feature $F_i$를 동일 해상도(1/4 resolution)로 bilinear upsampling한다:

   $$\hat{F}_i = \text{Upsample}(F_i) \quad \text{for } i = 1,2,3,4$$

2. **Feature Concatenation & MLP**:  
   Upsample된 feature들을 채널 차원으로 concat한 후, linear projection을 수행한다:
   
$$
F_{\text{concat}} = \text{Concat}(\hat{F}_1, \hat{F}_2, \hat{F}_3, \hat{F}_4)
$$

$$
y = \text{MLP}(F_{\text{concat}})
$$


4. **Segmentation Output**:  
   최종적으로 segmentation map으로 변환하기 위해 1×1 conv 후 원래 해상도로 upsample한다.

##### 왜 All-MLP인가?

- encoder에서 이미 rich한 표현을 얻었기 때문에 복잡한 decoder 구조가 필요 없음
- 낮은 레이어는 local 정보, 높은 레이어는 global 정보를 갖고 있어 단순 MLP로 결합해도 충분

---

### 📌 구조적 설계 철학

SegFormer는 다음과 같은 철학을 기반으로 설계되었다:

| 요소 | 기존 방법 (SETR, PVT 등) | SegFormer |
|------|--------------------------|-----------|
| Encoder 구조 | 단일 해상도 ViT | 계층적 Transformer (multi-scale) |
| Positional Encoding | 필수 | 제거 |
| Decoder | 복잡한 CNN / ASPP | All-MLP (비교적 단순) |
| 해상도 대응 | 낮음 | 해상도 변화에 강건 |
| 로컬-글로벌 분리 | 불명확 | 계층적 attention으로 분리 |

---

### 🔬 개념 요약

- **Overlapping Patch Embedding**: CNN과 유사하게 지역 정보를 보존하는 token 분할 방식
- **Hierarchical Feature Map**: 다양한 공간 크기의 feature 생성 가능
- **Position-free Attention**: 위치 정보를 직접 주입하지 않아도 attention 구조로 공간 인식 가능
- **Decoder 경량화**: 연산량은 줄이되, 성능은 유지 또는 향상

SegFormer는 이러한 설계를 통해 **효율성, 정확성, 해상도 유연성** 세 가지를 동시에 만족시키는 rare한 구조로 평가받는다.


---

## ⚖️ 기존 모델과의 비교

| 항목         | SegFormer (본 논문)                     | SETR (ViT 기반)                 | DeepLabV3+ (CNN 기반)      |
|--------------|------------------------------------------|----------------------------------|-----------------------------|
| 구조         | Hierarchical Transformer + All-MLP       | Flat ViT + CNN decoder           | ResNet backbone + ASPP      |
| 학습 방식    | Positional Encoding 없음, multi-scale   | Positional Encoding 있음, 단일 scale | CNN 기반 End-to-End       |
| 목적         | 경량화 + 정확도 + 해상도 유연성           | Transformer 성능 확인 중심         | 고정 해상도에서 성능 최적화 |

---

## 📉 실험 및 결과

* **데이터셋**:
  - ADE20K
  - Cityscapes
  - COCO-Stuff

* **비교 모델**:
  - DeepLabV3+
  - SETR
  - Swin Transformer
  - PVT (Pyramid Vision Transformer)

* **주요 성능 지표 및 결과**:

| 모델            | mIoU (ADE20K) | mIoU (Cityscapes) | FPS (Cityscapes) | 모델 크기 |
|-----------------|---------------|-------------------|------------------|------------|
| SegFormer-B0    | 37.4%         | 71.9%             | 48 FPS           | 3.7M       |
| SegFormer-B5    | **51.8%**     | **84.0%**         | 22 FPS           | 84.7M      |
| SETR-MLA        | 48.6%         | 79.3%             | < 5 FPS          | 308M       |
| DeepLabV3+      | ~45%          | ~78%              | 18–30 FPS        | >50M       |

> **실험 결과 요약 및 해석**:  
> SegFormer는 기존 ViT 기반 모델보다 훨씬 적은 연산량과 파라미터로도 높은 정확도를 달성함. 특히 작은 모델인 B0은 **ICNet 대비 60% 빠르고 4.2% 더 높은 정확도**, 큰 모델인 B5는 SETR보다 **1.8% 더 높은 mIoU와 5배 더 빠른 속도**를 보인다.

---

## ✅ 장점 및 한계

### **장점**:

- **계층적 Transformer 구조**를 통해 CNN처럼 multi-scale feature를 생성함
- **Positional Encoding 없이도 강력한 성능** 확보 (해상도 변화에 강건)
- **간단한 All-MLP Decoder**로 연산량을 줄이면서도 SOTA 성능 달성
- 다양한 입력 해상도와 상황에서도 **범용성 높은 성능**
- 실제 inference 속도(48 FPS, B0 기준)가 빠르고 경량 모델도 잘 작동

---

### **한계 및 개선 가능성**:

- Fully Transformer 구조이므로 **pre-training에 의존**하는 경향 있음 (ImageNet 등)
- ViT와 동일하게 **데이터 부족 환경에서는 학습 어려움**
- Decoder가 단순 MLP이므로 **복잡한 구조를 요구하는 scene parsing에는 한계 가능성**
- **Patch Embedding은 고정 resolution 기반이므로** 아주 작은 객체 표현은 어려울 수 있음

---

## 🧠 TL;DR – 한눈에 요약

> SegFormer는 **Positional Encoding 없이도 성능을 유지하는 Hierarchical Transformer Encoder**와 **복잡한 연산 없이도 강력한 표현력을 제공하는 All-MLP Decoder**를 결합하여, **효율성, 정확도, 해상도 적응성**을 모두 만족하는 차세대 semantic segmentation 모델을 제안한다.  
> 특히, 기존 Vision Transformer 기반 segmentation 모델들의 한계였던 **단일 해상도 처리, 고정된 positional encoding, 높은 연산량**을 구조적으로 극복하며, 경량 모델(B0)부터 대형 모델(B5)까지 **SOTA 성능과 빠른 FPS**를 동시에 달성한다.

---

| 구성 요소    | 설명 |
|-------------|------|
| 핵심 모듈    | **Hierarchical Transformer Encoder** + **All-MLP Decoder** 구조. CNN처럼 multi-scale feature 추출 후, MLP로 결합 |
| 학습 전략    | **End-to-End supervised learning** using ImageNet pre-trained weights, positional encoding 없이도 학습 가능 |
| 전이 방식    | 다양한 해상도와 입력 크기에서도 안정적인 성능을 보이며, ADE20K/Cityscapes/COCO-Stuff 등으로 fine-tuning |
| 성능/효율성 | 파라미터 수와 연산량을 줄이면서도 기존 SOTA 모델(DeepLabV3+, SETR 등) 대비 **더 높은 정확도와 FPS** 달성 |

---

## 🔗 참고 링크 (References)

* [📄 arXiv 논문](https://arxiv.org/abs/2105.15203)
* [💻 GitHub - NVlabs/SegFormer](https://github.com/NVlabs/SegFormer)
* [📈 Papers with Code](https://paperswithcode.com/paper/segformer-simple-and-efficient-design-for)


## 다음 논문: SAM
