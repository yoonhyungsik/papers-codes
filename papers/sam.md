# 📘 Segment Anything

## 1. 개요 (Overview)

- **제목**: Segment Anything  
- **저자**: Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C. Berg, Wan-Yen Lo, Piotr Dollar, Ross Girshick  
- **소속**: Meta AI Research (FAIR)  
- **학회**: arXiv Preprint, 2023  
- **링크**:  
  - [arXiv](https://arxiv.org/abs/2304.02643)  
  - [GitHub](https://github.com/facebookresearch/segment-anything)  
  - [Papers with Code](https://paperswithcode.com/paper/segment-anything)

---

### ❓ 논문 선정 이유 및 도입부

최근 컴퓨터 비전에서 이미지 세분화(Segmentation)는 객체 검출 및 인식보다 더 정밀한 정보를 제공하는 핵심 기술로 주목받고 있다. 하지만 대부분의 세분화 모델은 사전에 정의된 카테고리나 도메인에 제한되며, 새로운 객체에 대한 일반화 능력이 부족하다.

Meta AI의 Segment Anything은 "어떤 객체든지 분할할 수 있는 범용 세분화 모델"이라는 대담한 목표 아래 개발된 모델로, **프롬프트 기반 분할(Promptable Segmentation)**이라는 새로운 패러다임을 제시한다. 이 모델은 사용자의 입력(점, 박스, 마스크 등)을 받아 실시간으로 객체 마스크를 출력할 수 있으며, 대규모 데이터셋 SA-1B로 사전학습되어 강력한 zero-shot 성능을 자랑한다.


## 2. 문제 정의 (Problem Formulation)

### ❗ 문제 및 기존 한계

이미지 세분화(Segmentation)는 컴퓨터 비전에서 중요한 과제이며, 일반적으로 아래와 같은 방식들로 수행되어 왔다:

- **Semantic Segmentation**: 클래스 레벨에서 픽셀 단위로 라벨링
- **Instance Segmentation**: 동일 클래스 내 객체 개별 구분
- **Panoptic Segmentation**: 위 두 방식의 통합

그러나 기존의 segmentation 모델들은 다음과 같은 한계를 가진다:

1. **사전 정의된 클래스에 의존**: 학습 시 사용된 객체 클래스 외에는 일반화 불가
2. **도메인 특화 학습 필요**: 의료, 위성 등 새로운 도메인마다 별도 데이터 및 학습 필요
3. **프롬프트 기반 상호작용 미지원**: 사용자의 의도를 직접 반영하기 어려움
4. **대규모 데이터 수집의 어려움**: 정밀한 픽셀 단위 GT 라벨링에는 고비용·고노동이 요구됨

---

### 💡 제안 방식: Segment Anything Model (SAM)

SAM은 위 문제를 해결하기 위해 다음과 같은 **프롬프트 기반 범용 분할(Promptable Segmentation)** 프레임워크를 제안한다:

- **다양한 프롬프트 지원**: 점, 경계 상자, 마스크 등 입력 가능
- **범용 분할 성능**: 사전 정의된 클래스 없이도 **Zero-shot**으로 동작
- **SA-1B 데이터셋**: 인간 개입 없이 생성된 **1B 마스크**로 모델 사전학습
- **효율적인 구조**: Image Encoder와 Prompt Encoder를 분리하여 실시간 상호작용 지원

> "Segment Anything can segment any object in any image with any prompt."

이를 통해 기존의 제한적 segmentation 방식을 넘어서 범용성과 실용성을 동시에 갖춘 새로운 segmentation 패러다임을 제시한다.


## 3. 모델 구조 (Architecture)

![sam_architecture](/papers/images/sam_architecture.png)

Segment Anything Model (SAM)은 다양한 프롬프트(Point, Box, Mask 등)에 대해 범용 세분화가 가능한 구조로 설계된 파운데이션 모델이다. 모델은 다음의 세 주요 모듈로 구성된다:

$$
\text{SAM}(I, P) = f_{\text{mask}}(f_{\text{img}}(I), f_{\text{prompt}}(P))
$$
- $I \in \mathbb{R}^{H \times W \times 3}$: 입력 이미지
- $P$: 프롬프트 입력 (점, 박스, 마스크 등)
- $f_{\text{img}}$: 이미지 인코더
- $f_{\text{prompt}}$: 프롬프트 인코더
- $f_{\text{mask}}$: 마스크 디코더



### 📌 Image Encoder: $f_{\text{img}}$

SAM의 Image Encoder는 입력 이미지를 고차원 임베딩으로 변환하며, 전체 분할 파이프라인의 첫 번째 단계이다. 이 인코더는 **CLIP으로 사전학습된 Vision Transformer (ViT-H/14)**를 기반으로 한다.

---

#### 🔷 입력 및 출력

- 입력 이미지:\  
  $$I \in \mathbb{R}^{H \times W \times 3}$$

- 출력 임베딩:\  
  $$F_I \in \mathbb{R}^{N \times D}$$
  - $N$: 패치 개수 (예: $N = \frac{H}{P} \cdot \frac{W}{P}$)  
  - $D$: 임베딩 차원 (예: 1024 또는 1280)

---

#### 🔷 처리 흐름

1. **패치 분할 (Patch Splitting)**  
   이미지를 $P \times P$ 크기의 패치로 분할하고 flatten:\
   $$x_i = \text{Flatten}(I_{(i)}) \in \mathbb{R}^{P^2 \cdot 3}$$

2. **선형 투영 (Linear Projection)**  
   각 패치를 임베딩 차원으로 변환:\
   $$e_i = W_p x_i + b \in \mathbb{R}^D$$

3. **위치 임베딩 추가 (Positional Encoding)**  
   패치 순서를 보존하기 위해 위치 정보 $PE_i$ 추가:\
   $$z_i = e_i + PE_i$$

4. **ViT 인코딩**  
   최종 패치 시퀀스를 ViT에 입력하여 출력 임베딩 생성:\
   $$F_I = \text{ViT}(\{z_1, z_2, \dots, z_N\})$$

---

#### 🔷 특징 및 장점

- **CLIP 사전학습 ViT 사용**: 이미지-텍스트 정렬로 일반화 성능 우수
- **One-time 인코딩**: 한 번의 인코딩으로 다양한 프롬프트에 대응 가능
- **모듈화 구조**: 이미지와 프롬프트 인코더가 분리되어 효율적

---

### 📌 Prompt Encoder: $f_{\text{prompt}}$

Prompt Encoder는 사용자로부터 입력되는 다양한 프롬프트 (점, 박스, 마스크 등)를 고차원 임베딩 공간으로 매핑하는 역할을 한다. 이 임베딩은 이후 마스크 디코더와 함께 사용되어 대상 객체의 위치와 형태를 결정짓는 핵심 조건으로 작용한다.

---

#### 🔷 입력 및 출력

- 입력 프롬프트:  
  $$P = \text{point},\ \text{box},\ \text{mask}$$

- 출력 임베딩:  
  $$E_P \in \mathbb{R}^{k \times D}$$
  - $k$: 프롬프트 수 (점 1개 = 1, 박스 = 2, 마스크 = 1 또는 CNN 피처 수)
  - $D$: 임베딩 차원 (Image Encoder와 동일)

---

#### 🔷 처리 방식

- **Point Prompt**:  
  좌표 $(x, y)$와 클래스 라벨 정보 (foreground / background)를 함께 인코딩\
$$e_{\text{point}} = \text{Embed}(x, y) + \text{TypeEmbedding}_{\text{fg/bg}}$$

- **Box Prompt**:  
  좌상단과 우하단 좌표 $(x_1, y_1), (x_2, y_2)$를 각각 점으로 처리

- **Mask Prompt**:  
  저해상도 마스크 $M \in \mathbb{R}^{h \times w}$를 CNN을 통해 피처 맵으로 변환 후 flatten

---

#### 🔷 특징

- 서로 다른 유형의 프롬프트를 통합된 임베딩 표현으로 변환
- 이미지와 분리된 독립적 모듈 구조
- 마스크 입력은 SAM을 **interactive segmentation**이나 **fine-grained refinement**로 확장하는 데 사용됨

---
### 📌 Mask Decoder: $f_{\text{mask}}$

Mask Decoder는 Image Encoder와 Prompt Encoder의 출력 임베딩을 통합하여 최종 분할 마스크를 생성한다. 이 디코더는 **cross-attention 기반의 경량 Transformer 구조**로 설계되어 있으며, 입력에 따라 다양한 해석을 지원할 수 있도록 여러 개의 마스크 후보를 출력한다.

---

#### 🔷 입력 및 출력

- 입력:
  - 이미지 임베딩: $F_I \in \mathbb{R}^{N \times D}$
  - 프롬프트 임베딩: $E_P \in \mathbb{R}^{k \times D}$

- 출력:
  - 마스크 후보: $\hat{M}_1, \hat{M}_2, \hat{M}_3 \in \mathbb{R}^{H \times W}$
  - confidence score: $s_1, s_2, s_3 \in [0, 1]$

---

#### 🔷 처리 구조

1. **Cross-Attention**:  
   프롬프트 임베딩을 쿼리 ($Q$), 이미지 임베딩을 키와 값 ($K, V$)로 사용하여 다음과 같이 attention 계산:\
   $$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

2. **Feedforward + Upsampling**:  
   attention output을 MLP 및 업샘플링 블록을 통해 최종 마스크 공간으로 투영:\
   $$\hat{M}_i = \text{Upsample}(\text{MLP}(f_{\text{attn}}(E_P, F_I)))$$

---

#### 🔷 특징

- 1개의 프롬프트에 대해 **다중 마스크 후보 출력** → 모호한 입력에 대한 유연한 대응 가능
- 각 마스크에 대해 별도의 **confidence score** 예측
- Transformer 기반이지만 가벼운 구조로 **실시간 예측 속도 보장**


### 🔁 전체 구조 요약

```text
[Input Image] ─▶ Image Encoder ─┐
                                ▼
                   [Mask Decoder] ──▶ [Mask(s) + Score(s)]
                                ▲
       [Prompt: point / box / mask] ─▶ Prompt Encoder
```

## ⚖️ 비교 분석표

| 항목         | 본 논문 (SAM)                                | 기존 방법1 (DeepLab, FCN 등)             | 기존 방법2 (Interactive Segmentation) |
|--------------|-----------------------------------------------|-------------------------------------------|----------------------------------------|
| **구조**     | Image/Prompt 분리 인코더 + Mask Decoder       | 단일 CNN 기반 인코더-디코더               | 반복 피드백 기반 예측 구조             |
| **학습 방식**| Prompt 시뮬레이션 기반 대규모 사전학습        | Supervised 학습 (픽셀 단위 GT 필요)       | 사용자 입력을 누적하며 학습            |
| **목적**     | 범용 promptable segmentation (Zero-shot 포함) | 특정 클래스의 정확한 마스크 생성         | 사용자의 목표에 점진적으로 수렴         |

---

## 📉 실험 및 결과

### ● 데이터셋
- **SA-1B (Segment Anything 1B)**:  
  - 1100만 이미지, 10억 개의 고품질 마스크 (자동 수집)
  - COCO, LVIS, ADE20K 등과도 호환 가능

### ● 비교 모델
- DeepLabv3, Mask R-CNN, InteractiveSeg, FocalClick, etc.

### ● 주요 성능 지표 및 결과 (일부 실험 발췌)

| 모델         | mIoU ↑ | F1 ↑ | Precision ↑ | 기타 |
|--------------|--------|------|-------------|------|
| **SAM (Ours)** | 83.7   | 91.2 | 92.0        | Zero-shot COCO Seg.에서 최고 성능 |
| 기존 SOTA    | ~78.3  | ~87  | ~88         | 대부분 Fully supervised 필요      |

> SAM은 사전학습만으로 다양한 세분화 태스크에서 **zero-shot** 수준의 성능을 보이며,  
> 특히 **interactive segmentation 성능에서는 기존보다 클릭 수가 적음에도 높은 정확도**를 달성함.

---

## ✅ 장점 및 한계

### 장점
- 프롬프트 기반으로 다양한 세분화 태스크에 대응 가능 (범용성)
- 대규모 사전학습을 통해 Zero-shot 성능 확보
- 빠른 응답 시간 (단일 이미지 인코딩 후 프롬프트만 인코딩)
- SA-1B 데이터셋 자동 구축 → 비용 효율성

### 한계 및 개선 가능성
- 복잡한 객체가 많은 상황에서는 모호한 프롬프트 해석 가능성 존재
- 이미지 인코더가 고정이기 때문에 세밀한 특화 태스크에는 약할 수 있음
- Mask Decoder가 다소 얕은 구조 (디코더 튜닝 여지 있음)

---
## 🧠 TL;DR – 한눈에 요약

> Segment Anything Model (SAM)은 모든 객체에 대해 프롬프트 기반으로 마스크를 예측할 수 있는 **범용 세분화 파운데이션 모델**이다.  
> 기존 세분화 모델들이 특정 클래스나 태스크에 한정된 반면, SAM은 **점, 박스, 마스크 등의 프롬프트만으로도 zero-shot 방식으로 새로운 객체와 도메인에 대응**할 수 있으며, 대규모 자동 생성 데이터셋 SA-1B를 활용해 학습되어 뛰어난 일반화 성능과 효율성을 동시에 달성한다.

---

### 🔍 핵심 설계 철학 및 아이디어

- **Promptable Segmentation Task 정의**: 기존 segmentation 방식에서 탈피해, 분할 대상 정보를 *텍스트처럼 프롬프트로 주는 방식*으로 문제를 재정의
- **프롬프트의 형태 다양화**: 점 클릭, 바운딩 박스, 저해상도 마스크 등 다양한 입력을 통해 사용자/알고리즘이 원하는 객체를 지정 가능
- **Zero-shot Generalization**: fine-tuning 없이도 unseen class, unseen domain에서 바로 마스크 예측 가능
- **Composable Model Design**: 단일 모델을 다양한 비전 시스템에 조합 가능 (e.g., detector + SAM = instance segmentation)

---

### 🧠 SAM의 구조적 특징

| 구성 요소       | 설명 |
|----------------|------|
| **Image Encoder**  | CLIP으로 사전학습된 ViT-H를 사용하여 이미지의 고정 임베딩 추출. 프롬프트가 변경되어도 인코딩 재사용 가능 |
| **Prompt Encoder** | 점/박스/마스크 프롬프트를 동일 임베딩 공간으로 투영. positional + semantic 정보를 함께 반영 |
| **Mask Decoder**   | cross-attention 기반의 경량 Transformer로 프롬프트와 이미지 간 상호작용을 통해 다중 마스크 후보를 생성함 |

---

## 🔗 참고 링크 (References)

- 📄 [arXiv 논문](https://arxiv.org/abs/2304.02643)
- 💻 [GitHub - facebookresearch/segment-anything](https://github.com/facebookresearch/segment-anything)
- 📈 [Papers with Code](https://paperswithcode.com/paper/segment-anything)

---
다음 논문: Auto-Encoding Variational Bayes
