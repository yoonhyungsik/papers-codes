# 📘 High-Resolution Image Synthesis with Latent Diffusion Models

## 1. 개요 (Overview)

| 항목 | 내용 |
|------|------|
| **제목** | High-Resolution Image Synthesis with Latent Diffusion Models |
| **저자** | Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, Björn Ommer |
| **소속** | Heidelberg University (CompVis Group) |
| **학회** | CVPR 2022 |
| **링크** | [📄 arXiv](https://arxiv.org/abs/2112.10752) / [💻 GitHub](https://github.com/CompVis/latent-diffusion) / [📈 Papers with Code](https://paperswithcode.com/paper/high-resolution-image-synthesis-with-latent) |

## 2. 문제 정의 (Problem Formulation)

### 🧩 문제 및 기존 한계

- 기존 **DDPM (Denoising Diffusion Probabilistic Models)**은 픽셀 공간에서 학습하며, 고해상도 이미지 생성 시 **연산량과 메모리 사용량이 매우 크다**.
- 특히 512×512 이상의 이미지 생성에서는 **모델 크기, training time, inference time**이 병목이 되며, 확장성이 떨어진다.
- 또한 픽셀 공간에서의 노이즈 추가는 **시각적으로 비효율적이며** 복잡한 구조나 조건부 생성에 제한적이다.

### 💡 제안 방식

> "고차원 픽셀 공간이 아닌, **압축된 latent 공간에서 diffusion을 수행**하자"

- **VAE를 이용한 perceptual latent space**로 이미지의 정보를 압축  
- 이 latent 공간에서 DDPM을 적용 → 계산량 대폭 감소  
- 마지막에 VAE 디코더로 latent를 원래 이미지로 복원

→ 이로 인해 **연산량을 크게 줄이면서도 고해상도 생성 품질을 유지**할 수 있음

---

### 🔑 핵심 개념 정의

| 개념 | 설명 |
|------|------|
| **Latent Space** | VAE 인코더에 의해 추출된 저차원의 의미 있는 표현 공간. 이 공간에서 diffusion 수행 |
| **DDPM** | 점진적으로 가우시안 노이즈를 추가하고, 이를 복원하는 과정을 통해 이미지를 생성하는 모델 |
| **Perceptual Compression** | 인간의 인지 특성을 고려해 시각적으로 중요한 정보를 유지하며 압축 |
| **Conditioning** | CLIP text embedding 등 외부 정보를 latent DDPM 과정에 주입하는 메커니즘 |
| **Cross-Attention** | text/image 등 외부 조건을 이미지 생성 중간에 연결하는 방식으로 사용됨 |

---


## 3. 모델 구조 (Architecture)

### 🧩 전체 구조

LDM은 크게 다음과 같은 세 단계로 구성됩니다:

```plaintext
[Input Image x]
   ↓ (VAE Encoder: Eθ)
[Latent Representation z]
   ↓ (Latent Diffusion: ε_θ(z_t, t, c))
[Noisy z_t → z_0]
   ↓ (VAE Decoder: Dθ)
[Generated Image x̂]
```
- **$E_\theta$**: 이미지 $x$를 압축된 latent 공간 $z$로 인코딩  
- **$\epsilon_\theta$**: latent $z_t$를 반복적으로 복원하여 $z_0$에 도달  
- **$D_\theta$**: 최종 복원된 $z_0$를 이미지 공간으로 디코딩  

---

### 🧱 전반적인 블록 구성 및 입출력 흐름 (Detailed Block Structure)

Latent Diffusion Models(LDM)는 세 가지 주요 블록으로 구성되며, 각각은 생성 효율성과 품질을 동시에 보장하기 위한 역할을 수행합니다.

---

#### 1. **VAE ( $E_\theta$, $D_\theta$ ) — Latent Space로의 인코딩 및 복원**

- 입력 이미지 $x \in \mathbb{R}^{H \times W \times 3}$는 **perceptual autoencoder**를 통해 latent 공간으로 압축됩니다.
- **인코딩 단계**:\
  $z = E_\theta(x), \quad z \in \mathbb{R}^{h \times w \times c}, \quad h \ll H,\; w \ll W$\
  여기서 $z$는 low-dimensional, semantic latent representation입니다.
- 이 latent 표현은 시각적으로 중요한 정보를 보존한 채, 계산 효율이 높은 공간으로 매핑됩니다.
- **디코딩 단계**:\
  $\hat{x} = D_\theta(z)$\
  최종적으로, 복원된 이미지 $\hat{x}$는 원본 이미지 $x$에 최대한 근접하게 됩니다.
- 이 구조는 **VQ-VAE와 유사**하지만, continuous latent space를 사용합니다.

---

#### 2. **Latent Diffusion Module (LDM Core)** — Diffusion in Latent Space

- 인코딩된 latent 벡터 $z$는 이후 **확률적 forward process**에 따라 점진적으로 노이즈가 추가된 상태 $z_t$로 변환됩니다.
## Forward Diffusion 정의

**Forward diffusion 과정**:
```
q(z_t | z_0) = N(z_t; √(α̅_t) * z_0, (1 - α̅_t) * I)
```

여기서 `α̅_t = ∏(s=1 to t) α_s`는 누적된 noise schedule 계수입니다.

- **Reverse denoising 과정**은 학습 가능한 파라미터 `ε_θ`를 통해 latent 공간에서 복원합니다:

```
z_(t-1) = (1/√α_t) * (z_t - ((1 - α_t)/√(1 - α̅_t)) * ε_θ(z_t, t, c)) + σ_t * N(0, I)
```

- **학습 손실 함수**: DDPM 논문에서 유도된 simplified loss를 사용:

```
L_simple = E[z_0, t, ε] [||ε - ε_θ(z_t, t, c)||²]
```

이 손실은 `ε_θ`가 실제로 사용된 가우시안 노이즈 `ε`을 정확히 예측하도록 유도합니다.

- 조건 `c`는 text embedding이나 segmentation map 등의 외부 조건을 포함할 수 있으며, cross-attention을 통해 모델에 주입됩니다.

**디코딩 단계**:
```
x̂ = D_θ(z)
```
최종적으로, 복원된 이미지 `x̂`는 원본 이미지 `x`에 최대한 근접하게 됩니다.

---

## 요약: 전체 흐름

1. 입력 이미지 `x` → 인코딩 → `z = E_θ(x)`
2. `z`에 노이즈를 점진적으로 추가해 `z_t` 생성
3. `z_t`로부터 `ε_θ(z_t, t, c)`가 `ε`을 예측
4. 역방향 과정을 통해 `z_0` 복원
5. 복원된 latent `z_0`를 디코딩하여 최종 이미지 `x̂` 생성

---

## 💠 핵심 모듈 또는 구성 요소

### 📌 VAE 기반 인코더 및 디코더

- `E_θ`: 이미지의 시각 정보를 보존하면서 의미 압축된 표현 `z`를 생성
- `D_θ`: 학습된 latent 표현을 다시 RGB 이미지로 복원
- **핵심 아이디어**: 고해상도 이미지를 픽셀 공간이 아닌 latent 공간에서 생성함으로써 연산 효율 증가

### 📌 Latent DDPM Core (`ε_θ`)

- 기존 DDPM 구조를 latent 공간에 적용
- 시간 스텝 `t`에서 다음의 noising 식을 따름:

```
z_t = √(α̅_t) * z_0 + √(1 - α̅_t) * ε,  where ε ~ N(0, I)
```

- UNet 기반의 noise prediction 네트워크 `ε_θ`가 이 latent를 점진적으로 복원

### 📌 Cross-Attention Conditioning

- 텍스트 기반 생성의 경우, **CLIP text embedding**을 조건으로 사용
- UNet의 각 블록에서 latent feature와 텍스트 조건 간의 cross-attention 적용
- Cross-attention 계산:

```
Attn(Q, K, V) = softmax(QK^T / √d_k) * V
```

여기서 `Q`는 latent feature에서, `K, V`는 text embedding에서 유도됨

### ⚖️ 기존 모델과의 비교

| 항목       | **LDM (본 논문)**                    | **기존 DDPM**                       | **GAN 기반 생성**                 |
|------------|--------------------------------------|-------------------------------------|-----------------------------------|
| **구조**   | VAE + Latent DDPM + Cross-Attention | 픽셀 공간 DDPM                     | Generator + Discriminator         |
| **학습 방식** | 노이즈 예측 in latent space          | 노이즈 예측 in image space         | Adversarial loss + min-max 게임   |
| **목적**   | 고해상도 + 효율적인 생성            | 고품질 생성 (비효율적 계산)        | 빠른 생성 (but 불안정한 학습)     |

---

### 📉 실험 및 결과

- **데이터셋**:  
  COCO, CelebA-HQ, LSUN Church, FFHQ 등

- **비교 모델**:  
  BigGAN, ADM, Improved DDPM, VQGAN 등

- **주요 성능 지표 및 결과**:

| 모델         | FID ↓ | IS ↑ | 기타 메트릭 |
|--------------|-------|------|--------------|
| LDM          | 3.60  | -    | 텍스트 조건 생성 우수 |
| Improved DDPM | 4.59  | -    | 느린 샘플링 속도 |
| GAN 계열     | 7~20  | 9.1↑ | 빠르지만 불안정 |

- **실험 결과 요약 및 해석**:  
  LDM은 latent 공간에서 diffusion을 수행함으로써 **동일한 품질의 이미지를 더 빠르고 적은 자원으로 생성**할 수 있음.  
  텍스트 조건 생성에서도 CLIP-based conditioning을 통해 강력한 성능 달성.

---

### ✅ 장점 및 한계

#### 장점:
- 고해상도 이미지 생성에 적합한 **계산 효율성**  
- 다양한 **조건 생성** 가능 (텍스트, 마스크 등)  
- **오픈소스 구현이 활발**해 실제 제품화 가능성 높음

#### 한계 및 개선 가능성:
- VAE 기반 latent 압축으로 인해 **디테일 손실 가능성**  
- diffusion 특성상 여전히 **샘플링 속도는 느림**

---

### 🧠 TL;DR – 한눈에 요약

> 기존 DDPM이 픽셀 공간에서 수행하던 계산량 많은 diffusion 과정을, **의미적으로 압축된 latent 공간에서 수행함으로써** 생성 속도와 연산 효율을 극적으로 개선한 고해상도 이미지 생성 모델.  
> 또한 CLIP 기반의 텍스트 조건 및 다양한 modality conditioning을 통해 **범용적이고 유연한 생성 능력**을 갖춘 프레임워크를 제시함.

---

| 구성 요소     | 설명 |
|---------------|------|
| **핵심 모듈** | **1) VAE 기반 Encoder/Decoder**를 통해 이미지 데이터를 정보 밀도가 높은 latent space로 변환하고, <br> **2) 이 latent 공간에서 DDPM(reverse denoising process)를 수행**하며, <br> **3) Cross-Attention을 통해 텍스트나 이미지 등의 외부 조건을 유연하게 반영 |
| **학습 전략** | DDPM의 simplified objective인 noise prediction loss 사용:  <br> $\mathcal{L}_{\text{simple}} = \mathbb{E}_{z_0, t, \epsilon} \left[ \| \epsilon - \epsilon_\theta(z_t, t, c) \|^2 \right]$ <br> 이는 모델이 latent 공간 내 노이즈 $\epsilon$를 정확히 복원하도록 유도함 |
| **전이 방식** | CLIP 등으로부터 추출한 텍스트 임베딩을 조건 $c$로 사용하며, <br> UNet 내부의 cross-attention 블록을 통해 latent feature와 조건 임베딩을 결합. <br> 이를 통해 **텍스트, 세그멘테이션 맵, 이미지 등 다양한 modality 기반의 조건부 생성 가능** |
| **성능/효율성** | 기존 DDPM 대비 **최대 100배 이상 빠른 생성 속도** 확보. <br> 픽셀 공간이 아닌 latent 공간에서 연산이 이루어지므로 VRAM과 FLOPs 모두 절감. <br> 고해상도 이미지 (512×512 이상)에서도 SOTA 품질 달성 가능. <br> Stable Diffusion 등 실제 제품 수준 모델의 기반이 되었음. |

---

### 🔗 참고 링크 (References)

- 📄 [arXiv 논문](https://arxiv.org/abs/2112.10752)  
- 💻 [GitHub 구현](https://github.com/CompVis/latent-diffusion)  
- 📈 [Papers with Code](https://paperswithcode.com/paper/high-resolution-image-synthesis-with-latent)

---

