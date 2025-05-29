# 📘 Denoising Diffusion Probabilistic Models

## 1. 개요 (Overview)

- **제목**: Denoising Diffusion Probabilistic Models
- **저자**: Jonathan Ho, Ajay Jain, Pieter Abbeel
- **소속**: UC Berkeley
- **학회**: NeurIPS 2020
- **링크**:
  - [arXiv](https://arxiv.org/abs/2006.11239)
  - [GitHub](https://github.com/hojonathanho/diffusion)
  - [Papers with Code](https://paperswithcode.com/paper/denoising-diffusion-probabilistic-models)

---

### 📌 논문 선정 이유

DDPM은 GAN 이후 가장 주목받는 생성 모델 중 하나로, 고품질 이미지를 생성하면서도 학습이 안정적인 특성을 갖고 있다.  
Stable Diffusion, Imagen 등 최신 텍스트-투-이미지 생성기술의 기반이 되는 핵심 논문이며, 다양한 분야에 응용 가능한 범용성을 갖추고 있다.

---

### 🧠 간단한 도입부

이 논문은 이미지 생성 과정에서 **노이즈를 점진적으로 제거하는 방식**을 제안한다.  
먼저 데이터를 점차적으로 Gaussian noise로 오염시키는 **Forward Process**를 정의하고,  
그 역과정을 학습하여 노이즈로부터 원본 이미지를 복원하는 **Reverse Process**를 통해 샘플링을 수행한다.

이러한 접근은 GAN에서 흔히 발생하는 모드 붕괴나 학습 불안정 문제를 피하며,  
직관적이고 이론적으로도 안정적인 방식으로 고품질 이미지를 생성할 수 있게 한다.

---

## 2. 문제 정의 (Problem Formulation)

### 🔧 문제 및 기존 한계

기존 생성 모델(GANs, VAEs)은 다음과 같은 한계를 가진다:

- **GANs**: 고해상도 이미지 생성에 강력하지만, 모드 붕괴(mode collapse) 문제와 학습 불안정성 존재.
- **VAEs**: 수학적으로 안정적이지만, 생성 이미지의 품질이 낮고 흐릿함(blurriness) 문제 발생.
- **Autoregressive Models**: 높은 샘플 품질을 가질 수 있지만, 샘플링 속도가 매우 느리고 병렬처리 불가능.

이러한 문제를 해결하기 위해 논문은 **확률적 노이즈 제거(Denoising Diffusion)** 기반의 생성 모델을 제안한다.

---

### 💡 제안 방식: 확률적 점진적 노이즈 제거 기반 생성 모델

- 원본 이미지에 점차적으로 가우시안 노이즈를 추가하는 **Forward Process** 정의
- 학습 가능한 뉴럴 네트워크로 이를 역으로 복원하는 **Reverse Process** 학습
- Reverse Process는 조건부 확률 $p_\theta(x_{t-1} \mid x_t)$ 를 예측함
- 이 과정을 통해 순수 노이즈로부터 고품질 이미지를 생성 가능

---

### 📘 핵심 개념 정의

| 용어                     | 설명 |
|--------------------------|------|
| **Forward Process**      | 데이터 $x_0$에 점진적으로 가우시안 노이즈를 추가해 $x_1, x_2, ..., x_T$로 만드는 과정 |
| **Reverse Process**      | 학습된 모델이 $x_T$에서 시작하여 노이즈를 제거하며 $x_{T-1}, ..., x_0$을 복원하는 과정 |
| **Noise Schedule**       | 각 시점 $t$에서 노이즈를 얼마나 추가할지 결정하는 $\beta_t$ 시퀀스 |
| **Denoising Network**    | $x_t$를 입력받아 해당 시간의 노이즈 $\epsilon$을 예측하는 모델 (통상 U-Net) |
| **ELBO (Evidence Lower Bound)** | Diffusion 모델의 학습 목적함수로 사용되는 로그우도 하한 (log-likelihood lower bound) |

---

## 3. 모델 구조 (Architecture)

### 🧩 전체 구조 요약

1. **Forward Diffusion**:  
   - $x_0$ → $x_1$ → ... → $x_T$  
   - $q(x_t \mid x_{t-1}) = \mathcal{N}(x_t; \sqrt{1 - \beta_t} x_{t-1}, \beta_t I)$  
   - 전체 분포는 $q(x_{1:T} \mid x_0)$ 로 표현됨

2. **Reverse Denoising**:  
   - $x_T$ → $x_{T-1}$ → ... → $x_0$  
   - $p_\theta(x_{t-1} \mid x_t) = \mathcal{N}(x_{t-1}; \mu_\theta(x_t, t), \Sigma_\theta(x_t, t))$

3. **학습 목표**:  
   - 실제 노이즈 $\epsilon$과 예측 노이즈 $\epsilon_\theta(x_t, t)$ 사이의 MSE 손실 최소화  
   - $\mathcal{L}_{simple} = \mathbb{E}_{t, \epsilon} \left[ \| \epsilon - \epsilon_\theta(x_t, t) \|^2 \right]$

---

### 🧠 모델 구조 (네트워크 설계)

- **입력**: $x_t$ (노이즈가 섞인 이미지), $t$ (타임스텝)
- **출력**: $\epsilon_\theta(x_t, t)$ (노이즈 예측)
- **네트워크**:  
  - **U-Net 기반 구조**
  - 각 레벨마다 시간 정보 $t$를 embedding하여 추가
  - skip connection 사용
- **스케줄링**:
  - $\beta_t$는 선형(linear), cosine 등 다양한 방식으로 스케줄링 가능

---


## 💠 핵심 모듈 또는 구성 요소

---

### 📌 Forward Diffusion Process (노이즈 추가 과정)

#### 🔧 작동 방식

- 원본 데이터 $x_0$에 점진적으로 가우시안 노이즈를 추가하여 $x_t$를 생성
- 이 과정을 시간 $t = 1$부터 $T$까지 반복적으로 수행

#### 📐 수식

- 조건부 확률 분포 정의:\
  $q(x_t \mid x_{t-1}) = \mathcal{N}(x_t; \sqrt{1 - \beta_t} x_{t-1}, \beta_t I)$

- 누적 형태로 한 번에 $x_t$ 샘플링 가능:\
  $q(x_t \mid x_0) = \mathcal{N}(x_t; \sqrt{\bar{\alpha}_t} x_0, (1 - \bar{\alpha}_t) I)$

  여기서,\
  ![formula](https://latex.codecogs.com/svg.image?\alpha_t%20=%201%20-%20\beta_t,\quad%20\bar{\alpha}_t%20=%20\prod_{s=1}^{t}\alpha_s)

---

### 📌 Reverse Diffusion Process (노이즈 제거 과정)

#### 🌀 역할 및 기존 방식과의 차별점

- Forward에서 생성된 $x_t$를 이용해 $x_{t-1}$을 복원하는 확률적 역방향 과정
- 기존 GAN처럼 직접적인 이미지 생성을 시도하지 않고, **점진적인 복원 방식**을 통해 학습 안정성을 획득

#### 📐 수식

- 학습 가능한 분포:\
  ![formula](https://latex.codecogs.com/svg.image?p_\theta(x_{t-1}%20%7C%20x_t)%20%3D%20\mathcal{N}\left(x_{t-1},%20\mu_\theta(x_t,%20t),%20\Sigma_\theta(x_t,%20t)\right))



- 논문에서는 단순화를 위해 분산 $\Sigma_\theta$를 고정하거나 예측하지 않고, 평균만 예측하는 방식 사용

---

### 📌 Noise Prediction Network (Denoising Model)

#### 🔧 작동 방식

- 주어진 noisy image $x_t$와 time step $t$를 입력으로 받아 **original noise $\epsilon$**을 예측
- $\mu_\theta(x_t, t)$는 예측된 $\epsilon_\theta$로부터 계산

#### 📐 수식

- 샘플링된 $x_t$는 다음 수식으로 구성됨:\
  ![xt](https://latex.codecogs.com/svg.image?x_t%20=%20\sqrt{\bar{\alpha}_t}%20x_0%20+%20\sqrt{1%20-%20\bar{\alpha}_t}%20\epsilon,%20\quad%20\epsilon%20\sim%20\mathcal{N}(0,%20I))


- 학습 목표는 $\epsilon_\theta(x_t, t)$ ≈ 실제 노이즈 $\epsilon$ 을 맞추는 것

- 단순화된 손실 함수 (MSE 기반):\
  ![loss](https://latex.codecogs.com/svg.image?\mathcal{L}_{\text{simple}}%20=%20\mathbb{E}_{x_0,%20t,%20\epsilon}%20\left[%20\left\|%20\epsilon%20-%20\epsilon_\theta(x_t,%20t)%20\right\|^2%20\right])


---

### 📌 Sampling Algorithm (생성 알고리즘)

- Reverse Process를 이용해 $x_T \sim \mathcal{N}(0, I)$로부터 시작하여 $x_0$까지 반복적으로 샘플링

#### 🧮 간단 알고리즘 요약

```python
x_T = N(0, I)
for t in range(T, 1, -1):
    predict_noise = epsilon_theta(x_t, t)
    compute_mu = (1 / sqrt(alpha_t)) * (x_t - beta_t / sqrt(1 - bar_alpha_t) * predict_noise)
    sample x_{t-1} ~ N(mu, sigma_t^2)
```
## ⚖️ 기존 모델과의 비교

| 항목       | 본 논문 (DDPM)                                           | 기존 방법1 (GAN)                                        | 기존 방법2 (VAE)                                   |
|------------|-----------------------------------------------------------|----------------------------------------------------------|----------------------------------------------------|
| 구조       | 노이즈 추가 ↔ 제거 과정을 모사하는 확률적 프로세스           | 생성자-판별자(Generator-Discriminator) 대립 구조            | 인코더-디코더 구조에서 잠재 분포 학습                    |
| 학습 방식  | MSE 기반 노이즈 예측 학습<br> (ELBO의 간접 최적화)           | 판별자를 속이도록 생성자 업데이트 (비안정적)                  | Reconstruction + KL divergence로 로그우도 최대화     |
| 목적       | 고품질 이미지 생성을 안정적으로 수행                       | 사실적(realistic) 이미지 생성, 고해상도에 강함                | 잠재 분포를 기반으로 한 생성 모델, 수학적으로 안정적    |

---

## 📉 실험 및 결과

- **데이터셋**:
  - CIFAR-10
  - LSUN (bedroom, church)
  - CelebA-HQ
  - ImageNet 128×128

- **비교 모델**:
  - GAN variants (BigGAN, StyleGAN2)
  - PixelCNN, PixelSNAIL
  - Autoregressive Models

- **주요 성능 지표 및 결과**:

| 모델           | FID (↓) | IS (↑)  | 기타 |
|----------------|---------|--------|------|
| **DDPM**       | **3.17** (CIFAR-10) | 9.46 | 안정성 우수 |
| BigGAN         | 4.06    | 9.22   | 대규모 모델 필요 |
| PixelCNN++     | 65.93   | -      | 느린 샘플링 속도 |
| StyleGAN2      | **2.84** (best) | - | 매우 고품질 이미지 (다만 복잡) |

> **해석**: DDPM은 학습 안정성과 샘플 품질 간 균형을 잘 맞추며, SOTA GAN 수준의 성능에 도달함.  
> 특히 FID 기준으로는 StyleGAN2에 근접하며, 학습 안정성과 단순한 구조 측면에서 우수함.

---

## ✅ 장점 및 한계

### ✅ 장점:
- 학습 안정성 뛰어남 (mode collapse 없음)
- 다양한 노이즈 스케줄로 유연한 설계 가능
- 조건부 생성(Conditional generation)에 확장 용이
- 확률 기반으로 해석 가능 (ELBO 최적화 기반)

### ⚠️ 한계 및 개선 가능성:
- 샘플링 속도 느림 (T-step reverse sampling 필요)
- 초기엔 고해상도 이미지 생성에 시간 오래 걸림
- Sampling Step을 줄이기 위한 DDIM, DPM-Solver 등 후속 연구 필요

---

## 🧠 TL;DR – 한눈에 요약

- **핵심 아이디어 요약**:  
  이 논문은 데이터를 생성하기 위해 **정방향(Forward) 과정에서 데이터를 점진적으로 노이즈로 오염시키고**,  
  이후 학습된 신경망을 통해 **역방향(Reverse) 과정에서 이 노이즈를 제거하며 원본 데이터를 복원**하는 방식을 제안한다.  
  구체적으로는, 원본 이미지 $x_0$에 Gaussian noise를 일정 단계(T-step)에 걸쳐 추가하여 $x_T$에 도달하고,  
  학습된 모델 $p_\theta(x_{t-1} \mid x_t)$이 이를 거꾸로 되돌려 고품질 이미지를 생성한다.

  이 방식은 다음의 문제들을 동시에 해결한다:
  - **GAN의 모드 붕괴 및 학습 불안정성** 문제 없음
  - **VAE의 이미지 품질 저하 문제** 개선
  - **샘플 품질과 안정성의 균형**을 이룸

  학습은 단순한 MSE 손실을 통해 각 timestep에서의 실제 노이즈와 모델이 예측한 노이즈 간 차이를 최소화하는 방식으로 수행되며,  
  이는 ELBO(증거 하한)를 최적화하는 확률적 해석을 뒷받침한다.

---

- **이 논문의 기여를 한 줄로 요약**:  
  _“DDPM은 이미지를 노이즈 제거 과정을 통해 생성하는 새로운 접근을 제안하며, VAE의 안정성과 GAN의 고품질 생성을 결합한 강력한 확률 기반 생성 모델이다.”_

---

## 구성 요소 요약

| 구성 요소     | 설명                                                                 |
|--------------|----------------------------------------------------------------------|
| 핵심 모듈     | Forward Process, Reverse Process, Denoising Network                  |
| 학습 전략     | 노이즈 예측 기반 MSE 손실, ELBO 유도 기반의 간접 최적화              |
| 전이 방식     | 조건부 확장 가능 (e.g., Class-conditioned, Text-to-Image)             |
| 성능/효율성   | 높은 품질의 이미지 생성 + 높은 안정성, 그러나 느린 샘플링 속도          |

---

## 🔗 참고 링크 (References)

- 📄 [arXiv 논문](https://arxiv.org/abs/2006.11239)
- 💻 [GitHub - Official Code](https://github.com/hojonathanho/diffusion)
- 📈 [Papers with Code](https://paperswithcode.com/paper/denoising-diffusion-probabilistic-models)

📈 Papers with Code
다음 논문:Latent Diffusion Models (LDM)
