📘 Auto-Encoding Variational Bayes

## 1. 개요 (Overview)

- **제목**: Auto-Encoding Variational Bayes  
- **저자**: Kingma, Diederik P.; Welling, Max  
- **소속**: University of Amsterdam  
- **학회**: ICLR 2014  
- **링크**:  
  - [arXiv](https://arxiv.org/abs/1312.6114)  
  - [GitHub (VAE example)](https://github.com/pytorch/examples/tree/main/vae)  
  - [Papers with Code](https://paperswithcode.com/paper/auto-encoding-variational-bayes)

### 논문 선정 이유 및 도입부

이 논문은 **Variational Autoencoder (VAE)**라는 생성 모델의 기초를 제시한 대표적인 연구로, 복잡한 확률 분포를 근사하기 위한 **Variational Inference**와 딥러닝을 결합한 최초의 프레임워크입니다.  

재파라미터라이제이션 트릭을 활용해 **gradient flow가 가능한 잠재 변수 모델 학습**을 가능케 하였으며, 이후 GAN, Flow, Diffusion 모델 등 다양한 생성모델 연구에 큰 영향을 끼쳤습니다.  
딥러닝 기반 생성모델의 출발점이 된 핵심 논문입니다.

## 2. 문제 정의 (Problem Formulation)

### 🔧 문제 및 기존 한계

- **잠재 변수 모델(latent variable model)**에서는 관측 데이터 $x$에 대해 그 원인을 설명하는 잠재 변수 $z$를 가정하고,  
  $p(x) = \int p(x \mid z) p(z) \, dz$
  와 같은 형태로 데이터를 모델링합니다.
  
- 그러나 이 모델의 **후방 분포 $p(z \mid x)$**는 복잡하고, 정확한 계산이 어렵기 때문에 일반적으로 근사를 사용해야 합니다.
  
- 기존에는 **Markov Chain Monte Carlo (MCMC)**나 **변분 추론(Variational Inference)** 기반 기법들이 사용되었으나, 계산량이 크고 샘플링이 느리며 딥러닝과 결합하기 어렵다는 한계가 있었습니다.

---

### 💡 제안 방식 (Auto-Encoding Variational Bayes)

- **Encoder-Decoder 구조**를 갖는 **변분 오토인코더(Variational Autoencoder, VAE)**를 제안

- 근사 posterior 분포 $q_\phi(z \mid x)$를 신경망으로 모델링하고, 이를 통해 **ELBO (Evidence Lower Bound)**를 최대화하는 방식으로 학습함:\
$$\log p(x) \geq E_{q_\phi(z | x)}[\log p_\theta(x | z)] - D_{KL}(q_\phi(z | x) \parallel p(z))$$
- 또한, **Reparameterization Trick**을 도입하여 샘플링 과정을 미분 가능하게 만들어 backpropagation으로 학습할 수 있도록 함

---

### 🧠 핵심 개념 정의

- **Latent Variable**: 관측되지 않은 잠재 요인. 예: 이미지 생성 시 스타일, 내용, 배경 등

- **Variational Inference**: 복잡한 posterior를 tractable한 $q(z \mid x)$로 근사하여 최적화

- **ELBO (Evidence Lower Bound)**:  
  $$\mathcal{L} = E_{q_\phi(z | x)}[\log p_\theta(x | z)] - D_{KL}(q_\phi(z | x) \parallel p(z))$$
  → 모델 학습 시 최적화되는 목표 함수

- **Reparameterization Trick**:  
  $$z = \mu + \sigma \cdot \epsilon,\quad \epsilon \sim \mathcal{N}(0, I)$$
  → 샘플링을 deterministic하게 재구성하여 gradient 전파 가능


## 3. 모델 구조 (Architecture)

### 🏗 Overall Architecture

Auto-Encoding Variational Bayes (VAE)는 확률적 생성 모델의 잠재 변수 $z$에 대해, 관측된 데이터 $x$를 최대한 잘 설명할 수 있도록 **approximate inference**를 학습하는 프레임워크다. 

모델은 다음과 같은 구성으로 이루어진다:

$$
x \xrightarrow{\text{Encoder } q_\phi(z \mid x)} (\mu, \sigma^2) \xrightarrow{\text{Sampling}} z \xrightarrow{\text{Decoder } p_\theta(x \mid z)} \hat{x}
$$

학습의 목표는 $\log p_\theta(x)$를 직접 최대화하는 것이지만, 이 항은 다음과 같은 난해한 적분을 포함한다:

$$
\log p_\theta(x) = \log \int p_\theta(x \mid z) p(z) \, dz
$$

이 때문에, tractable한 분포 $q_\phi(z \mid x)$를 통해 아래와 같은 **Evidence Lower Bound (ELBO)**를 유도하고 이를 최대화함으로써 근사 학습을 수행한다:

$$ \log p_\theta(x) \geq E_{q_\phi(z | x)}[\log p_\theta(x | z)] - D_{KL}(q_\phi(z | x) \parallel p(z)) $$

---

### 🔷 Encoder: $q_\phi(z|x)$
- 입력 $x$로부터 잠재 변수 $z$의 posterior 분포를 근사하는 인퍼런스 모델이다.
- 신경망을 통해 평균 $\mu_\phi(x)$와 로그 분산 $\log \sigma_\phi^2(x)$를 출력하며, 이를 통해 다음과 같은 다변량 정규분포를 정의한다:\
  $$q_\phi(z | x) = \mathcal{N}(z; \mu_\phi(x), \text{diag}(\sigma^2_\phi(x)))$$
- 이 분포는 실제 posterior $p(z|x)$를 근사하며, KL 발산 항을 통해 prior $p(z)$와의 정렬을 학습한다.

---

### 🔷 Latent Variable: $z$
- 잠재 변수 $z$는 데이터 $x$의 생성 원인을 내포한 저차원의 추상 표현이며, prior 분포로는 isotropic Gaussian을 사용:\
  $$p(z) = \mathcal{N}(0, I)$$
- VAE의 목적은 학습된 posterior $q_\phi(z|x)$가 이 prior와 정렬되도록 유도하는 것

---

### 🔷 Reparameterization Trick
- $z \sim q_\phi(z|x)$에서 직접 샘플링하면 gradient 전파가 불가능하므로, 이를 미분 가능한 형태로 변환:\
  $$z = \mu_\phi(x) + \sigma_\phi(x) \cdot \epsilon,\quad \epsilon \sim \mathcal{N}(0, I)$$
- 이 방식은 샘플링을 deterministic한 연산처럼 다루어 backpropagation이 가능하게 만든다.
- 이 기법 덕분에 VAE는 end-to-end로 학습 가능하며, 이는 본 논문의 핵심 기여 중 하나다.

---

### 🔷 Decoder: $p_\theta(x|z)$
- 생성 모델로, 잠재 변수 $z$를 입력받아 관측 데이터 $x$를 복원한다.
- 일반적으로 다음과 같은 조건부 정규분포 형태를 가정한다:\
  $$p_\theta(x | z) = \mathcal{N}(x; f_\theta(z), \sigma^2 I)$$
- 여기서 $f_\theta(z)$는 디코더 신경망으로, 잠재 표현 $z$를 입력으로 받아 재구성된 $x$를 출력한다.
- Reconstruction loss는 다음과 같이 유도된다:\
  $$E_{q_\phi(z | x)}[\log p_\theta(x | z)] \approx -\frac{1}{2\sigma^2} \| x - f_\theta(z) \|^2 + \text{const}$$

---

### 🔷 Overall Objective: ELBO
- 전체 학습 목표는 다음의 ELBO를 최대화하는 것이다:\
  $$\mathcal{L}(x; \theta, \phi) = E_{q_\phi(z | x)}[\log p_\theta(x | z)] - D_{KL}(q_\phi(z | x) \parallel p(z))$$
- 첫 번째 항은 $z$를 통해 $x$를 얼마나 잘 복원했는지를 나타내며 (likelihood),
- 두 번째 항은 posterior가 prior와 얼마나 가까운지를 측정하는 정규화 항이다.
- KL divergence 항은 정규분포 간 닫힌 형태(closed form)로 계산된다:\
  $$D_{KL}(q_\phi(z | x) \parallel p(z)) = \frac{1}{2} \sum_{i=1}^d \left( \mu_i^2 + \sigma_i^2 - \log \sigma_i^2 - 1 \right)$$
  
---

### 내 말로 정리하는 VAE

## 1. VAE란 무엇인가?

**VAE (Variational Autoencoder)**는 기존의 Autoencoder 구조를 기반으로 하되, **확률적인(latent) 분포를 학습**하여 새로운 데이터를 생성할 수 있도록 고안된 **생성 모델(Generative Model)**이다.

> 목표: 현실 데이터와 유사한 데이터를 생성하는 것 → 입력 데이터의 **실제 분포 p(x)** 를 근사하는 것

---

## 2. VAE의 전체 구조

VAE는 크게 **세 가지 구성 요소**로 이루어진다:

- **Encoder**: 입력 데이터를 잠재 공간(latent space)의 확률 분포로 매핑  
- **Latent Space**: 학습된 잠재 벡터 z를 저장하는 공간  
- **Decoder**: 잠재 벡터 z를 이용해 원래 데이터를 복원하거나 새로운 데이터를 생성

Input x → [Encoder] → z (latent) → [Decoder] → Output x'

## 3. Encoder: 입력을 확률 분포로 변환

### 역할
- 입력 $x$를 받아서 잠재 변수 $z$의 분포 $q(z|x)$를 추정
- VAE에서는 이 분포를 **정규분포 $\mathcal{N}(\mu, \sigma^2)$**로 가정

### 학습 내용
- 입력 데이터를 인코딩한 뒤,
  - 평균 $\mu$
  - 표준편차 $\sigma$
  를 추정하여 latent variable의 확률 분포를 정의함

$q(z|x) = \mathcal{N}(\mu(x), \sigma^2(x))$

## 4. 🧭 Latent Space: 잠재 공간의 벡터 표현

### ❗ 문제점
- 일반적인 Autoencoder는 latent space가 **불연속적**이고 **의미 없는 공간**이 될 수 있음
- 따라서 새로운 z를 샘플링해도 유효한 x를 생성하지 못할 수 있음

### ✅ 해결책: Reparameterization Trick

잠재 벡터 z는 다음과 같이 재정의됨:

$$
z = \mu + \sigma \cdot \epsilon, \quad \epsilon \sim \mathcal{N}(0, 1)
$$
- $\epsilon$: 표준 정규분포에서 샘플링한 노이즈
- $\mu, \sigma$: Encoder에서 추출한 평균과 표준편차
> 이 방식은 확률적 sampling을 deterministic 함수로 변형하여 **역전파(Backpropagation)**이 가능하게 함
---
## 5. 🛠 Decoder: z를 x로 복원
- Decoder는 잠재 벡터 $z$를 입력으로 받아 원래 입력 $x$를 복원
- 우리가 추정하고자 하는 분포는:
$$p(x|z)$$
> 학습이 완료되면 **Decoder만으로도 새로운 데이터를 생성**할 수 있게 됨
---
## 6. 🎯 VAE 학습 목표: Likelihood 최대화
### 기본 목표
입력 데이터 $x$의 marginal likelihood를 최대화:
$$\log p(x) = \log \int p(x|z) p(z) \, dz$$
→ 하지만 위 항은 직접 계산이 불가능
### ▶ 해결책: Evidence Lower Bound (ELBO)
---
## 7. 📉 Evidence Lower Bound (ELBO)
마르코프 부등식과 변분 추정을 통해 다음 식을 얻을 수 있음:\
$$
\log p(x) \geq \mathrm{E}_{q(z|x)}[\log p(x|z)] - D_{\mathrm{KL}}(q(z|x) \| p(z))
$$
### 항목별 의미
| 항 | 이름 | 설명 |
|----|------|------|
| $\mathbb{E}_{q(z\|x)}[\log p(x \| z)]$ | Reconstruction Term | Decoder가 x를 얼마나 잘 복원하는지를 측정하는 항. Autoencoder의 Reconstruction Loss와 유사 |
| $D_{KL}[q(z \| x) \| p(z)]$ | Regularization Term | 잠재 공간에서 $q(z \| x)$가 prior 분포 $p(z) \sim \mathcal{N}(0,1)$와 얼마나 유사한지를 측정하는 항. 모델이 의미 있는 latent space를 갖도록 함 |

> 따라서 **ELBO를 최대화하는 것이 VAE의 학습 목표**가 됨

---

## 8. 🔍 ELBO 유도 간단 설명
ELBO는 다음 식에서 유도됨:

$$\log p(x) = \mathbb{E}_{q(z|x)}[\log p(x|z)] - D_{KL}(q(z|x)\|p(z)) + D_{KL}(q(z|x)\|p(z|x))$$

- 마지막 항 $D_{KL}(q(z|x)\|p(z|x)) \geq 0$
- 따라서 우변의 나머지 두 항이 **Evidence Lower Bound (ELBO)**가 됨

$$\log p(x) \geq \mathbb{E}_{q(z|x)}[\log p(x|z)] - D_{KL}[q(z|x)\|p(z)]$$

---

## 9. 🎨 학습 후 데이터 생성 (Sampling)
학습이 완료된 후, 새로운 데이터를 생성하는 과정:

1. 잠재 공간에서 새로운 벡터 $z$를 샘플링:

$z \sim \mathcal{N}(0, 1)$

2. 이를 decoder에 입력:

$x_{\text{new}} = \text{Decoder}(z)$

→ 이 방식으로 **완전히 새로운 데이터를 생성**할 수 있음

---## 🔚 최종 정리
| 구성 요소 | 역할 |
|-----------|------|
| **Encoder** | 입력 $x$ → 잠재 분포 $q(z|x)$ 추정 |
| **Latent Space** | $z$ 벡터 공간. Reparameterization trick 적용 |
| **Decoder** | $z$ → $x$ 복원. $p(x|z)$ 학습 |
| **목적 함수** | ELBO = Reconstruction Term + Regularization Term |
| **학습 방식** | ELBO를 최대화하여 $\log p(x)$를 근사 |
| **생성 방식** | $z \sim \mathcal{N}(0,1) \Rightarrow x = \text{Decoder}(z)$ |
---
### ⚖️ Comparison with Deterministic Autoencoder
| Component     | Traditional Autoencoder         | Variational Autoencoder (VAE)             |
|---------------|----------------------------------|--------------------------------------------|
| Latent $z$    | Deterministic vector             | Probabilistic latent variable              |
| Encoder       | $z = f(x)$                       | $q_\phi(z \mid x) = \mathcal{N}(\mu, \sigma^2)$ |
| Decoder       | $\hat{x} = g(z)$                 | $p_\theta(x \mid z)$ (likelihood model)    |
| Objective     | $\|x - \hat{x}\|^2$              | ELBO (reconstruction + KL divergence)      |
| Regularization| L2 weight decay (optional)       | Prior-matching via KL divergence           |
## 4. 실험 및 결과 (Experiments & Results)

| 항목       | 본 논문 (VAE)                                      | 기존 방법 1 (AE)                          | 기존 방법 2 (MCMC 기반 Variational Inference) |
|------------|----------------------------------------------------|-------------------------------------------|----------------------------------------------|
| 구조       | Probabilistic encoder-decoder + latent sampling   | Deterministic encoder-decoder             | Probabilistic graphical model + sampling     |
| 학습 방식  | ELBO maximization via SGD + reparameterization    | MSE minimization via SGD                 | Variational EM or Gibbs sampling              |
| 목적       | Efficient variational inference with deep nets    | Dimensionality reduction & reconstruction | Exact posterior inference                    |

---

### 📉 실험 및 결과

- **데이터셋**:
  - MNIST (handwritten digits)
  - Frey Faces (face image sequences)
  - Synthetic 2D data (for visualization of latent space)

- **비교 모델**:
  - Traditional Autoencoder (AE)
  - Deep Belief Network (DBN)
  - Factorial Mixture Models (for probabilistic comparison)

- **주요 성능 지표**:
  - 정량 지표: **Negative log-likelihood (NLL)** (lower is better)
  - 정성 지표: **샘플링된 이미지의 시각적 품질**, latent space interpolation

| 모델         | Accuracy | F1 Score | NLL ↓       | 기타 |
|--------------|----------|----------|-------------|------|
| 본 논문 (VAE)| -        | -        | ~86.6 (MNIST)| Latent interpolation, smooth manifold |
| 기존 AE      | -        | -        | ~103         | Latent space not smooth               |
| 기존 DBN     | -        | -        | ~84.6        | Better NLL, but harder to train       |

> **Note**: VAE는 classification 성능보다는 **생성 확률 모델로서의 적절성**(NLL, 샘플 품질, latent 구조 등)에 초점을 둔 실험을 수행합니다.

---

### 실험 결과 요약 및 해석

- VAE는 MNIST에서 기존 AE보다 훨씬 낮은 NLL을 기록하며, probabilistic 모델로서 더 나은 성능을 보임
- latent space가 연속적이며 의미 있는 interpolation을 가능케 함
- 샘플링된 이미지는 DBN 대비 약간 품질이 낮지만, 학습과 샘플링의 효율성과 해석 가능성이 뛰어남
- Frey Faces에서도 smooth한 face transition 가능

---

## ✅ 장점 및 한계

### 장점

- 확률 모델 기반의 end-to-end 학습 가능 (SGD + backprop)
- 복잡한 posterior 분포를 neural network로 근사 가능
- Reparameterization trick을 통해 샘플링 기반 추론이 gradient-friendly하게 처리됨
- latent space가 smooth하고 의미 있는 구조를 가짐 (generative prior와의 정렬)

### 한계 및 개선 가능성

- Gaussian 가정 하의 복원 품질 제한 (샘플이 blurry)
- Likelihood 기반 학습은 정확한 분포 모델링에 한계
- Posterior expressiveness가 제한적 (단일 Gaussian)
- 이후 논문에서는 이를 개선하기 위해 VAE-GAN, Flow-based VAE, Hierarchical VAE 등이 등장함


## 🧠 TL;DR – 한눈에 요약

**🔹 핵심 아이디어 요약**

> Auto-Encoding Variational Bayes (VAE)는 **intractable한 베이지안 추론**을 **신경망 기반 근사 posterior와 reparameterization trick**을 통해 효율적으로 학습할 수 있도록 만든 **최초의 end-to-end deep generative model**이다.

- 기존의 MCMC 또는 Variational EM 기반 추론은 계산 복잡도와 gradient 흐름 문제로 deep learning과 결합이 어려웠음
- VAE는 이를 해결하기 위해 **(1) 근사 posterior 분포를 인코더 네트워크로 모델링**하고, **(2) 샘플링을 미분 가능하게 만드는 reparameterization trick**을 도입하여 학습 가능성을 확보함
- 학습 목표는 evidence lower bound (ELBO)를 최대화하는 것:
  $$
  \log p(x) \geq \mathbb{E}_{q(z \mid x)}[\log p(x \mid z)] - D_{\mathrm{KL}}(q(z \mid x) \parallel p(z))
  $$
- 결과적으로 VAE는 **확률적 생성 모델과 representation learning을 자연스럽게 통합**하며, 이후 수많은 generative model의 기반이 됨

---

### 📦 구성 요소 정리

| 구성 요소       | 설명 |
|----------------|------|
| **핵심 모듈**     | - Encoder $q_\phi(z \mid x)$: 입력을 정규분포 파라미터 $(\mu, \sigma)$로 매핑<br>- Decoder $p_\theta(x \mid z)$: 샘플링된 $z$로부터 $x$ 복원<br>- Reparameterization Trick: $z = \mu + \sigma \cdot \epsilon$, $\epsilon \sim \mathcal{N}(0, I)$ |
| **학습 전략**     | - ELBO maximization<br>- SGD + backpropagation<br>- Closed-form KL divergence 사용 |
| **전이 방식**     | - Posterior-to-prior regularization<br>- Latent space를 $p(z) = \mathcal{N}(0, I)$로 정렬<br>- Interpolation과 sampling이 자연스럽게 가능 |
| **성능/효율성**   | - MNIST 등에서 low NLL<br>- AE보다 generalization 및 latent smoothness 우수<br>- DBN 수준 성능 + 더 나은 학습 안정성 |

---

### 🔗 참고 링크 (References)

- 📄 [arXiv 논문](https://arxiv.org/abs/1312.6114)
- 💻 [PyTorch VAE 예제 코드](https://github.com/pytorch/examples/tree/main/vae)
- 📈 [Papers with Code – VAE](https://paperswithcode.com/paper/auto-encoding-variational-bayes)

---

다음 논문:**[Generative Adversarial Nets (Goodfellow et al., 2014)]**  aka GAN
