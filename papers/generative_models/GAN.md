# 📘 Generative Adversarial Nets

## 1. 개요 (Overview)

- **제목**: Generative Adversarial Nets  
- **저자**: Ian Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, Yoshua Bengio  
- **소속**: Université de Montréal  
- **학회**: NeurIPS (NIPS) 2014  
- **링크**:  
  - [arXiv](https://arxiv.org/abs/1406.2661)  
  - [GitHub (PyTorch 구현 예시)](https://github.com/eriklindernoren/PyTorch-GAN)  
  - [Papers with Code](https://paperswithcode.com/paper/generative-adversarial-nets)

### 📌 논문 선정 이유 및 간단한 도입부

본 논문은 **생성 모델의 새로운 패러다임**을 제시한 시초적 연구로, 생성자(Generator)와 판별자(Discriminator) 간의 **적대적 훈련(Adversarial Training)** 개념을 도입해 이후 수많은 GAN 기반 모델들의 기반이 되었다.  
특히, **지도 학습 없이도 고품질 데이터를 생성**할 수 있다는 점에서 이론적 혁신성과 실용성을 모두 갖춘 논문이다.

해당 논문을 선정한 이유는 다음과 같다:

- 생성 모델(GAN)의 핵심 개념을 처음으로 제안한 기초 논문  
- 이후 DCGAN, StyleGAN, CycleGAN 등 발전형 모델들의 기반  
- 적대적 구조의 수학적 정의와 실험을 통해 새로운 가능성을 제시

> “본 논문은 단순한 MLP 구조를 통해서도 실제처럼 보이는 이미지를 생성할 수 있다는 가능성을 보여주며, 이후 GAN 계열 연구의 폭발적인 발전을 이끈 전환점이 된 연구이다.”


## 2. 문제 정의 (Problem Formulation)

### 🎯 문제 및 기존 한계

기존의 생성 모델(Generative Models)은 주로 다음 두 가지 접근 방식에 의존했다:

1. **Explicit Density Estimation**  
   - 데이터의 분포를 명시적으로 모델링 (예: Gaussian Mixture, PixelRNN 등)
   - 복잡한 데이터 분포에 대해 정확한 likelihood를 정의하거나 계산하기 어려움

2. **Variational 또는 Approximate Inference 기반 모델 (ex. VAE)**  
   - 잠재공간(latent space)을 이용해 샘플 생성  
   - 확률적 모델링은 가능하지만, 생성 결과물이 **blurry**하거나 제한적인 품질을 보임

→ **한계**:  
- 복잡한 고차원 데이터(예: 이미지)의 실제 분포를 정밀하게 추정하기 어려움  
- 고품질 이미지를 자연스럽게 생성하는 데에 성능 한계 존재

---

### 💡 제안 방식: 적대적 생성 (Adversarial Generation)

GAN은 다음과 같은 새로운 접근을 제안한다:

- **Generator (G)**: 랜덤한 노이즈 벡터 $z$를 받아 **그럴듯한 데이터 샘플 $G(z)$**을 생성
- **Discriminator (D)**: 입력이 실제 데이터인지($x \sim p_{\text{data}}$) 또는 생성된 가짜 데이터인지($x \sim G(z)$)를 판별

> G는 D를 속이도록 학습되고, D는 G를 판별하도록 학습됨 → **미니맥스 게임(Minimax Game)** 형식

---

### 📌 핵심 개념 정의

| 용어 | 정의 |
|------|------|
| **Generator (G)** | 잠재 벡터 $z \sim p_z(z)$를 입력 받아, 실제같은 데이터를 생성하는 신경망 |
| **Discriminator (D)** | 입력 데이터가 진짜인지 가짜인지 판별하는 신경망 |
| **Adversarial Loss** |![GAN 수식](https://latex.codecogs.com/svg.latex?\min_G%20\max_D%20V(D,%20G)%20=%20\mathbb{E}_{x%20\sim%20p_{\text{data}}}[\log%20D(x)]%20+%20\mathbb{E}_{z%20\sim%20p_z}[\log(1%20-%20D(G(z)))])|
| **Minimax Training** | G와 D는 서로 반대 목표를 가진 게임에서 교대로 업데이트되며, 균형점에 도달하도록 학습됨 |

---

> ✅ 핵심 아이디어는 "직접 데이터 분포를 모델링하지 않고도, 실제처럼 보이는 데이터를 생성할 수 있는 네트워크를 훈련시키는 방법"을 제안한 것이다.

## 3. 모델 구조 (Architecture)

### 🏗️ 전체 구조

Generative Adversarial Network(GAN)은 두 개의 모델로 구성된 **적대적 구조(adversarial framework)**이다:

- **Generator (G)**: 노이즈 $z \sim p_z(z)$를 입력받아 **가짜 데이터 샘플 $G(z)$** 생성
- **Discriminator (D)**: 입력된 데이터가 진짜인지(실제 데이터 $x \sim p_{data}$) 가짜인지($G(z)$) 판단

두 네트워크는 **동시에 학습**되며, G는 D를 속이고자 하고 D는 G의 출력을 판별하려 한다.  
→ **Minimax 게임**의 형태로 학습이 진행됨.

---

### 🔄 입력/출력 흐름 요약

```text
[Noise Vector z] ─► Generator ─► Fake Data ─┬─► Discriminator ─► Real/Fake
                                             │
                     [Real Data x] ──────────┘
```

## 💠 핵심 모듈 및 구성 요소 (Core Components)

GAN은 크게 두 개의 모델로 구성된다:  
1. **Generator (생성자)**  
2. **Discriminator (판별자)**  

이 두 모델은 서로 적대적 관계로 작동하며, 경쟁을 통해 점차 더 정교한 생성 능력을 학습해간다.

---

### 📌 1. Generator (생성자)

#### 🔹 정의
- **Generator G**는 랜덤 노이즈 벡터 $z \sim p_z(z)$를 입력 받아, 실제처럼 보이는 데이터를 생성하는 모델이다.
- 출력값 $G(z)$는 실제 데이터처럼 보이는 "가짜 이미지" 또는 샘플이다.

#### 🔹 구조
- 초기 논문에서는 **MLP 기반의 Fully Connected Neural Network** 사용
- 이후 연구 (DCGAN, StyleGAN 등)에서는 **CNN 구조**로 진화하여 이미지 생성 품질이 크게 향상됨

#### 🔹 학습 목표
- Discriminator를 **속일 수 있을 정도로 그럴듯한 데이터를 생성**
- 즉, Discriminator가 $G(z)$를 **진짜 데이터**라고 잘못 판단하게 만드는 것이 목적

#### 🔹 수식
- Generator는 다음과 같은 방식으로 학습된다:\
  $\min_G \mathbb{E}_{z \sim p_z}[\log(1 - D(G(z)))]$

- 또는 학습 안정화를 위해 다음 형태를 사용하기도 한다:\
  $\max_G \mathbb{E}_{z \sim p_z}[\log(D(G(z)))]$

#### 🔹 직관적 해설
> G는 “아무것도 없는 노이즈로부터, D가 진짜라고 착각할만한 데이터를 만들어내는 사기꾼” 역할을 한다.

#### 🔹 발전 방향
- **StyleGAN**: 스타일 공간을 도입하여 더 정밀한 제어 가능
- **Conditional GAN (cGAN)**: 클래스 조건을 주어 특정한 이미지 생성 가능
- **BigGAN**: 대규모 클래스와 고해상도 이미지 생성에 적합

---

### 📌 2. Discriminator (판별자)

#### 🔹 정의
- **Discriminator D**는 입력된 데이터가 **진짜(real)**인지 **가짜(fake)**인지 판별하는 이진 분류기
- 출력값 $D(x)$는 0~1 사이의 확률값이며, 1에 가까울수록 진짜라고 판단

#### 🔹 구조
- 마찬가지로 초기 구조는 **MLP 기반**
- 이후 이미지 입력에는 **CNN 구조** 적용이 일반적

#### 🔹 학습 목표
- 실제 데이터 $x$에 대해 $D(x) \approx 1$,  
- 생성된 데이터 $G(z)$에 대해 $D(G(z)) \approx 0$

#### 🔹 수식
- Discriminator는 다음 손실을 최대화:
![GAN 수식](https://latex.codecogs.com/svg.latex?\max_D%20\mathbb{E}_{x%20\sim%20p_{\text{data}}}[\log%20D(x)]%20+%20\mathbb{E}_{z%20\sim%20p_z(z)}[\log(1%20-%20D(G(z)))])




#### 🔹 직관적 해설
> D는 “가짜 데이터를 찾아내려는 감별사”로서, G가 속이려는 시도에 맞서 진짜/가짜를 판별하는 역할

#### 🔹 발전 방향
- **Wasserstein Discriminator (Critic)**: WGAN에서는 sigmoid 없이 real-valued 판별 점수를 사용
- **PatchGAN**: 전체 이미지가 아닌 로컬 패치 단위로 진위 여부 판별 (CycleGAN 등에서 사용)

---

### 📌 3. Adversarial Loss (적대적 손실 함수)

#### 🔹 정의
- GAN의 핵심은 **Generator와 Discriminator 간의 경쟁**이다.
- 이 경쟁은 다음과 같은 **Minimax 게임**으로 정의된다:

 ![GAN Minimax 수식](https://latex.codecogs.com/svg.latex?%5Cmin_G%20%5Cmax_D%20V%28D%2C%20G%29%20%3D%20%5Cmathbb%7BE%7D_%7Bx%20%5Csim%20p_%7B%5Ctext%7Bdata%7D%7D%7D%5Blog%20D%28x%29%5D%20%2B%20%5Cmathbb%7BE%7D_%7Bz%20%5Csim%20p_z%7D%5Blog%281%20-%20D%28G%28z%29%29%29%5D)


#### 🔹 작동 원리
- G는 D의 판단을 속이기 위해, D는 G의 출력을 잘 구분하기 위해 학습됨
- 이 과정은 **交互 최적화 (Alternating Optimization)**로 진행됨: G와 D를 번갈아 업데이트

#### 🔹 직관적 해설
> “G와 D는 서로의 실수를 노리며 발전한다. G가 잘 속이면 D도 더 정밀해지고, D가 잘 구분하면 G도 더 정교해진다.”

---

### 📌 4. Noise Vector $z$ (잠재 벡터)

#### 🔹 정의
- Generator의 입력으로 사용되는 **랜덤한 벡터**
- 일반적으로 정규분포 $\mathcal{N}(0, I)$ 또는 균등분포에서 샘플링

#### 🔹 역할
- 생성 샘플의 **다양성을 유도**
- 같은 G라도 서로 다른 $z$ 값을 입력하면 서로 다른 출력이 생성됨

---

### 📌 5. 분포 $p_{\text{data}}$ vs $p_g$

| 분포 | 정의 | 특징 |
|------|------|------|
| $p_{\text{data}}$ | 실제 데이터 분포 | 우리가 학습하고자 하는 대상 |
| $p_g$             | Generator가 생성하는 분포 | 학습이 진행될수록 $p_{\text{data}}$에 근접해야 함 |

#### 🔹 목적
- 학습이 잘 되면 $p_g \approx p_{\text{data}}$가 되어, G는 실제와 거의 구분되지 않는 데이터를 생성

---

### 🔚 전체 구성 요약

| 구성 요소        | 역할 및 설명 |
|------------------|--------------|
| **Generator (G)** | 노이즈 → 실제처럼 보이는 데이터 생성. D를 속이는 것이 목표 |
| **Discriminator (D)** | 입력이 진짜인지 가짜인지 판별. G를 판별하는 것이 목표 |
| **Loss Function** | Minimax 게임. G와 D가 번갈아 최적화됨 |
| **Noise Vector z** | 다양성을 보장하는 잠재공간의 입력값 |
| **Data Distribution** | G의 분포 $p_g$가 실제 분포 $p_{\text{data}}$와 일치하는 것이 이상적인 상태 |

---
## 🔗 참고 링크 (References)

- 📄 **arXiv 논문**:  
  [https://arxiv.org/abs/1406.2661](https://arxiv.org/abs/1406.2661)

- 💻 **GitHub (PyTorch 기반 구현)**:  
  [https://github.com/eriklindernoren/PyTorch-GAN](https://github.com/eriklindernoren/PyTorch-GAN)  
  다양한 GAN 변종을 포함한 깔끔한 PyTorch 예제 모음 (DCGAN, cGAN, WGAN 등 포함)

- 📈 **Papers with Code**:  
  [https://paperswithcode.com/paper/generative-adversarial-nets](https://paperswithcode.com/paper/generative-adversarial-nets)  
  성능 비교, 코드 구현, 데이터셋 정보 등을 통합 제공

---

### 📚 다음 논문:

> 다음으로 읽을 논문은: Diffusion Model (DDPM) 그 다음에 autoregressive model(GPT계열)

---
