# 📘 [Neural Discrete Representation Learning]

## 1. 개요 (Overview)

* **제목**: Neural Discrete Representation Learning  
* **저자**: Aaron van den Oord, Oriol Vinyals, Koray Kavukcuoglu  
* **소속**: DeepMind  
* **학회**: NeurIPS 2017  
* **링크**: [arXiv](https://arxiv.org/abs/1711.00937) / [GitHub](https://github.com/deepmind/sonnet/blob/v2/examples/vqvae_example.ipynb) / [Papers with Code](https://paperswithcode.com/paper/neural-discrete-representation-learning)

> 이 논문은 Variational Autoencoder의 연속적인 latent 공간을 대체하기 위해 **discrete latent space**를 사용하는 새로운 구조인 **VQ-VAE(Vector Quantized VAE)**를 제안한다.  
> 이미지, 음성, 비디오 등 다양한 데이터에 적용 가능하며, latent 공간에서 **시퀀스 모델**을 학습할 수 있다는 점에서 중요한 기여를 한다.  
> **Discrete latent + strong prior**의 조합이 이후 VQ-GAN, DALL·E 등 다양한 생성 모델의 기반이 되므로 필독 가치가 있다.


## 2. 문제 정의 (Problem Formulation)

**문제 및 기존 한계**:

* 기존 VAE(Variational AutoEncoder) 구조에서는 latent space가 **연속적(continuous)**이기 때문에, latent 벡터에 대한 **해석 가능성**이 낮고, **prior**를 학습하기가 어렵다.
* 연속적인 latent space는 시퀀스 모델(PixelCNN, WaveNet 등)을 적용하기 어렵고, **샘플의 다양성과 품질**을 모두 확보하기 어려웠다.
* RNN이나 CNN 기반의 시퀀스 모델은 입력 전체를 압축한 continuous representation에 의존하며, 이로 인해 정보 손실이 발생하거나 학습이 불안정해질 수 있다.

**제안 방식**:

* encoder가 출력한 continuous latent vector를 **고정된 codebook entry들 중 가장 가까운 discrete token**으로 변환하는 방식(= **Vector Quantization**)을 도입한다.
* decoder는 이 **quantized된 discrete latent token**을 기반으로 원래 입력을 복원하고, 추가로 **autoregressive prior 모델**(ex. PixelCNN)을 통해 token 시퀀스의 구조를 학습할 수 있다.
* 이 구조는 discrete latent 공간에서 **효율적인 학습**과 **샘플 품질 향상**, **시퀀스 모델과의 결합**을 모두 가능하게 한다.

> ※ **핵심 개념 정의**:

* **Vector Quantization (VQ)**:  
  continuous vector를 가장 가까운 codebook entry로 치환하는 과정. 이 과정을 통해 discrete representation을 얻게 됨.

* **Discrete Latent Representation**:  
  연속적인 공간이 아니라, 사전 정의된 벡터 집합(codebook)에서 선택된 벡터들로 구성된 표현. 이를 통해 시퀀스 모델링에 적합한 구조 확보.

* **Commitment Loss**:  
  encoder가 특정 codebook entry에 "책임지고" 매핑되도록 유도하는 손실 항. encoder와 codebook 간의 안정적인 학습을 유도함.

* **Straight-Through Estimator**:  
  backpropagation이 불가능한 quantization 단계에서 gradient를 근사적으로 전달하기 위한 기법.



## 3. 모델 구조 (Architecture)

### 전체 구조

![모델 구조](https://raw.githubusercontent.com/deepmind/sonnet/master/docs/_images/vqvae.png)  
> 출처: DeepMind Sonnet

VQ-VAE(Vector Quantized Variational Autoencoder)는 기본적인 autoencoder 구조에서 연속적인 latent vector 대신, **고정된 벡터 집합(codebook)**에서 선택된 **이산적인(discrete) 토큰**을 사용하는 것이 핵심이다.  

전체 파이프라인은 다음과 같은 세 가지 주요 단계로 구성된다:

1. **Encoder**: 입력 데이터를 continuous latent vector로 변환  
2. **Vector Quantizer**: 이를 가장 가까운 discrete codebook entry로 치환  
3. **Decoder**: discrete latent를 기반으로 원본 데이터를 복원  

이러한 구조는 discrete latent space에 적합한 **prior 모델(예: PixelCNN)**을 결합할 수 있게 하여, 더 나은 generative 모델을 구성한다.

---

### 💠 핵심 모듈 및 수식 설명

#### 📌 Encoder: $x \rightarrow z_e(x)$

* 입력 $x$ (예: 이미지, 음성 등)를 받아, encoder 네트워크는 이를 continuous latent vector $z_e(x)$로 매핑한다.
* $z_e(x) \in \mathbb{R}^D$는 잠재공간(latent space) 상의 벡터로, 이후 양자화(quantization)를 거쳐 discrete 표현으로 변환된다.

#### 📌 Vector Quantizer: $z_e(x) \rightarrow z_q(x)$

* codebook (또는 embedding space) $e = \{ e_k \}_{k=1}^K$, $e_k \in \mathbb{R}^D$는 학습 가능한 K개의 벡터로 구성된다.
* encoder의 출력 $z_e(x)$는 해당 codebook에서 가장 가까운 entry $e_k$로 치환되며, 이를 통해 discrete latent 표현 $z_q(x)$를 얻는다:

$$
z_q(x) = e_k \quad \text{where} \quad k = \arg\min_j \| z_e(x) - e_j \|_2
$$

* 이 과정은 **벡터 양자화 (Vector Quantization)**라고 하며, continuous latent space를 discrete한 표현 공간으로 투영하는 역할을 한다.

#### 📌 Decoder: $z_q(x) \rightarrow \hat{x}$

* decoder는 양자화된 discrete latent vector $z_q(x)$를 입력으로 받아, 원래 입력 $x$를 복원한 $\hat{x}$를 출력한다.
* 재구성 손실은 아래와 같은 로그 가능도(loss likelihood)를 최대화하는 방식으로 정의된다:

$$
\mathcal{L}_{\text{recon}} = -\log p(x \mid z_q(x))
$$

---

### 📌 전체 학습 목표: Loss Function

VQ-VAE는 다음의 세 구성 요소로 이루어진 손실 함수를 사용한다:

$\mathcal{L} = \log p(x \mid z_q(x)) + \| \text{sg}[z_e(x)] - e \|_2^2 + \beta \| z_e(x) - \text{sg}[e] \|_2^2$


#### (1) 재구성 손실 $\log p(x \mid z_q(x))$

* decoder가 양자화된 latent로부터 원본 입력 $x$를 잘 복원할 수 있도록 유도함.

#### (2) codebook 손실 $\| \text{sg}[z_e(x)] - e \|_2^2$

* 선택된 codebook entry $e_k$가 encoder의 출력 $z_e(x)$에 가까워지도록 codebook을 학습시킴.
* 여기서 $\text{sg}[\cdot]$는 **stop-gradient** 연산자로, 해당 변수에 대해 역전파가 이루어지지 않도록 설정한다.
→ codebook entry가 encoder 출력을 따라가도록 유도 (codebook update용)

#### (3) commitment 손실 $\| z_e(x) - \text{sg}[e] \|_2^2$

* encoder가 특정 codebook entry에 "책임을 지도록(commit)" 강제함으로써, 양자화의 안정성과 표현 일관성을 유지한다.
* 하이퍼파라미터 $\beta$는 encoder가 얼마나 강하게 commitment에 책임을 지도록 할지 조절하는 계수이다.

---

### 📌 Straight-Through Estimator (STE)

* 양자화는 비분화(discontinuous) 연산이므로 backpropagation이 불가능하다.
* 이를 해결하기 위해 **straight-through estimator**를 사용한다:
  
$$
\frac{\partial \mathcal{L}}{\partial z_e(x)} \approx \frac{\partial \mathcal{L}}{\partial z_q(x)}
$$

* 즉, forward pass에서는 양자화를 수행하되, backward에서는 양자화된 벡터 $z_q(x)$가 마치 $z_e(x)$인 것처럼 gradient를 전달한다.

---

### 📌 Prior Network (선택적 구성)

* discrete latent token $z_q(x)$ 시퀀스에 대한 **오토리그레시브 prior 모델**을 학습할 수 있다.
* 예: PixelCNN, WaveNet 등을 사용하여 다음과 같은 분포를 학습:

$p(z_1, z_2, \dots, z_n) = \prod_{i=1}^n p(z_i \mid z_{<i})$

* 이를 통해 학습된 latent token을 기반으로 **샘플 생성 또는 예측**이 가능해진다.

---

### 🔍 요약 정리

| 구성 요소        | 설명 |
|------------------|------|
| **Encoder**      | 입력 $x$를 continuous latent vector $z_e(x)$로 인코딩 |
| **Quantizer**    | $z_e(x)$를 가장 가까운 codebook entry $e_k$로 매핑하여 $z_q(x)$ 생성 |
| **Decoder**      | $z_q(x)$로부터 입력 복원 $\hat{x}$ |
| **Loss**         | 재구성 손실 + codebook 학습 + commitment 강제 |
| **STE**          | 양자화 구간에서도 gradient 흐름 유지 |
| **Prior (선택)** | discrete latent에 대해 시퀀스 모델링 가능 |



## ⚖️ 기존 모델과의 비교

| 항목    | 본 논문 (VQ-VAE) | 기존 방법1 (VAE) | 기존 방법2 (Autoencoder + PixelCNN) |
|--------|------------------|------------------|----------------------------|
| 구조    | Encoder → Vector Quantizer → Decoder | Encoder → μ, σ → Sampling → Decoder | Encoder → Continuous latent → PixelCNN |
| 학습 방식 | Non-differentiable quantization with STE | Variational inference (reparameterization trick) | Deterministic encoding + pixel-wise autoregressive decoder |
| 목적    | Discrete latent space로 효율적인 생성 및 시퀀스 모델링 | Continuous latent space 기반 밀도 추정 | 고해상도 이미지 모델링 (prior만 autoregressive)

---

## 📉 실험 및 결과

* **데이터셋**:
  - CIFAR-10 (이미지)
  - VCTK (음성)
  - DeepMind 비디오 데이터셋

* **비교 모델**:
  - Variational Autoencoder (VAE)
  - PixelCNN
  - WaveNet
  - JPEG (압축률 비교 시 baseline)

* **주요 성능 지표 및 결과**:

| 모델            | Accuracy | F1 | BLEU | 기타 (Perplexity, PSNR 등) |
|-----------------|----------|----|------|-----------------------------|
| 본 논문 (VQ-VAE) | N/A      | N/A| N/A  | 음성 재생 품질 우수 (MOS 기준), 이미지 압축률 우수 |
| 기존 VAE        | N/A      | N/A| N/A  | reconstruction blur 심함, 샘플 다양성 떨어짐 |
| JPEG            | N/A      | N/A| N/A  | 압축률 대비 시각적 품질 낮음 |

> 실험 결과 요약 및 해석:

- VQ-VAE는 VAE에 비해 **복원 품질이 훨씬 우수하며**, latent 공간에서 discrete token을 생성하기 때문에 **PixelCNN, WaveNet 등과의 결합이 용이함**.
- 음성 실험에서는 WaveNet decoder와의 결합을 통해 고품질 음성 복원이 가능했고, 이미지 실험에서는 JPEG보다 **높은 압축률에서 더 나은 시각적 품질**을 보임.
- 특히 latent 공간의 이산적 구조 덕분에 **오토리그레시브 prior 학습이 자연스럽게 가능**함.

---

## ✅ 장점 및 한계

### **장점**:

* Continuous → Discrete 변환을 통해 **시퀀스 모델링(autogressive prior)** 이 가능
* VAE 대비 **복원 품질이 높고 블러 현상이 적음**
* Codebook을 통해 **고정된 표현 공간(discrete latent space)**에서 **효율적인 추론 및 생성** 가능
* 다양한 도메인(이미지, 음성, 비디오)에 모두 적용 가능

### **한계 및 개선 가능성**:

* Codebook collapse: 일부 code만 계속 사용되어 다양성이 사라질 수 있음
* Quantization은 본질적으로 정보 손실을 동반함
* Straight-Through Estimator는 gradient 근사이므로 학습 불안정 가능
* Codebook 크기나 $\beta$ 값에 따라 성능이 민감하게 변함 → 튜닝 필요

---

## 🧠 TL;DR – 한눈에 요약

> 연속적인 latent 공간의 한계를 극복하기 위해, encoder 출력을 고정된 codebook entry로 양자화하여 **이산적인 latent 토큰(discrete latent token)**을 학습하는 **VQ-VAE** 구조를 제안함.  
> 이를 통해 high-fidelity data generation + autoregressive prior 학습이 가능해졌고, 이미지/음성/비디오 전반에 강력한 생성 성능을 보임.

| 구성 요소     | 설명 |
|--------------|------|
| 핵심 모듈     | Vector Quantizer, Discrete Latent, Codebook |
| 학습 전략     | Reconstruction + Codebook update + Commitment loss + STE |
| 전이 방식     | Encoder-decoder 구조에 prior (PixelCNN 등) 추가 가능 |
| 성능/효율성  | 고해상도 복원 + 낮은 비트 수의 latent 표현 + 빠른 inference 가능 |

---

## 🔗 참고 링크 (References)

* [📄 arXiv 논문](https://arxiv.org/abs/1711.00937)
* [💻 GitHub (DeepMind Sonnet)](https://github.com/deepmind/sonnet/blob/v2/examples/vqvae_example.ipynb)
* [📈 Papers with Code](https://paperswithcode.com/paper/neural-discrete-representation-learning)

---

## 다음 논문:

> 🔜 **VQ-VAE-2: Generating Diverse High-Fidelity Images with VQ-VAE-2 (2019)**  
> Hierarchical 구조 + powerful prior (PixelSNAIL)로 고해상도 이미지 생성까지 확장한 follow-up 논문
