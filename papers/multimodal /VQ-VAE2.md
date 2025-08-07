# 📘 VQ-VAE2 (Hierarchical Neural Discrete Representation Learning)

## 1. 개요 (Overview)

* **제목**: Hierarchical Neural Discrete Representation Learning (VQ-VAE2)  
* **저자**: Aaron van den Oord, Yazhe Li, Igor Babuschkin, Karen Simonyan, Oriol Vinyals, Andrew Brock, Koray Kavukcuoglu  
* **소속**: DeepMind, Google Brain  
* **학회**: NeurIPS 2019  
* **링크**: [arXiv](https://arxiv.org/abs/1906.00446) / [GitHub (DeepMind)](https://github.com/deepmind/vq-vae) / [Papers with Code](https://paperswithcode.com/paper/hierarchical-neural-discrete-representation)

> **논문 선정 이유**:  
> 고해상도 이미지 생성에서 discrete latent representation이 가지는 장점을 계층적 구조로 확장한 VQ-VAE2의 핵심 아이디어와 성능 향상 기법을 이해해, 향후 generative 모델 연구 및 실험에 적용할 수 있는 실용적 인사이트를 얻고자  
>
> **간단한 도입부**:  
> 전통적인 VAE 기반 이미지 생성 모델은 연속 잠재 공간의 한계로 인해 세밀한 디테일 표현과 다양성 확보에 어려움을 겪는다. VQ-VAE2는 벡터 양자화(Vector Quantization) 기법을 이용해 discrete latent를 도입하고, 이를 상위(저해상도)와 하위(고해상도) 단계로 계층화함으로써 글로벌 구조와 로컬 디테일을 동시에 효과적으로 학습, BigGAN 수준의 고품질 이미지를 생성해내는 성과를 보였다.

---
## 2. 문제 정의 (Problem Formulation)

**문제 및 기존 한계**:

* **연속 잠재 공간의 표현력 한계**  
  기존 VAE나 GAN 기반 모델은 연속적인 잠재(z) 공간에 의존하기 때문에, 복잡한 글로벌 구조와 미세한 로컬 디테일을 동시에 고해상도로 캡처하기 어려움.  
* **단일 수준의 잠재 표현으로 인한 품질 저하**  
  하나의 latent 코드북만 사용하는 VQ-VAE는 전체 이미지 구조나 텍스처 디테일 중 하나를 희생하게 되고, 고해상도 합성 시 샘플 품질과 다양성이 제한됨.

**제안 방식**:

* **계층적 디스크리트 잠재 표현**  
  – **상위 레벨(저해상도)**: 이미지의 큰 틀과 전역 구조를 압축하는 저해상도 코드북  
  – **하위 레벨(고해상도)**: 상위 레벨 출력을 보강하여 세밀한 텍스처와 디테일을 복원하는 고해상도 코드북  
* **두 단계 VQ-VAE 학습**  
  – 상위 인코더/디코더로 global latent 학습 → 디코딩 후 업샘플링  
  – 업샘플된 특징에 하위 인코더/디코더를 적용해 local latent 학습  
  – 각각의 단계에서 **코드북 벡터 양자화** 및 **commitment loss**를 사용해 안정적 훈련  

> ※ **핵심 개념 정의**  
> * **Vector Quantization (VQ)**: 연속 벡터를 미리 정의된 이산(codebook) 벡터 중 최인접 이산값으로 매핑하는 기법  
> * **Codebook**: 모델이 학습하는 이산 잠재 벡터의 집합, 각 벡터는 이미지의 특정 패턴 또는 구조를 대표  
> * **Commitment Loss**: 인코더 출력이 코드북 벡터에 “과도하게 쏠리지” 않도록 유도해 표현의 다양성과 안정성을 높이는 페널티  
> * **계층적 구조 (Hierarchical Representation)**: 다중 해상도/단계에서 잠재를 분리 학습해, 전역 구조와 로컬 디테일을 각각 최적화  
---  
## 3. 모델 구조 (Architecture)

### 전체 구조

![모델 구조](path/to/vq-vae2_architecture.png)

* 입력 이미지 $x \in \mathbb{R}^{H\times W\times 3}$를 상위 인코더 $E^t$에 투입해 저해상도 특징 맵 $f^t = E^t(x)$을 얻습니다.  
* 벡터 양자화 모듈 $VQ^t$가 $f^t$를 코드북 $\{e^t_k\}_{k=1}^{K_t}$에서 최근접 임베딩으로 양자화하여 이산 표현 $z^t$를 생성합니다.  
* 상위 디코더 $D^t$가 $z^t$를 업샘플링하여 전역 구조 정보를 담은 보간된 특징 맵 $\tilde f^t$를 복원합니다.  
* 하위 인코더 $E^b$는 원본 이미지 $x$와 $\tilde f^t$를 결합해 고해상도 특징 맵 $f^b = E^b(x,\tilde f^t)$을 산출합니다.  
* 하위 벡터 양자화 $VQ^b$가 $f^b$를 코드북 $\{e^b_k\}_{k=1}^{K_b}$에서 양자화해 이산 표현 $z^b$를 생성합니다.  
* 최종 디코더 $D^b$는 $z^t$와 $z^b$를 결합해 고해상도 이미지 $\hat{x}$를 재구성합니다.  
* Autoregressive Prior 네트워크(예: Gated PixelCNN)는 $p(z^t)$ 및 $p(z^b \mid z^t)$를 학습해, 실제 샘플링 단계에서 고품질 이미지를 생성합니다.

---

### 💠 핵심 모듈 또는 구성 요소

#### 📌 Top-Level Vector Quantizer (VQ¹)

* **입력**: 저해상도 특징 맵 $f^t \in \mathbb{R}^{h\times w\times D}$  
* **양자화 방식**:  

```math
k^* = \arg\min_k \|f^t_i - e^t_k\|_2
```

```math
z^t_i = e^t_{k^*}
```

* **Loss 구성**:  

```math
\|\text{sg}[f^t] - e^t_{k^*}\|_2^2
```

```math
\beta \|f^t - \text{sg}[e^t_{k^*}]\|_2^2
```
* **VQ-VAE 대비 차별점**:  
  - 단일 레벨 대신 대규모 코드북 $(K_t \gg)$과 깊은 임베딩 차원 $(D \gg)$을 사용해 전역 구조 표현력 극대화  
  - 계층적 분리로 local/global 상충 문제 해소  

#### 📌 Bottom-Level Vector Quantizer (VQ²)

* **입력**: 업샘플된 전역 특징 $\tilde{f}^t$와 원본 임베딩을 결합한 $f^b \in \mathbb{R}^{H\times W\times D}$  
* **양자화 방식**: VQ¹와 동일하되, 코드북 $\{e^b_k\}$을 통해 local 디테일 학습  
* **차별점**: 전역(latent)과 국부(latent)를 분리해 처리함으로써 고해상도 텍스처와 디테일을 동시에 보존  

#### 📌 Autoregressive Prior Network (PixelCNN)

* **역할**:  
  - 전역 잠재 $z^t$에 대한 사전분포 $p(z^t)$ 학습  
  - 조건부 분포 $p(z^b \mid z^t)$ 학습  
* **구현**: Gated PixelCNN 구조를 채택해 각 위치의 이산 코드 인덱스를 이전 코드 순차적으로 예측  
* **VQ-VAE 대비 차별점**:  
  - 단일 prior 대신 계층적 prior로 복잡도 분산  
  - 전역 코드 생성 후 하위 코드를 조건부로 생성해 샘플 품질 및 다양성 대폭 개선  

---

**기존 VQ-VAE와의 핵심 차별점 요약**  
1. **계층화된 이산 표현**: 하나의 latent 대신 두 단계(latent¹/global, latent²/local)로 분리  
2. **확장된 코드북 용량**: 전역 및 국부 코드북 크기와 임베딩 차원 증가  
3. **계층적 Autoregressive Prior**: $p(z^t)\to p(z^b\mid z^t)$으로 순차적 샘플링  
4. **고해상도 이미지 적합성**: 글로벌 구조와 미세 디테일을 동시에 포착해 BigGAN 수준 성능 달성
---

## ⚖️ 기존 모델과의 비교

| 항목       | 본 논문 (VQ-VAE2)                       | 기존 방법1 (VQ-VAE)                                 | 기존 방법2 (BigGAN-deep)                         |
| ---------- | --------------------------------------- | --------------------------------------------------- | ----------------------------------------------- |
| **구조**   | 다단계 계층적 VQ (글로벌·로컬 codebook)  | 단일 수준 VQ codebook                              | Generator-Discriminator 구조 (Adversarial)      |
| **학습 방식** | 비지도 계층적 VQ 학습 + Autoregressive Prior  
              | 비지도 VQ 학습  
              | 적대적 학습 (GAN)                              |
| **목적**   | 고해상도 이미지의 전역 구조와 세부 디테일 동시 학습  | 고해상도에서 샘플 품질·다양성 한계  | 우수한 FID/IS 성능, 그러나 모드 붕괴 및 다양성 저하 위험  |

---

## 📉 실험 및 결과

* **데이터셋**:  
  * ImageNet (256×256)  
  * FFHQ (1024×1024)

* **비교 모델**:  
  * VQ-VAE (Neural Discrete Representation Learning) 
  * BigGAN-deep (State-of-the-Art GAN) 

* **주요 성능 지표 및 결과**:  
  - **샘플 품질**:  
    - VQ-VAE2는 ImageNet 256×256에서 BigGAN-deep에 근접하는 FID 및 Inception Score를 달성하며,  
      모드 붕괴 없이 높은 다양성을 유지 
  - **샘플링 속도**:  
    - 픽셀 공간에서 직접 샘플링하는 것보다 **약 30배 빠른** Latent 공간 샘플링 속도 

> **실험 결과 요약 및 해석**  
> VQ-VAE2는 계층적 이산 표현과 강력한 Autoregressive Prior 조합을 통해,  
> 기존 VQ-VAE 대비 FID를 대폭 감소시키고(≈33→≈11),  
> BigGAN-deep 수준의 샘플 품질과 다양성을 동시에 확보했다.

---

## ✅ 장점 및 한계

### **장점**  
* **고해상도 샘플 품질**: 글로벌 구조와 로컬 디테일을 동시에 포착, BigGAN 수준의 FID/IS 달성 
* **다양성 유지**: Mode Collapse 없이 데이터 분포의 다양한 모드를 포괄 
* **빠른 샘플링**: Latent 공간에 국한된 Autoregressive 샘플링으로 픽셀 공간보다 획기적 속도 개선 
* **엔코딩/디코딩 효율**: 단순 Feed-forward Encoder/Decoder 구조로 실시간 응용에 적합  

### **한계 및 개선 가능성**  
* **Prior 학습 비용**: 대규모 Autoregressive Prior 학습 시 메모리·연산 부담 증가  
* **순차적 샘플링**: Latent 공간에서는 빨라졌으나 여전히 순차적 특성으로 실시간 제약  
* **아키텍처 복잡도**: 두 단계 VQ 및 Prior 네트워크 설계·튜닝 난이도 높음  
* **조건부 샘플링 한계**: Classifier-based rejection sampling 필요성이 일부 품질 향상 과정에 도입됨  

---

## 🧠 TL;DR – 한눈에 요약

> VQ-VAE2는 **계층적 이산 잠재 표현**과 **Autoregressive Prior**를 결합해,  
> 글로벌 구조와 로컬 디테일을 동시에 학습하며  
> **고해상도 이미지**에서 BigGAN 수준의 **샘플 품질·다양성**을  
> **30× 빠른** Latent 공간 샘플링으로 달성한 모델입니다.

| 구성 요소      | 설명                                                           |
| -------------- | -------------------------------------------------------------- |
| **핵심 모듈**    | Top-Level VQ + Bottom-Level VQ + Hierarchical PixelCNN Prior  |
| **학습 전략**    | Two-Stage VQ 학습 → Latent Prior 학습 (PixelCNN with Self-Attention) |
| **전이 방식**    | 비지도 압축 → Autoregressive Prior 연계                         |
| **성능/효율성** | BigGAN-like FID/IS + 모드 붕괴 없음 + 30× 빠른 샘플링 속도       |

---

## 🔗 참고 링크 (References)

* [📄 arXiv 논문](https://arxiv.org/abs/1906.00446)  
* [📄 NeurIPS PDF – Table 및 구현 상세](https://papers.neurips.cc/paper/9625-generating-diverse-high-fidelity-images-with-vq-vae-2.pdf)  
* [💻 GitHub (DeepMind VQ-VAE)](https://github.com/deepmind/vq-vae)  
* [📈 Papers with Code](https://paperswithcode.com/paper/hierarchical-neural-discrete-representation)  



