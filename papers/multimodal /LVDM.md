# 📘 Latent Video Diffusion Models for High-Fidelity Long Video Generation

## 1. 개요 (Overview)

* **제목**: Latent Video Diffusion Models for High-Fidelity Long Video Generation
* **저자**: Yingqing He, Tianyu Yang, Yong Zhang, Ying Shan, Qifeng Chen
* **소속**: The Hong Kong University of Science and Technology, Tencent AI Lab
* **학회**: arXiv (preprint), 2022.11
* **링크**: [arXiv](https://arxiv.org/abs/2211.13221) / [GitHub](https://github.com/voletiv/lvdm) / [Papers with Code](https://paperswithcode.com/paper/latent-video-diffusion-models-for-high)

> 기존 비디오 생성 모델들은 영상 품질이나 시간 길이 측면에서 뚜렷한 한계를 지녔고, 특히 Diffusion 기반 방법은 픽셀 공간에서의 샘플링으로 인해 계산 비용이 매우 컸다. 본 논문은 이를 해결하기 위해 VAE 기반 3D 비디오 오토인코더를 사용하여 비디오를 latent space로 압축하고, 이 latent 공간에서의 hierarchical diffusion 과정을 통해 긴 길이의 고해상도 비디오를 효율적으로 생성할 수 있도록 설계하였다. LVDM은 저차원 표현에서의 생성으로 속도와 품질 모두를 확보하며, text-to-video 확장도 가능한 범용적 구조를 지닌다.


---

## 2. 문제 정의 (Problem Formulation)

### **문제 및 기존 한계**:

* **비디오 생성(Video Generation)**은 자연스럽고 정합성 있는 시공간적 표현이 요구되는 어려운 생성 과제임.
* 기존 방법들은 다음과 같은 한계를 가짐:
  - **GAN 기반 비디오 생성기**는 훈련 불안정성 및 프레임 정합성 부족
  - **Autoregressive 모델**은 계산 비용이 높고 긴 시퀀스에서 누적 오류 발생
  - **Diffusion 기반 비디오 생성기**는 픽셀 공간에서 작동 → 고해상도 비디오 생성 시 **연산량과 메모리 사용량이 급증**
* 특히 긴 영상 생성(예: 수백~수천 프레임)에서 **시각 품질 저하, 시간 축 정합성 붕괴, 프레임 단절성 문제가 빈번히 발생**

---

### **제안 방식**:

LVDM은 이러한 한계를 해결하기 위해 다음과 같은 전략을 제안함:

1. **Latent Video Autoencoder**  
   → 비디오 데이터를 3D spatiotemporal latent로 압축하여 **저차원에서 diffusion 수행**

2. **Hierarchical Latent Diffusion**  
   → 시간 해상도를 점진적으로 확장하며 프레임 생성 → **수천 프레임도 안정적으로 생성 가능**

3. **Conditional Latent Perturbation**  
   → 조건 기반 노이즈를 추가하여 긴 비디오 생성 중 발생할 수 있는 **drift 현상 완화**

4. **Unconditional Guidance**  
   → 조건 없는 샘플링 경로를 함께 고려하여 **샘플링 안정성 강화 및 다양성 확보**

---

> ### ※ 핵심 개념 정의  
> - **Latent Diffusion**: 비디오를 VAE를 통해 latent space로 압축한 후, 그 공간에서 확산(reverse denoising)을 수행하는 구조  
> - **3D Video Autoencoder**: 시간 + 공간 축 정보를 통합적으로 인코딩하는 오토인코더 구조  
> - **Hierarchical Sampling**: coarse temporal step → fine temporal step 순으로 프레임을 점진적으로 생성하는 방식  
> - **Latent Perturbation**: condition 정보를 더 강하게 반영하기 위해 latent 공간에 노이즈를 의도적으로 삽입하는 전략  
> - **Unconditional Guidance**: 조건 샘플링과 무조건 샘플링 결과를 혼합해 더 유연하고 안정적인 생성 유도

---

## 3. 모델 구조 (Architecture)

![모델구조조]

### 🔷 전체 시스템 구성 흐름

LVDM은 비디오 생성 과정을 다음의 세 단계로 구성된 파이프라인으로 수행한다다:
```
[1] 입력 비디오 또는 조건 (ex. 텍스트 prompt)
                ↓
[2] Video Autoencoder (3D VAE)
                ↓
[3] Latent Representation (T × H' × W' × C)
                ↓
[4] Hierarchical Latent Diffusion
                ↓
[5] Reconstructed Latent
                ↓
[6] VAE Decoder → 최종 비디오 출력
```

---

### 💠 핵심 모듈 또는 구성 요소 

---

#### 🔷 1. 3D Video Autoencoder (VAE 기반)

**📌 작동 방식**  
- 입력 비디오 $x \in \mathbb{R}^{T \times H \times W \times 3}$는 시간 축 $T$, 공간 해상도 $H \times W$를 갖는 RGB 시퀀스입니다.  
- 이 비디오는 **3D convolution 기반 인코더**를 거쳐 **시공간적으로 압축된 latent representation** $z$로 매핑됩니다:

```math
z \in \mathbb{R}^{T \times H' \times W' \times C}
```

- 이 latent는 이후 diffusion 과정에서 사용되며, 최종적으로 VAE 디코더를 통해 원래의 해상도로 복원됩니다.

**📌 학습 목표**  
- reconstruction 손실 + 정규화(KL divergence)를 최소화하여 latent 공간이 자연스러운 prior $p(z)$를 따르도록 학습합니다.

**📌 복원 손실 수식**

```math
\mathcal{L}_{\text{VAE}} = \mathbb{E}_{q(z|x)} \left[ \| x - \text{Dec}(z) \|^2 \right] + D_{KL}(q(z|x) \| p(z))
```
**📌 효과**  
- 픽셀 공간보다 훨씬 저차원인 latent 공간에서 diffusion을 수행 → 메모리, 연산량 절감  
- 3D 구조를 사용함으로써 **시간 축 연속성(motion consistency)**을 유지한 압축 가능

---

#### 🔷 2. Hierarchical Latent Diffusion

**📌 작동 방식**  
- 전체 비디오를 한 번에 생성하지 않고, **시간 해상도를 점차 증가시키는 hierarchical sampling**을 사용합니다:

  1. **Coarse step**: 낮은 frame rate (예: 8fps) 수준의 latent 시퀀스를 생성  
  2. **Interpolation**: latent space에서 시간 축 보간 수행  
  3. **Refinement**: 보간된 latent를 고해상도 (예: 30fps 등) latent로 점진적 refinement

**📌 Denoising Step 수식**

```math
z_{t-1} = z_t - \epsilon_{\theta}(z_t, t, c)
```

- $z_t$: 현재 timestep에서의 noisy latent

- $\epsilon_{\theta}$: 조건부 노이즈 예측 네트워크 (U-Net 계열)

- $c$: 조건 정보 (텍스트, 클래스 등)

**📌 효과**

- 짧은 시퀀스에 비해 긴 비디오 생성에서도 시간 축 정합성 유지

- 긴 시퀀스를 효율적으로 생성할 수 있으며, 점진적 refinement로 고품질 유지
---

#### 🔷 3. Conditional Latent Perturbation

**📌 문제 인식**  
- 조건(condition) 기반 비디오 생성에서, 샘플링이 길어질수록 **조건 정보가 흐려지고** 내용이 **drift**되는 문제가 발생합니다.

**📌 해결책**  
- 조건 정보 $c$ (예: 텍스트 프롬프트)로부터 perturbation 벡터를 생성해 **latent 공간에 직접 주입**합니다.  
- 이를 통해 diffusion 도중에도 조건의 영향을 지속적으로 반영할 수 있습니다.

**📌 수식**

```math
z_t = z_t + \alpha \cdot \text{Perturb}(c)
```

- $z_t$: 현재 timestep의 latent

- $\alpha$: 조건 노이즈의 반영 강도를 조절하는 하이퍼파라미터

- $\text{Perturb}(c)$: 조건 벡터에 랜덤 노이즈를 섞은 perturbation 임베딩

**📌 효과**

- 조건 일관성 유지 (ex. 프롬프트 주제에 벗어나지 않음)

- 긴 영상 생성에서도 drift를 방지하고 내용 안정성 확보
---

**🔷 4. Unconditional Guidance**

**📌 문제 인식**

- 조건이 모호하거나 불완전한 경우, 모델이 mode collapse 되거나 품질이 낮은 샘플을 생성할 수 있음

**📌 해결 전략**

- Classifier-Free Guidance (CFG) 방식을 차용해

- 조건이 있는 경로와 조건이 없는 경로의 예측값을 혼합하여 샘플링을 유도합니다.

**📌 수식**
```math
\epsilon_{\text{final}} = (1 + w) \cdot \epsilon_{\theta}(z_t, c) - w \cdot \epsilon_{\theta}(z_t)
```
- $w$: guidance scale — 조건 반영 강도를 조절하는 계수 (일반적으로 3~7 사이)

- $\epsilon_{\theta}(z_t, c)$: 조건이 포함된 노이즈 예측

- $\epsilon_{\theta}(z_t)$: 조건이 없는 노이즈 예측

**📌 해석**

- $w = 0$일 경우 → 완전히 unconditional (조건 없는 생성)

- $w$가 커질수록 → 조건 $c$에 더 정확히 맞는 샘플 생성

- 단, 너무 높을 경우 다양성 저하 가능
→ 조건 정확도 vs 생성 다양성 간 trade-off를 조절할 수 있음

**📌 효과**

- 텍스트 조건이 모호하거나 불완전한 경우에도 안정적인 결과 생성 가능

- 다양성 있는 샘플 생성 및 mode collapse 방지
---
## ⚖️ 기존 모델과의 비교

| 항목       | 본 논문 (LVDM)                         | 기존 방법1 (VideoGAN)               | 기존 방법2 (Video Diffusion in Pixel Space)   |
|------------|----------------------------------------|--------------------------------------|------------------------------------------------|
| 구조       | 3D VAE + Hierarchical Latent Diffusion | CNN 기반 GAN                         | U-Net 기반 확산 (픽셀 공간)                    |
| 학습 방식  | VAE 사전학습 후, latent diffusion joint 학습 | Adversarial Loss 기반 End-to-End     | 픽셀 공간의 DDPM 학습, 메모리 사용량 큼       |
| 목적       | 고해상도 + 장시간 비디오 생성         | 짧고 반복 가능한 low-res 비디오 생성 | 고해상도 생성 가능하나 연산량 많고 긴 비디오 어려움 |
| 장점       | 고속, 정합성, 길이 확장성 모두 확보    | 속도 빠름, 구조 간단                 | 고품질 샘플 생성 가능, CLIP 등과 결합 가능     |
| 한계       | 텍스트 기반 확장은 아직 실험적         | 품질/정합성 떨어짐                   | 계산량/시간 부담 크고 시공간 consistency 낮음 |

---

## 📉 실험 및 결과

**📌 사용 데이터셋**  
- **UCF-101**: 동작 중심 짧은 비디오 클립  
- **SkyTimelapse**: 정적인 배경 + 느린 움직임  
- **MEAD**, **FaceForensics**: 얼굴 기반 long-form 비디오

**📌 비교 모델**  
- VideoGPT  
- Video Diffusion Models (VDM)  
- TATS  
- StyleGAN-V

**📌 주요 성능 지표**

| 모델           | FVD↓     | FID↓     | CLIPSIM↑ | 비고                            |
|----------------|----------|----------|----------|---------------------------------|
| **LVDM (ours)** | **256.1** | **45.3** | **0.301** | 전체적으로 가장 우수한 정합성    |
| VDM            | 423.5    | 58.1     | 0.211    | 픽셀 공간 diffusion              |
| TATS           | 287.3    | 51.4     | 0.229    | AR 기반, 긴 비디오 약함         |
| StyleGAN-V     | 601.2    | 73.5     | 0.188    | 얼굴 데이터는 가능, 일반성 낮음 |

> **해석**: LVDM은 FVD, FID, CLIPSIM 모든 지표에서 가장 우수한 성능을 기록하며  
> 고해상도 장시간 비디오 생성에서 **정합성과 품질을 모두 확보한 첫 latent diffusion 기반 방법**임을 입증함.

---

## ✅ 장점 및 한계

### ✅ 장점

* **Latent space diffusion**을 통한 **연산 효율성** 확보 (메모리/속도 모두 유리)
* **3D Video Autoencoder**로 시간-공간 정보 동시 압축 → **motion consistency 향상**
* **Hierarchical time sampling**을 통해 **수천 프레임 비디오도 안정적 생성**
* **Perturbation + CFG** 조합으로 **조건 정합성과 다양성** 동시 확보
* 여러 real-world 데이터셋에서 SOTA 성능 달성

---

### ⚠️ 한계 및 개선 가능성

* 텍스트 기반 conditional generation은 아직 초기 수준 (Prompt-to-video fully end-to-end는 아님)
* Autoencoder 성능에 따라 latent 품질이 제한될 수 있음
* 시각 정보 외의 오디오, 자연어, multimodal integration은 미지원
* Latent interpolation이 고속 변화 장면에서 artifacts 발생 가능

---

## 🧠 TL;DR – 한눈에 요약

> **LVDM**은 3D VAE로 비디오를 압축한 latent 공간에서 hierarchical diffusion을 수행하여,  
> 고해상도이면서도 시간적으로 긴 비디오를 빠르고 안정적으로 생성할 수 있는 **최초의 시공간 latent 기반 비디오 생성 모델**이다.

| 구성 요소         | 설명                                                       |
|------------------|------------------------------------------------------------|
| 핵심 모듈         | 3D Video Autoencoder + Hierarchical Latent Diffusion      |
| 학습 전략         | 사전학습된 VAE + joint diffusion 학습                     |
| 전이 방식         | coarse-to-fine 시간 해상도, condition perturbation 적용   |
| 성능/효율성       | FVD/FID/CLIPSIM에서 SOTA 달성, 고해상도+장시간 영상 가능  |

---

## 🔗 참고 링크 (References)

* [📄 arXiv 논문](https://arxiv.org/abs/2211.13221)
* [💻 GitHub](https://github.com/voletiv/lvdm)
* [📈 Papers with Code](https://paperswithcode.com/paper/latent-video-diffusion-models-for-high)




