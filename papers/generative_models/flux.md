# 📘 FLUX.1 Kontext: Flow Matching for In-Context Image Generation and Editing in Latent Space

## 1. 개요 (Overview)

* **제목**: FLUX.1 Kontext: Flow Matching for In-Context Image Generation and Editing in Latent Space  
* **저자**:  
* **소속**: 
* **학회**: arXiv (Preprint, 2025)  
* **링크**: [arXiv](https://arxiv.org/abs/2506.15742) / [GitHub](https://huggingface.co/6chan/flux1-kontext-dev-fp8) / [Papers with Code](https://paperswithcode.com/paper/flux-1-kontext-flow-matching-for-in-context)  

---

> **논문 선정 이유**  
> 최근 이미지 생성 연구는 단순한 “텍스트 → 이미지” 단계를 넘어,  
> **기존 이미지를 보존하면서 원하는 영역만 편집하거나, 캐릭터·스타일 일관성을 유지하며 새로운 이미지를 생성하는 문제**로 확장되고 있다.  
> 그러나 기존 접근법은 (1) 캐릭터 일관성 부족, (2) 반복 편집 시 품질 저하, (3) 생성과 편집 간 통합 부재라는 한계가 있었다.  
>
> 본 논문은 이러한 문제를 해결하기 위해 **Flow Matching 기반 통합 아키텍처**를 제안한다.  
> 이 프레임워크는 **텍스트 기반 생성과 이미지 기반 편집을 하나의 모델**에서 처리할 수 있으며,  
> 새로운 벤치마크인 **KontextBench**를 통해 단일·다중 턴 편집 성능을 체계적으로 평가한다는 점에서 의미가 크다.


## 2. 문제 정의 (Problem Formulation)

**문제 및 기존 한계**:

* 기존 텍스트-이미지 생성 모델은 **편집 기능**이 약하거나, 별도의 추가 네트워크/모듈을 필요로 함 → **생성과 편집의 통합된 프레임워크 부재**  
* 이미지 편집 시, **캐릭터 일관성(character consistency)** 과 **객체 보존(object preservation)** 이 잘 유지되지 않음 → 반복 편집 시 성능 급격히 저하  
* 현재 벤치마크들은 **편집/생성의 실제 시나리오(멀티턴, 로컬 편집 등)** 를 충분히 반영하지 못해, 모델의 실제 활용도를 평가하기 어려움  

---

**제안 방식**:

* **Flow Matching 기반 아키텍처**를 도입하여,  
  - 텍스트 기반 생성(Text-to-Image)  
  - 이미지 기반 편집(Image Editing)  
  을 **하나의 통합 모델**에서 처리 가능  
* 단순한 **시퀀스 연결(sequence concatenation)** 로 컨텍스트(텍스트·이미지)를 결합 → 별도의 복잡한 구조 필요 없음  
* 새로운 **KontextBench 벤치마크**를 제안해,  
  - 단일-턴(single-turn) 편집 품질  
  - 다중-턴(multi-turn) 일관성  
  을 동시에 평가  

---

> **핵심 개념 정의**  
> - **Flow Matching**: 확률적 생성 과정을 deterministic한 연속적 흐름(ODE)으로 근사하여, 빠르고 안정적인 샘플링을 가능하게 하는 기법  
> - **In-Context Image Editing**: 기존 이미지를 입력으로 받아, 원하는 텍스트 지시문(prompt)에 따라 **특정 영역만 수정**하거나 **스타일/캐릭터를 보존하면서 새로운 뷰를 생성**하는 작업  
> - **KontextBench**: 1,026개의 이미지-프롬프트 쌍으로 구성된 벤치마크. 로컬 편집, 글로벌 편집, 캐릭터 참조, 스타일 참조, 텍스트 편집 등 5가지 시나리오를 포함  


# Kontext 모델 구조 (Architecture)

## 전체 구조

![모델 구조](./images/flux_architecture.png)

### 주요 특징

- **Latent-space 파이프라인**: 입력 이미지(컨텍스트/타깃)는 **FLUX 오토인코더(Flux-VAE)** 로 잠재 토큰으로 변환되고, 텍스트 토큰과 함께 **Rectified Flow Transformer (RFT)** 로 전달되어 잠재를 업데이트한 뒤 디코더로 복원한한다.

- **시퀀스 결합(Sequence Concatenation)**: 컨텍스트 이미지 잠재 토큰을 타깃 잠재 토큰 앞/뒤로 **단순 연결**하여 하나의 네트워크가 **I2I 편집과 T2I 생성을 동시에** 처리합니다. 위치 정보는 **3D RoPE**를 사용하되, 컨텍스트 블록에 **상수 오프셋(가상 time-step)** 을 주어 블록 간 경계를 분리한다.

- **백본 개요**: FLUX.1은 **더블-스트림(이미지/텍스트 가중치 분리)** 블록 이후 **싱글-스트림 블록 ×38**로 통합 처리하고, 마지막에 텍스트 토큰을 폐기합니다. **Fused FFN**과 **3D RoPE**로 효율을 높인다.

---

## 핵심 모듈 및 구성 요소

### 🔧 Flux-VAE (Latent Autoencoder)

**역할**: 픽셀 ↔ 잠재 토큰 양방향 변환

16채널 잠재로 학습되어 SDXL/SD3 계열 VAE 대비 **재구성 품질(SSIM/PSNR/PDist)**이 개선됨 (4096 ImageNet 샘플 평가 기준)

### 🔧 Token Sequence Construction + 3D RoPE Offset

#### 시퀀스 구성

```math
z_c = \text{Enc}_{\text{VAE}}(I_c), \quad z_* = \text{target latent (learned via flow)}, \quad \tau = \text{text tokens}
```

```math
S = [z_c \parallel z_* \parallel \tau]
```

#### 3D RoPE 오프셋

각 잠재 토큰의 좌표를 $(x,y,t)$로 두고 RoPE를 적용합니다:

- 타깃 토큰: $t = 0$  
- 컨텍스트 토큰: $t = \Delta$ (상수 오프셋)

```math
\text{RoPE}_{3D}(x,y,t) \text{ with } t = \begin{cases} 
0, & \text{target tokens} \\
\Delta, & \text{context tokens}
\end{cases}
```

이 설계로 해상도/종횡비 차이를 견디고 다중 컨텍스트 이미지로의 확장이 용이합니다. 채널 결합보다 시퀀스 결합이 더 우수함이 확인되었다.

### 🔧 Rectified Flow Transformer (RFT) Backbone

#### 구성

![dit](./images/fused_dit.png)

- **Double-stream**: 이미지/텍스트 토큰을 분리 가중치로 처리하되, 연결된 시퀀스 상에서 어텐션
- **Single-stream ×38**: 이미지/텍스트를 공용 가중치로 통합 처리 → 출력에서 텍스트 토큰 폐기
- **Fused FFN + 3D RoPE**: 연산을 융합해 메모리/속도 효율을 높임

---

## 학습 목적식과 스케줄

### Rectified-Flow Matching Loss

선형 보간 잠재 $\tilde{z}_t$를 따르는 속도(velocity) 회귀 형태의 손실을 사용:

```math
\tilde{z}_t = (1-t)z + t\epsilon, \quad \epsilon \sim N(0,I), \quad t \sim D
```

```math
L_{RF} = E_{z,\epsilon,t}[\|f_\theta(\tilde{z}_t, t, \text{ctx}) - (\epsilon - z)\|_2^2]
```

여기서:
- $f_\theta$: 속도장을 예측하는 네트워크
- $\text{ctx}$: $[z_c \parallel \tau]$로 구성된 컨텍스트

순수 T2I 샘플링 시에는 컨텍스트 이미지 토큰($z_c$)을 생략해 동일 네트워크로 T2I를 수행.

### Logit-Normal Shift Schedule

해상도에 따라 $t$의 분포 모드를 조절하기 위해 Logit-Normal 분포를 사용:

```math
u \sim N(\mu, \sigma^2), \quad t = \sigma(u) = \frac{1}{1+e^{-u}} \Rightarrow t \sim \text{LogitNormal}(\mu, \sigma^2)
```

고해상도 학습에 유리하도록 $t$ 분포(=log-SNR 스케줄)를 좌/우로 이동시키는 효과를 갖는다.

#### RF의 전방 과정과 log-SNR

Rectified-flow의 전방 과정은 선형 보간으로 볼 수 있다:

```math
z_t = (1-t)z + t\epsilon \Leftrightarrow \alpha_t = 1-t, \quad \sigma_t = t
```

따라서 log-SNR은:

```math
\log\text{SNR}(t) = 2\log\left(\frac{\alpha_t}{\sigma_t}\right) = 2\log\left(\frac{1-t}{t}\right)
```

---

## LADD (Latent Adversarial Diffusion Distillation)

다단계 ODE/SDE 샘플링(대략 50–250 스텝, 가이던스 포함)은 느린 추론과 가이던스 아티팩트 문제를 야기할 수 있다. 

LADD는 잠재 공간에서의 적대적 증류로 스텝 수를 줄이면서 품질을 향상시키는 방법:

- **Kontext [pro]**: RF 학습 후 LADD를 적용
- **Kontext [dev]**: 12B DiT로 가이던스 디스틸을 수행

---

## 입력/출력 흐름

### 1. 인코딩
- (옵션) 컨텍스트 $I_c$를 Flux-VAE로 $z_c$로 변환
- 타깃 잠재 $z_*$ 초기화
- 텍스트 토큰 $\tau$ 준비

### 2. 시퀀스 결합 + 위치부여
- $[z_c \parallel z_* \parallel \tau]$ 구성
- 3D RoPE 적용 (컨텍스트: $t = \Delta$, 타깃: $t = 0$)

### 3. RFT 추론
- RF 목적($L_{RF}$)에 맞춰 $z_*$를 업데이트하여 $\hat{z}$ 획득

### 4. 디코딩
- Flux-VAE 디코더로 $\hat{z} \rightarrow \hat{I}$
- T2I 모드에서는 $z_c$를 생략

---
## ⚖️ 기존 모델과의 비교

| 항목    | 본 논문 (FLUX.1 Kontext) | 기존 방법1 (Emu Edit) | 기존 방법2 (ICEdit) |
| ----- | ---- | ------ | ------ |
| 구조    | Rectified Flow Transformer + Flux-VAE 잠재공간. 컨텍스트/타깃 토큰을 **시퀀스 단순 연결** + **3D RoPE 오프셋**. 더블-스트림 후 싱글-스트림 ×38. | 다중 과제(MTL) 기반 diffusion 편집기. 인식 태스크를 생성 태스크로 통일, Task Embedding으로 편집 유형 제어. | In-Context Editing 프레임워크. 구조 변경 없이 프롬프트+컨텍스트로 편집. LoRA-MoE + Early-Filter 추론 스케일링. |
| 학습 방식 | Flow-matching(velocity 회귀) + Logit-Normal shift. T2I 체크포인트 → I2I+T2I 공동 미세조정. 추론 효율을 위해 LADD 적용. | 대규모 합성 Instruction–Response 데이터로 지도 미세학습. 다양한 편집과 비전 태스크 공동 학습. | 소량 데이터·저비용 목표. LoRA-MoE로 효율적 적응, Early-Filter로 초기 노이즈 선택 개선. |
| 목적    | T2I와 I2I를 **단일 모델**에서 통합. 캐릭터·객체 일관성 강화 + 빠른 추론(3–5s/이미지). | 정밀한 지시문 준수, 다양한 편집 유형을 하나의 모델에서 달성. | 최소 학습비용으로 지시문 기반 편집을 고정밀·고효율로 구현. |

---

## 📉 실험 및 결과

* **데이터셋**: **KontextBench** – 1,026 이미지-지시문 쌍(108개 베이스 이미지).  
  포함 태스크: 로컬/글로벌 편집, 캐릭터 참조, 스타일 참조, 텍스트 편집.  
* **비교 모델**: Emu Edit, OmniGen, HiDream-E1, ICEdit, GPT-Image, Gemini Native Image Gen 등.  
* **주요 성능 지표 및 결과**:
  - **일관성 & 속도**: 다중 턴 편집에서도 캐릭터/객체 보존이 뛰어나며 3–5초/이미지 추론.  
  - **VAE 재구성 품질**: Flux-VAE → PDist 0.332, SSIM 0.896, PSNR 31.1 dB (SD3/SDXL 대비 향상).  
  - **휴먼 선호 평가**: KontextBench 기반 휴먼 선호도 평가에서 기존 공개/블랙박스 모델 대비 우위.  

| 모델      | Accuracy | F1 | BLEU | 기타 |
| ------- | -------- | -- | ---- | -- |
| 본 논문    | — | — | — | KontextBench 휴먼 선호 우위, 3–5s/이미지 추론, Flux-VAE: PSNR 31.1 dB |
| 기존 SOTA | — | — | — | Emu Edit/ICEdit/HiDream-E1 등과 비교 참조 |

> **해석:** 단순한 시퀀스 결합 구조임에도 불구하고 다중 턴 시나리오에서 드리프트를 억제하고, 인터랙티브 응답성을 확보. Flux-VAE가 안정적 잠재공간 학습과 고해상도 편집 품질을 뒷받침.

---

## ✅ 장점 및 한계

### **장점**:
* T2I와 I2I를 **단일 모델**로 완전 통합.  
* 캐릭터/객체 일관성이 높고 다중 턴에서도 안정적.  
* 추론 속도가 빠름 (3–5초/장).  
* 단순한 시퀀스 연결 + 3D RoPE 오프셋으로 다양한 상황(해상도, 종횡비, 다중 컨텍스트)에 유연.  
* **오픈 가중치(dev, 12B)** 공개(비상업 연구용).

### **한계 및 개선 가능성**:
* 실험은 **단일 컨텍스트 이미지** 위주 – 다중 컨텍스트 활용은 향후 확장 과제.  
* 평가가 **휴먼 선호도 중심** – 자동화된 표준 지표 부족.  
* **관계형(pair) 데이터** 큐레이션 품질이 성능에 직접적 영향.  
* 공개 가중치는 **비상업 라이선스** – 산업 활용 제약 존재.

---

## 🧠 TL;DR – 한눈에 요약

> **“텍스트+이미지 컨텍스트를 단순 시퀀스 연결과 RoPE 오프셋으로 통합한 rectified-flow 모델. 다중 턴 편집에서도 캐릭터/객체 일관성을 유지하며, 3–5초 수준의 빠른 추론을 달성.”**

| 구성 요소  | 설명 |
| ------ | -- |
| 핵심 모듈  | Rectified Flow Transformer, Flux-VAE, 3D RoPE(+오프셋) |
| 학습 전략  | Flow-matching(velocity), Logit-Normal 스케줄, I2I+T2I 공동 미세조정 |
| 전이 방식  | T2I 체크포인트 → I2I/T2I 통합 확장 |
| 성능/효율성 | 다중 턴 일관성↑, 인터랙티브 속도(3–5s)↑, VAE 재구성 품질↑ |

---

## 🔗 참고 링크 (References)

* [📄 arXiv 논문](https://arxiv.org/abs/2506.15742)  
* [💻 GitHub (FLUX 추론 레포)](https://github.com/black-forest-labs/flux)  
* [💾 Hugging Face: FLUX.1 Kontext [dev]](https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev)  
* [🧪 KontextBench (HF Dataset)](https://huggingface.co/datasets/black-forest-labs/kontext-bench)  
* (비교) [Emu Edit](https://arxiv.org/abs/2311.10089), [ICEdit](https://arxiv.org/abs/2504.20690)

## 다음 논문:
