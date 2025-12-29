# 📘 [Normalized Attention Guidance: Universal Negative Guidance for Diffusion Models]

## 1. 개요 (Overview)

* **제목**: Normalized Attention Guidance: Universal Negative Guidance for Diffusion Models ([arXiv][1])
* **저자**: Dar-Yen Chen, Hmrishav Bandyopadhyay, Kai Zou, Yi-Zhe Song ([arXiv][1])
* **소속**: SketchX Lab ([Chendaryen][2])
* **학회**: NeurIPS 2025 (Poster) / arXiv preprint ([NeurIPS][3])
* **링크**:

  * [arXiv](https://arxiv.org/abs/2505.21179) ([arXiv][1])
  * [Project Page](https://chendaryen.github.io/NAG.github.io/) ([Chendaryen][2])
  * [GitHub](https://github.com/ChenDarYen/Normalized-Attention-Guidance) ([GitHub][4])
  * [ComfyUI](https://github.com/ChenDarYen/ComfyUI-NAG) ([GitHub][5])

> **논문 선정 이유(너의 연구/관심사 관점에서)**
> 너가 지금 하고 있는 “(1) negative prompt를 step-wise로 다루는 시스템(ANSWER/DNP류)”, “(2) verifier/VLM로 샘플링을 안정화시키는 System-2 diffusion”과 바로 맞닿아 있어.
> 이 논문은 **학습 없이(inference-time), 모델 아키텍처/모달리티에 거의 상관없이** negative prompt 효과를 되살리는 “플러그인”을 제안하고, 특히 **few-step(예: 2~8 step)에서 CFG가 깨지는 현상**을 정면으로 해결해. ([ar5iv][6])

---

## 2. 문제 정의 (Problem Formulation)

**문제 및 기존 한계**:

* **Negative guidance(=negative prompt로 ‘억제’)**는 T2I/T2V에서 핵심 기능인데,
  **few-step diffusion(아주 적은 denoising step)**에서는 기존 대표 기법 **CFG**가 잘 안 먹거나 오히려 붕괴(artifact/깨짐)한다. ([ar5iv][6])
* 왜 CFG가 깨지나?
  CFG는 보통 “positive branch”와 “negative/uncond branch”의 예측이 **동일한 구조/매니폴드 위에서 선형적으로 외삽 가능**하다고 암묵적으로 가정한다.
  그런데 few-step에서는 초기 스텝에서 브랜치가 **급격히 분기(diverge)**해서, output(노이즈 예측) 공간에서의 외삽이 **out-of-manifold**로 튀면서 이미지가 망가진다. ([ar5iv][6])
* 기존 “attention 계열” 음성 가이던스(예: NASA)도 시도되지만, DiT 계열에서 불안정하거나 scale을 올리면 쉽게 깨진다고 보고한다. ([ar5iv][6])

**제안 방식**:

* 핵심 아이디어: **노이즈 예측(output) 공간이 아니라, “attention feature space”에서 negative guidance를 수행**한다.
* 단, 단순 외삽만 하면 여전히 폭주할 수 있으니,

  1. **L1 기반 정규화로 크기 폭주를 제어**하고
  2. **α-blending(positive feature와 섞기)로 매니폴드로 다시 당겨** 안정화한다.
* 이 전체를 **학습 없이**, **UNet/DiT/MM-DiT**, **이미지/비디오**, **few-step/multi-step**에 공통으로 적용 가능한 **universal plug-in**으로 제시한다. ([ar5iv][6])

> ※ **핵심 개념 정의**

* **Few-step diffusion**: 2~8 step 같은 매우 짧은 샘플링으로 빠르게 생성(대신 가이던스/안정성이 취약). ([ar5iv][6])
* **CFG(Classifier-Free Guidance)**: conditional과 (unconditional/negative) 예측을 섞어 방향을 강화하는 표준 가이던스. few-step에서 브랜치 발산으로 붕괴 가능. ([ar5iv][6])
* **Attention feature space**: cross-attention(또는 MM-DiT 블록)에서 텍스트 토큰과 이미지 특징이 상호작용해 생성되는 중간 특징(저자들은 이를 Z로 표기). 여기서 negative prompt가 “어떤 속성을 밀어낼지”를 더 직접적으로 반영한다고 본다. ([ar5iv][6])
* **Out-of-manifold drift**: 모델이 학습 분포 밖으로 튀어 artifact/텍스처 붕괴가 나는 현상(가이던스 강할수록 위험). ([ar5iv][6])

---

## 3. 모델 구조 (Architecture)

### 전체 구조

![모델 구조](경로)

* **입력**:

  * positive prompt (p) (원하는 내용)
  * negative prompt (n) (억제하고 싶은 속성: 예 “Low resolution, blurry”) ([ar5iv][6])
* **기본 생성 흐름(denoising step t마다)**

  1. 모델 내부 attention에서 **positive feature (Z^{+})**, **negative feature (Z^{-})** 를 얻음
  2. attention space에서 (Z^{+})를 (Z^{-})로부터 “멀어지는 방향”으로 외삽(extrapolation)
  3. **L1 정규화 + clip**으로 폭주를 막음
  4. **α-blending**으로 (Z^{+}) 쪽으로 다시 당겨 안정화
  5. 바뀐 attention feature로 나머지 블록을 진행 → 최종 샘플 업데이트 ([ar5iv][6])

---

### 💠 핵심 모듈 또는 구성 요소

#### 📌 (1) Attention-space Extrapolation (방향성 만들기)

저자 표기에서 cross-attention 출력 특징을 (Z)로 둔다고 하면:

```math
\tilde{Z} = Z^{+} + s\,(Z^{+} - Z^{-})
```

* (s): NAG scale(= guidance strength 같은 역할)
* 의미: (Z^{+})에서 (Z^{-}) 방향의 반대로 “밀어내기(negative를 회피)”를 attention 특징 차원에서 수행. ([ar5iv][6])

#### 📌 (2) L1-based Normalization + Guidance Boundary (폭주 제어)

외삽만 하면 (\tilde{Z}) 크기가 커지며 out-of-manifold 위험.
그래서 **L1 norm 비율**을 계산해 과도한 확대를 **상한 (\tau)** 로 제한:

```math
r = \frac{\|\tilde{Z}\|_1}{\|Z^{+}\|_1}
\qquad
\hat{Z} = \frac{\min(r,\tau)}{r}\,\tilde{Z}
```

* 직관: “방향”은 유지하되, **feature magnitude가 (Z^{+}) 대비 (\tau)배 이상 커지지 않게** 강제로 눌러서 안정화. ([ar5iv][6])

#### 📌 (3) Feature Refinement via α-blending (매니폴드로 당기기)

정규화해도 분포가 흔들릴 수 있으니 마지막으로 positive feature와 섞음:

```math
Z_{\text{NAG}} = \alpha\,\hat{Z} + (1-\alpha)\,Z^{+}
```

* (\alpha)가 작을수록 (Z^{+})에 더 붙어 안정적(하지만 가이던스 약해질 수 있음)
* 저자 설명: (\hat{Z})를 “Guidance Boundary” 안으로 넣고, 다시 (Z^{+})쪽 “Refinement manifold”로 당겨 **분포 일관성**을 유지. ([ar5iv][6])

#### 📌 (4) 어디에 주입하나? (UNet vs DiT/MM-DiT)

* UNet 계열: **cross-attention layer 출력**을 (Z_{\text{NAG}})로 대체하는 형태로 주입
* DiT/MM-DiT 계열: **(멀티모달) 트랜스포머 블록 내부 attention feature**에 동일하게 적용
  → 핵심은 “노이즈 예측값(ε/velocity) 자체”를 섞는 게 아니라 **attention feature를 조작**하는 것. ([ar5iv][6])

#### 📌 (5) CFG/PAG와의 결합

* multi-step(예: 25-step)에서는 기존 CFG/PAG를 쓰면서도 **추가로 NAG를 attention에 결합** 가능하다고 실험으로 보임. ([ar5iv][6])

#### 📌 (6) Early stopping(추가 팁)

Appendix에서 “NAG는 초반 step에서 영향이 크고 후반엔 감소”하는 경향을 관찰하고,
전체 step 중 처음 (\theta) 비율까지만 NAG를 켜는 **early stopping**을 제안.
few-step 모델에서는 (\theta=0.25)만 적용해도 성능이 거의 유지되면서 latency가 크게 감소. ([ar5iv][6])

---

## ⚖️ 기존 모델과의 비교

| 항목                | 본 논문 (NAG)               | 기존 방법1 (CFG)    | 기존 방법2 (NASA/PAG류)      |
| ----------------- | ------------------------ | --------------- | ----------------------- |
| **적용 위치**         | attention feature (Z)    | 노이즈 예측(출력)      | attention/출력 혼합(방법별 상이) |
| **few-step 안정성**  | 높음(정규화+blending)         | 낮음(브랜치 발산으로 붕괴) | DiT 등에서 불안정 보고          |
| **multi-step 성능** | CFG/PAG 위에 추가 개선 가능      | 표준 강력           | 결합 가능/상황 의존             |
| **학습 필요**         | 없음(plug-in)              | 없음              | 없음                      |
| **계산 비용**         | 대체로 CFG보다 낮거나 비슷(모델에 따라) | 보통 추가 비용 큼      | 방법별 상이                  |
| **모달리티**          | 이미지 + 비디오까지 실험           | 주로 이미지          | 주로 이미지                  |

* NAG는 “**out-of-manifold drift를 억제하는 안정화 장치(정규화/블렌딩)**”를 설계의 중심에 둔 점이 차별점. ([ar5iv][6])

---

## 📉 실험 및 결과

### 세팅/지표

* **데이터셋**: COCO-5K prompts로 정량 평가 ([ar5iv][6])
* **보편 negative prompt(Universal)**: “Low resolution, blurry” (NASA 설정을 따라 사용) ([ar5iv][6])
* **지표**: CLIP Score(↑), FID(↓), PFID(↓), ImageReward(↑) ([ar5iv][6])
* **하드웨어**: (주로) NVIDIA A100에서 latency 측정 ([ar5iv][6])

---

### (A) Few-step 모델에서 NAG 단독 효과 (Table 1)

| Arch |             Model | Steps |      CLIP (↑) Base→NAG |          FID (↓) Base→NAG |         PFID (↓) Base→NAG |   ImageReward (↑) Base→NAG |
| ---- | ----------------: | ----: | ---------------------: | ------------------------: | ------------------------: | -------------------------: |
| DiT  |       SANA-Sprint |     2 | 31.4 → **31.9** (+0.5) | 30.29 → **28.31** (–1.98) | 37.56 → **33.29** (–4.27) | 1.008 → **1.075** (+0.067) |
| DiT  |      Flux-Schnell |     4 | 31.4 → **32.0** (+0.6) | 25.47 → **24.46** (–1.01) | 38.26 → **34.95** (–3.31) | 1.029 → **1.099** (+0.070) |
| DiT  | SD3.5-Large-Turbo |     8 | 31.4 → **31.8** (+0.4) | 29.97 → **29.81** (–0.18) | 44.37 → **41.87** (–2.50) | 0.944 → **1.118** (+0.174) |
| DiT  |          Flux-Dev |    25 | 30.9 → **31.5** (+0.6) | 31.04 → **28.11** (–2.93) | 43.22 → **39.01** (–4.21) | 1.066 → **1.166** (+0.100) |
| UNet |   NitroSD-Realism |     1 | 31.8 → **32.4** (+0.6) | 26.21 → **23.98** (–2.23) | 30.53 → **28.73** (–1.80) | 0.847 → **0.948** (+0.101) |
| UNet |         DMD2-SDXL |     4 | 31.6 → **32.2** (+0.6) | 24.79 → **23.32** (–1.47) | 27.11 → **25.61** (–1.50) | 0.876 → **0.960** (+0.084) |
| UNet |    SDXL-Lightning |     8 | 31.1 → **31.8** (+0.7) | 27.01 → **24.99** (–2.02) | 34.02 → **31.70** (–2.32) | 0.730 → **0.842** (+0.112) |

* 해석: few-step에서 **CLIP/품질(ImageReward)**가 거의 전 모델에서 상승하고, **FID/PFID도 대부분 개선**. 즉 “negative prompt가 진짜로 먹히게” 만들면서도 깨짐을 억제. ([ar5iv][6])

---

### (B) CFG/PAG와 결합 시 추가 이득 (Table 2)

| Arch | Model       | Steps | Setting |       CLIP (↑) w/o→NAG |           FID (↓) w/o→NAG |          PFID (↓) w/o→NAG |    ImageReward (↑) w/o→NAG |
| ---- | ----------- | ----: | ------- | ---------------------: | ------------------------: | ------------------------: | -------------------------: |
| DiT  | SD3.5-Large |    25 | CFG     | 31.8 → **32.0** (+0.2) |     25.07 → 25.42 (+0.35) |     31.68 → 31.63 (–0.05) | 1.029 → **1.130** (+0.101) |
| DiT  | SD3.5-Large |    25 | CFG+PAG | 31.5 → **31.8** (+0.3) | 24.49 → **24.35** (–0.14) |     37.93 → 39.09 (+1.16) | 0.939 → **1.063** (+0.124) |
| UNet | SDXL        |    25 | CFG     | 31.9 → **32.7** (+0.8) | 23.25 → **20.90** (–2.35) | 30.01 → **27.90** (–2.11) | 0.791 → **0.906** (+0.115) |
| UNet | SDXL        |    25 | CFG+PAG | 31.5 → **32.3** (+0.8) | 26.25 → **23.53** (–2.72) | 35.58 → **31.80** (–3.78) | 0.748 → **0.914** (+0.166) |

* 해석 포인트:

  * multi-step에서도 NAG는 “단독 대체”가 아니라 **기존 가이던스 위에 얹어 이득**을 주는 형태.
  * 다만 FID/PFID는 세팅에 따라 출렁일 수 있고, 저자도 “보완적으로 개선” 관점으로 정리. ([ar5iv][6])

---

### (C) 사용자 선호(User Study) (Table 3)

| Model        | Modal | Steps | CFG |    Text 선호 |  Visual 선호 |  Motion 선호 |
| ------------ | ----- | ----: | --- | ---------: | ---------: | ---------: |
| Flux-Schnell | Image |     4 | ✗   | **+25.0%** | **+33.9%** |          – |
| SD3.5-Large  | Image |    25 | ✓   |  **+9.2%** | **+15.5%** |          – |
| Wan2.1-14B   | Video |    25 | ✓   | **+20.5%** |  **+8.7%** | **+14.3%** |

* user study는 T2I에서 Pick-a-Pic v2 test에서 100 prompts, T2V에서 50 prompts 등으로 구성했다고 Appendix에서 설명. ([ar5iv][6])

---

### (D) 계산 비용(Per-step latency) (Table 4)

| Model family | Baseline |        CFG 추가 |       NAG 추가 |
| ------------ | -------: | ------------: | -----------: |
| Flux         |    487ms | +488ms (100%) | +426ms (87%) |
| SD3.5-Large  |    231ms |  +219ms (95%) | +109ms (43%) |
| SANA         |     39ms |   +35ms (90%) |   +5ms (13%) |
| SDXL         |     75ms |   +25ms (34%) |  +17ms (22%) |
| Wan2.1       |    10.7s | +10.7s (100%) |  +1.3s (12%) |

* 해석: 평균적으로 NAG는 **CFG 대비 추가 비용이 낮은 편**(특히 SANA/Wan2.1에서 크게 절감). 단 Flux에서는 거의 비슷. ([ar5iv][6])

---

### (E) Early stopping (Table 9, Appendix G) — “초반만 켜도 거의 유지”

아래는 “NAG 적용 비율 (\theta)”를 0.25/0.5/1.0로 바꾸며 **품질 vs 속도**를 본 결과 중 핵심만 요약:

* Flux-Schnell(4 step): (\theta=0.25)에서도 CLIP/FID/PFID/ImageReward가 거의 full((\theta=1)) 수준, latency는 +40%로 감소(풀 적용은 +78%).
* SDXL(25 step): (\theta=0.25)도 성능 상승 유지, latency 증가도 +3% 수준. ([ar5iv][6])

---

### (F) 추천 하이퍼파라미터 (Table 5, Appendix C)

| Architecture | Model family | nag_scale (s) | clip (\tau) | blend (\alpha) |
| ------------ | ------------ | ------------: | ----------: | -------------: |
| DiT          | Flux         |             4 |         2.5 |           0.25 |
| DiT          | SD3.5        |             4 |         2.5 |          0.125 |
| DiT          | SANA         |             2 |         2.5 |            0.5 |
| UNet         | SDXL         |             2 |         2.5 |            0.5 |
| UNet         | SD1.5        |             1 |         2.5 |            0.5 |

* 감: DiT 계열은 (s)를 4 근처로, UNet은 더 낮게(1~2) 두는 쪽이 기본값. (\tau)는 2.5로 고정에 가깝고, (\alpha)는 모델별로 안정성/강도 타협. ([ar5iv][6])

---

## ✅ 장점 및 한계

## **장점**:

* **few-step에서 negative prompt를 “실제로” 작동**시키는 데 초점이 맞춰져 있고, 정량/정성/유저스터디가 다 들어가 있음. ([ar5iv][6])
* **Training-free / plug-in**: 재학습 없이 inference에서 attention feature만 조작. ([ar5iv][6])
* **Universal**: UNet/DiT, 이미지/비디오(Wan2.1)까지 확장 실험. ([ar5iv][6])
* **안정화 설계가 명확**: “외삽 → (L1 clip) → (α-blend)”로 out-of-manifold를 구조적으로 막는 흐름이 분명. ([ar5iv][6])

## **한계 및 개선 가능성**:

* 여전히 **억제가 잘 안 되는 개념/프롬프트**가 존재하고, 너무 강한 scale이나 부적절한 negative prompt에서는 텍스처 붕괴/불안정이 남을 수 있다고 명시. ([ar5iv][6])
* 가이던스 강도 (s), 블렌딩 (\alpha) 선택이 모델/태스크에 따라 민감할 수 있음(저자도 scale에 따른 trade-off(정렬 vs 품질)를 보여줌). ([ar5iv][6])
* “더 미세한 토큰 단위 조절(token-wise modulation)” 같은 방향을 future work로 언급. ([ar5iv][6])

---

## 🧠 TL;DR – 한눈에 요약

> **Negative prompt가 few-step에서 망가지는 이유는 “출력 공간 외삽(CFG)”이 out-of-manifold로 튀기 때문이고,
> NAG는 이를 “attention feature 공간 외삽 + (L1 clip) + (α-blend)”로 안정화해, 학습 없이 범용 negative guidance를 복구한다.** ([ar5iv][6])

| 구성 요소  | 설명                                                                          |
| ------ | --------------------------------------------------------------------------- |
| 핵심 모듈  | attention feature (Z)에서 (Z^{+},Z^{-}) 외삽 + L1 기반 크기 제한((\tau)) + α-blending |
| 학습 전략  | 없음(완전 inference-time)                                                       |
| 전이 방식  | UNet/DiT/MM-DiT, 이미지/비디오로 그대로 적용                                            |
| 성능/효율성 | COCO-5K에서 CLIP/FID/PFID/ImageReward 개선 + 유저 선호 증가, 비용은 대체로 CFG보다 낮음(모델에 따라) |

---

## 🔗 참고 링크 (References)

* [📄 arXiv 논문](https://arxiv.org/abs/2505.21179) ([arXiv][1])
* [💻 GitHub](https://github.com/ChenDarYen/Normalized-Attention-Guidance) ([GitHub][4])
* [🧩 ComfyUI](https://github.com/ChenDarYen/ComfyUI-NAG) ([GitHub][5])
* [🌐 Project Page](https://chendaryen.github.io/NAG.github.io/) ([Chendaryen][2])
* [🎤 NeurIPS 2025 Poster](https://neurips.cc/virtual/2025/poster/117946) ([NeurIPS][3])

## 다음 논문:

* (추천) **Token Perturbation Guidance**, **Entropy Rectifying Guidance** 같이 “guidance 안정화” 계열을 바로 이어 읽으면, 너의 ANSWER/DNP/Verifier-guided 샘플링 아이디어랑 연결이 훨씬 쉬워져. ([NeurIPS][7])

[1]: https://arxiv.org/abs/2505.21179?utm_source=chatgpt.com "Normalized Attention Guidance: Universal Negative Guidance for Diffusion Model"
[2]: https://chendaryen.github.io/NAG.github.io/ "Normalized Attention Guidance: Universal Negative Guidance for Diffusion Models"
[3]: https://neurips.cc/virtual/2025/poster/117946 "NeurIPS Poster Normalized Attention Guidance: Universal Negative Guidance for Diffusion Models"
[4]: https://github.com/ChenDarYen/Normalized-Attention-Guidance "GitHub - ChenDarYen/Normalized-Attention-Guidance: Official implementation of \"Normalized Attention Guidance\""
[5]: https://github.com/ChenDarYen/ComfyUI-NAG "GitHub - ChenDarYen/ComfyUI-NAG: ComfyUI implemtation for NAG"
[6]: https://ar5iv.org/pdf/2505.21179 "[2505.21179] Normalized Attention Guidance: Universal Negative Guidance for Diffusion Models"
[7]: https://neurips.cc/virtual/2025/papers.html "NeurIPS 2025 Papers"
