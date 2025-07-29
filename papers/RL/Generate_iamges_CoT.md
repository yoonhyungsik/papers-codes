# 📘 Can We Generate Images with CoT? Let’s Verify and Reinforce Image Generation Step by Step

## 1. 개요 (Overview)

* **제목**: Can We Generate Images with CoT? Let’s Verify and Reinforce Image Generation Step by Step  
* **저자**: Ziyu Guo, Renrui Zhang, Chengzhuo Tong, Zhizheng Zhao, Peng Gao, Hongsheng Li, Pheng-Ann Heng  
* **소속**: CUHK MiuLar Lab, CUHK MMLab, Peking University, Shanghai AI Lab  
* **학회**: – (arXiv 예비 공개)  
* **링크**: [arXiv](https://arxiv.org/pdf/2501.13926) / [GitHub]() / [Papers with Code]()

> **논문 선정 이유 및 간단한 도입부**  
> Chain-of-Thought(CoT) 추론이 언어 모델에서 복잡한 문제 해결에 크게 기여해 왔지만, 이미지 생성 모델에 CoT를 적용한 연구는 드뭅니다. 이 논문은 CoT 전략을 단계별 검증 및 강화(reinforcement)에 적용하여, 이미지 생성 품질을 비약적으로 향상시킬 수 있음을 실험적으로 입증합니다.

---

## 2. 문제 정의 (Problem Formulation)

**문제 및 기존 한계**:  
* 자동회귀 이미지 생성은 토큰 단위로 점진적 디코딩하나, 중간 단계의 품질 검증이나 선호도 반영이 부족함.  
* 기존 Best-of-k 방식은 최종 후보만 평가해 중간 오류를 수정하지 못함.

**제안 방식**:  
* **Test-Time Verification**: 각 디코딩 단계에서 다수 후보를 생성하고, PARM 보상 모델로 중간 품질을 평가·선택.  
* **DPO 기반 Preference Alignment**: Direct Preference Optimization(DPO)으로 모델 선호도를 인간/보상 모델에 맞춰 미세 조정.  
* **Reflection 메커니즘 (PARM++)**: 최종 생성 후 불만족 영역을 식별해 재디코딩 또는 보정하는 자기반영 루프 도입.

> ※ **핵심 개념 정의**  
> - **자동회귀 이미지 생성 (autoregressive image generation)**: 토큰별로 순차 생성  
> - **Direct Preference Optimization (DPO)**: 선호도 학습을 통한 모델 파인튜닝

---
## 3. 모델 구조 (Architecture)

### 3.1 전체 파이프라인 개요

![전체 파이프라인](papers/images/Cot.png)

1. **텍스트 프롬프트 인코딩**  
   - Prompt 토큰을 텍스트 인코더(Transformer)로 임베딩  
   - 차원: $\mathbb{R}^{L_{\text{prompt}}\times d}$

2. **Show-o 기반 디코딩**  
   - 기본 디코더(Transformer)에서 이미지 토큰을 순차 생성  
   - 매 스텝 $t$마다 확률 분포 $P_{\theta}(x_t \mid x_{<t}, p)$ 계산

3. **Test-Time Verification (PARM)**  
   - 스텝 $t$별로 $k$개 후보 $\{x_t^{(i)}\}_{i=1}^k$ 생성  
   - PARM 보상 모델로 각 후보 점수 $r_{\mathrm{PARM}}(x_t^{(i)})$ 평가  
   - 최댓값을 갖는 토큰 선택:  
     $i^* = \displaystyle \arg\max_{1 \le i \le k} r_{\mathrm{PARM}}(x_t^{(i)})$\
       
     $x_t = x_t^{(i^*)}$

4. **DPO 기반 파인튜닝**  
   - 생성된 시퀀스 쌍 $(x^+, x^-)$에 대해 DPO 손실 계산\
     
      ![DPO Loss Function](https://latex.codecogs.com/svg.latex?\mathcal{L}_{\mathrm{DPO}}%20=%20-\log%20\sigma\left(s_\theta(x^+)%20-%20s_\theta(x^-)\right))
     
   - 여기서 $s_\theta(\cdot)$는 모델의 로그 확률 점수, $\sigma$는 시그모이드 함수

5. **Reflection 루프 (PARM++)**  
   - 완성 이미지에서 품질 저하 영역 $\Omega$ 식별  
   - 영역 $\Omega$ 재디코딩:  
     $x_{\Omega}' \sim P_{\theta}\bigl(x_{\Omega} \mid x_{<\Omega},\, p\bigr)$

   - 반복 횟수 $R$까지 수행하며,  
      ![Convergence Condition](https://latex.codecogs.com/svg.latex?\|I^{(r)}%20-%20I^{(r-1)}\|_2%20<%20\epsilon)
     에 도달하면 종료 
  
---

### 3.2 Show-o 베이스라인 디코더

- **구조**: 12-layer Transformer 디코더  
- **역할**: 텍스트 프롬프트 임베딩을 기반으로, 이미지 토큰을 한 단계씩 자동회귀 방식으로 생성하는 기본 생성기  
- **입력**: 텍스트 임베딩과 이전까지 생성된 이미지 토큰의 조합

  ![Input Formula](https://latex.codecogs.com/svg.latex?H_0%20=%20\text{Concat}(\text{PromptEmbeddings},%20\text{ImageTokens}_{<t}))

- **출력**: 다음 이미지 토큰 $x_t$에 대한 확률 분포 $P_\theta(x_t \mid x_{<t}, p)$ 계산

---

### 3.3 Test-Time Verification 모듈 (PARM)

**목적**: 디코딩 과정에서 무작정 하나의 토큰을 생성하는 것이 아니라, 후보들 중 “더 나은 선택”을 할 수 있도록 보상 모델을 활용해 품질을 평가

#### 3.3.1 PARM 네트워크 구조

- **입력 요소**:
  - 프롬프트 임베딩 ![p](https://latex.codecogs.com/svg.latex?p)
  - 이전까지의 이미지 토큰 ![x_<t](https://latex.codecogs.com/svg.latex?x_{<t})
  - 후보 토큰 ![x_t^(i)](https://latex.codecogs.com/svg.latex?x_t^{(i)})

- **모델 구성**:
  1. 6-layer Transformer Encoder (hidden size ![d=512](https://latex.codecogs.com/svg.latex?d=512), heads=8)
  2. MLP Head:
     - FC (512 → 256) + GeLU
     - FC (256 → 1)

- **출력**: 각 후보 토큰의 "가치"를 나타내는 스칼라 점수 $r_i$

  ![PARM Output](https://latex.codecogs.com/svg.latex?r_i%20=%20\text{PARM}\left(p,\;x_{<t},\;x_t^{(i)}\right))

> 이 점수는 보상 기반 선택 기준으로 사용되어, 다음 스텝에서 더 나은 토큰을 선택하게 해준다.

#### 3.3.2 점수 계산 및 토큰 선택

- $t$번째 디코딩 스텝에서 $k$개의 후보 토큰을 생성하고,
- PARM을 통해 각 후보의 점수를 계산한 뒤,
- 가장 높은 점수를 가진 토큰 $x_t^{(i^*)}$를 선택한다:

![Token Selection](https://latex.codecogs.com/svg.latex?i^*%20=%20\arg\max_{i}%20r_i,\quad%20r_i%20=%20\mathrm{PARM}\left(p,\,x_{<t},\,x_t^{(i)}\right),\quad%20x_t%20=%20x_t^{(i^*)})

---

### 3.4 Preference Alignment 모듈 (DPO)

**목적**: 사람이 선호할 만한 이미지 시퀀스를 모델이 더 잘 생성하도록 파인튜닝하는 전략. 단순 확률 최적화가 아닌, ‘더 나은 결과’를 판단할 수 있는 방식으로 학습한다.

#### 3.4.1 DPO 손실 정의

- DPO(Direct Preference Optimization)는 두 시퀀스 $(x^+, x^-)$ 중 어떤 것이 더 나은지를 기반으로 손실을 구성한다.
- 더 나은 쪽의 점수를 높이고, 열등한 시퀀스의 점수를 낮추는 방식이다.

![DPO Loss](https://latex.codecogs.com/svg.latex?\mathcal{L}_{\mathrm{DPO}}%20=%20-\log%20\sigma\left(s_\theta(x^+)%20-%20s_\theta(x^-)\right))

- $s_\theta(\cdot)$: 모델이 시퀀스에 부여하는 로그 확률 점수  
- $\sigma$: 시그모이드 함수

#### 3.4.2 학습 절차

1. 선호 쌍 ![preference pair](https://latex.codecogs.com/svg.latex?(x^+,%20x^-)) 샘플링 (보상 또는 사람 선택 기반)
2. Forward: 각 시퀀스에 대해 log-prob 계산  
   ![forward](https://latex.codecogs.com/svg.latex?s_\theta(x^+)), ![forward2](https://latex.codecogs.com/svg.latex?s_\theta(x^-))
3. Backward: 손실 함수의 그래디언트 계산  
   ![backward](https://latex.codecogs.com/svg.latex?\nabla_\theta%20\mathcal{L}_{\mathrm{DPO}})
4. Optimizer: AdamW 사용  
   ![eta](https://latex.codecogs.com/svg.latex?\eta=1\times10^{-5}), weight decay=0.01

> DPO는 강화학습(RLHF) 없이도 비교만으로 선호도 조정이 가능해 효율적이다.

---

### 3.5 Reflection 루프 (PARM++)

**목적**: 전체 이미지 생성 후, 품질이 떨어지는 부분만 부분적으로 수정하는 자기보정 메커니즘. LLM의 "반성(reflection)" 개념을 이미지 생성에 도입함.

#### 3.5.1 불만족 영역 식별

- 생성된 이미지를 $M \times M$ 패치로 나눈 뒤,
- 각 패치 $I_{ij}$에 대해 CLIPScore를 계산하여 기준 점수 $\tau$보다 낮은 영역을 식별

![Dissatisfaction Region](https://latex.codecogs.com/svg.latex?S_{ij}%20=%20\mathrm{CLIPScore}(p,\,I_{ij}),\quad%20\Omega%20=%20\{(i,j)\mid%20S_{ij}%20<%20\tau\})

- $\Omega$: 재디코딩 대상이 되는 저품질 영역들의 좌표 집합

#### 3.5.2 영역 재디코딩

- $\Omega$에 속한 영역의 토큰들만 다시 생성 (나머지 토큰은 고정)

![Region Redecoding](https://latex.codecogs.com/svg.latex?x_{\Omega}'%20\sim%20P_{\theta}\left(x_{\Omega}\mid%20x_{<\Omega},\,p\right))

- 이를 통해 전체 이미지를 다시 생성하지 않고, 문제 있는 영역만 개선할 수 있음

#### 3.5.3 반복 및 수렴 조건

- 위 과정을 최대 $R$번 반복하거나, 이미지 변화가 충분히 작아지면 종료:

```math
\|I^{(r)} - I^{(r-1)}\|_2 < \epsilon \quad\text{또는}\quad r = R
```

- PARM++는 한 번에 완성도 높은 이미지를 생성하기 어려운 경우, 자기반영을 통해 점진적으로 개선하는 기능을 수행한다.

---

## ⚖️ 기존 모델과의 비교

| 항목      | 본 논문 (CoT + PARM + DPO)         | 기존 방법1 (Show-o + Best-of-k) | 기존 방법2 (Stable Diffusion 3) |
| --------- | ---------------------------------- | ------------------------------- | ------------------------------- |
| 구조      | Autoregressive + Verification + Reflection | Autoregressive + 단순 후보 선택 | Diffusion 기반 multi-step decoding |
| 학습 방식 | DPO (선호도 학습), Reward 모델 동시 학습 | 없음 (pretrained only)          | 텍스트-이미지 정합 loss 기반 학습 |
| 목적      | 단계별 검증 및 선호도 강화, 자기반영 재생성 | 최종 결과 중 선택               | 고품질 이미지 생성               |

---

## 📉 실험 및 결과

* **데이터셋**: GenEval (텍스트-이미지 정합도 및 품질 평가용 벤치마크)
* **비교 모델**:
  - Show-o (기본 AR 모델)
  - Stable Diffusion 3 (SOTA Diffusion 모델)

* **주요 성능 지표 및 결과**:

| 모델                       | Accuracy / 정합도 | FID↓ | 기타 |
| -------------------------- | ----------------- | ---- | ---- |
| 본 논문 (CoT 통합 방식)     | +24% (Show-o 대비) | -    | GenEval 기준 최상 성능 |
| 기존 SOTA (Stable Diffusion 3) | +15%               | -    | 본 논문보다 낮은 정합도 |

> 실험 결과 요약 및 해석  
> 단계별 보상 검증(PARM), 선호 정렬(DPO), 반영 루프(PARM++)를 결합함으로써 단일 전략 또는 기존 SOTA보다 높은 이미지 품질과 정합도를 달성함. 특히 중간단계 검증과 자기 수정 루프가 품질 향상에 큰 기여.

---

## ✅ 장점 및 한계

## **장점**:

* Chain-of-Thought 방식의 이미지 생성 최초 적용 사례
* 각 디코딩 단계에서 품질 검증 및 보상 모델을 통한 후보 선택 가능
* DPO 기반의 선호 정렬로 학습 성능 극대화
* 자기반영 메커니즘을 통해 이미지 품질 자체 보정 가능

## **한계 및 개선 가능성**:

* 추론 시간(test-time) 비용 증가: 후보 생성 및 평가 과정이 병렬화되지 않으면 느림
* PARM, PARM++와 같은 보상 모델의 학습이 필수적이며 추가 자원 요구
* Diffusion 모델 기반 구조로의 일반화는 아직 미적용 상태

---

## 🧠 TL;DR – 한눈에 요약

> **Chain-of-Thought 추론 전략을 이미지 생성 과정에 도입하여, 각 단계별 품질 검증(Test-Time Verification), 선호도 학습(DPO), 자기반영(Reflection)을 통합함으로써 GenEval에서 +24%의 성능 향상을 달성한 연구**

| 구성 요소      | 설명 |
| ------------- | ---- |
| 핵심 모듈      | PARM, PARM++ (보상 + 반영) |
| 학습 전략      | DPO 기반 선호 정렬 |
| 전이 방식      | Show-o 기반 AR 디코더에 CoT 구조 통합 |
| 성능/효율성    | GenEval 기준 Show-o 대비 +24%, SD3 대비 +15% 향상 |

---

## 🔗 참고 링크 (References)

* [📄 arXiv 논문](https://arxiv.org/pdf/2501.13926)
* [💻 GitHub]()  
* [📈 Papers with Code]()

## 다음 논문:
