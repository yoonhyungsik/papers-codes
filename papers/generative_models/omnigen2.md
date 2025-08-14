# 📘 OmniGen2: Exploration to Advanced Multimodal Generation

## 1. 개요 (Overview)

- **제목**: OmniGen2: Exploration to Advanced Multimodal Generation  
- **저자**: Chenyuan Wu, Pengfei Zheng, Ruiran Yan, Shitao Xiao, Xin Luo, Yueze Wang, Wanli Li, Xiyan Jiang, Yexin Liu, Junjie Zhou, Ze Liu, Ziyi Xia, Chaofan Li, Haoge Deng, Jiahao Wang, Kun Luo, Bo Zhang, Defu Lian, Xinlong Wang, Zhongyuan Wang, Tiejun Huang, Zheng Liu  
- **소속**: Beijing Academy of Artificial Intelligence (BAAI)  
- **학회 / 공개 형태**: arXiv preprint (2025-06-23, v1 / 2025-06-25, v2)  
- **링크**:  
  - [arXiv](https://arxiv.org/abs/2506.18871)  
  - [GitHub](https://github.com/VectorSpaceLab/OmniGen2)  
  - [Papers with Code](https://huggingface.co/papers/2506.18871)  

> 이 논문은 멀티모달 생성 분야의 최신 진보를 보여주는 연구로, 텍스트-이미지 생성, 이미지 편집, 인컨텍스트(subject-driven) 생성, 그리고 **자기반성(reflection) 메커니즘**까지 통합한 완결형 프레임워크를 제안.
> 특히 **텍스트와 이미지의 디코딩 경로를 분리한 dual-path 구조**와 **비결합(decoupled) 이미지 토크나이저**를 적용해, 언어 생성 능력을 유지하면서 이미지 생성 품질을 향상시킴.
> 또한, 새로운 **OmniContext** 벤치마크를 제안해 인컨텍스트 생성 성능을 체계적으로 평가하며, 오픈소스로 공개되어 다양한 응용과 확장이 가능.

---

## 2. 문제 정의 (Problem Formulation)

**문제 및 기존 한계**:

* 기존 멀티모달 생성 모델들은 **텍스트-이미지 생성, 이미지 편집, 인컨텍스트 생성**과 같은 다양한 작업을 단일 모델에서 모두 수행하기 어려움.
* 단일 경로(shared-parameter) 아키텍처를 사용할 경우,  
  - **언어 생성 능력이 이미지 토크나이저에 의해 훼손**되거나,  
  - **이미지 생성 품질이 언어 전용 구조에 의해 제한**되는 상호 간섭 문제가 발생.
* 기존 모델들은 **in-context(subject-driven) 생성**에 특화된 데이터와 평가 지표가 부족해, 해당 능력을 체계적으로 훈련·검증하기 어려움.
* 생성 결과의 **자체 품질 평가 및 반복 개선(reflection)** 메커니즘 부재로, 제어 가능성과 품질 일관성이 떨어짐.

**제안 방식**:

* **Dual-Path Decoding 구조** 도입:  
  - **텍스트**는 autoregressive 디코더로 처리.  
  - **이미지**는 diffusion transformer 기반 디코더로 처리.  
  - 두 경로의 파라미터를 분리해 상호 간섭 최소화.
* **Decoupled Image Tokenizer** 적용:  
  - 텍스트 처리 시 이미지 토큰의 부정적 영향을 제거하고,  
  - 이미지 생성 시 고품질 비주얼 표현 유지.
* **Reflection Mechanism** 구축:  
  - 모델이 자신의 출력물을 다시 입력받아 품질 평가 및 개선을 수행.
* **OmniContext 벤치마크** 제안:  
  - in-context 생성 과제를 평가하기 위한 새로운 데이터셋과 지표 설계.
* 다양한 멀티모달 작업을 **하나의 통합 모델**로 처리하면서도,  
  언어와 이미지 양쪽에서 높은 품질 유지.

> ※ **핵심 개념 정의**
> - **In-Context Generation**: 프롬프트에 포함된 예시(subject/instance) 정보를 바탕으로 유사 스타일·콘텐츠를 생성하는 기법.  
> - **Decoupled Image Tokenizer**: 텍스트와 이미지 토큰 임베딩 경로를 분리해 언어 모델의 성능 저하를 방지하는 토크나이저 구조.  
> - **Reflection Mechanism**: 모델이 생성 결과를 스스로 평가하고 반복적으로 개선하는 피드백 루프.


---

## 3. 모델 구조 (Architecture)
![omnigen_architecture][../images/omnigen2_architecture.png]
### 전체 구조

- **Dual‑path 디코딩**: 텍스트는 **autoregressive LM 경로**, 이미지는 **diffusion transformer 경로**로 **완전히 분리된 파라미터**를 사용해 디코딩한다. 이때 **MLLM의 히든 상태**가 이미지 디퓨전 경로의 조건(condition)으로 전달되며, **이미지 저수준 정보는 VAE 인코더**에서만 추출해 디퓨전 경로에 **독점적으로** 주입한다. 이렇게 **ViT(이해)** ↔ **VAE(생성)** 을 분리한 **decoupled 설계**가 언어 능력을 보존하면서 시각 품질과 일관성을 끌어올린다.
- **모드 전환 토큰**: 출력 시퀀스에 **`<|img|>`** 특수 토큰을 삽입하면 이미지 생성 모드가 트리거되고, 그 시점의 **MLLM 히든 상태**를 조건으로 **디퓨전 디코더**가 이미지를 합성한다.
- **Omni‑RoPE(멀티모달 RoPE)**: 텍스트/이미지/다중 이미지 간 포지션·정체성 정보를 안정적으로 구분·정렬하기 위해 **(i) 시퀀스·모달 식별자 `id_seq`**, **(ii) 2D 높이 좌표 `h`**, **(iii) 2D 너비 좌표 `w`** 의 세 성분으로 분해된 RoPE를 사용한다. 동일 이미지에 속한 모든 토큰은 같은 `id_seq`를 공유하고, 각 이미지 단위로 (0,0)에서 **지역적 2D 좌표**를 다시 계산해 편집/인컨텍스트에서의 **위치 일관성**을 강화한다.

---

### 💠 핵심 모듈 또는 구성 요소

#### 📌 Multimodal LLM 백본 (Text Path)
- **구성/역할**: **Qwen2.5‑VL‑3B**로 초기화한 MLLM이 텍스트·비전 입력을 이해하고, 텍스트 출력은 **AR 언어 헤드**로 생성한다. 이미지 생성 시에는 시퀀스 내 **`<|img|>`** 토큰을 만나는 순간 **디퓨전 경로**를 호출하여 해당 시점의 **MLLM 히든 상태**를 조건으로 넘긴다. 대부분의 MLLM 파라미터는 **동결**, 새 토큰 임베딩만 업데이트한다.

$\log p_{\theta}(\mathbf{y}|\mathbf{x}) = \sum_{t=1}^{T} \log p_{\theta}(y_t | y_{<t}, \mathbf{x})$

여기서 $\mathbf{x}$는 (텍스트/이미지) 컨텍스트, $\mathbf{y}$는 텍스트 출력 시퀀스. 이미지 생성의 경우 $h_{MLLM}$이 조건으로 전달된다.

- **설계 의도**: **학습 가능한 쿼리 토큰 몇 개**로 모든 조건을 압축하는 최근 방식과 달리, OmniGen2는 **인터리브된 조건의 히든 상태 전체**를 디퓨전에 제공해 긴 프롬프트·복잡한 제약에서도 정보 손실을 줄인다.

#### 📌 Decoupled Visual Encoders (ViT & VAE)
- **ViT (이해 경로)**: 이미지 이해(질의응답·설명·지시 해석)는 **ViT 인코더 → MLLM**으로 진행한다. ViT는 **고수준 의미** 정렬을 담당한다.
- **VAE (생성 경로)**: **저수준 텍스처·정확한 픽셀 일관성**은 **VAE 인코더**가 추출한 라틴(latent) 특징을 **디퓨전 디코더**에 **직접** 공급하여 확보한다. **VAE 특징은 MLLM으로는 보내지 않는다**(언어 능력 보존/중복 인코딩 회피).

#### 📌 Diffusion Transformer Decoder (Image Path; RF 기반)
- **입력/토큰화**: 디퓨전 트랜스포머는 **MLLM 히든 상태(텍스트 기반 조건)**, **VAE 특징(저수준 시각 정보)**, **노이즈 토큰**을 **동일 시퀀스로 결합**해 **공동 어텐션(joint attention)**으로 처리한다. 전처리로 **refiner 네트워크**가 멀티 조건을 정렬한다( Lumina‑Image 2.0의 설계를 차용 ).
- **아키텍처 규모**: **32‑layer, hidden size 2520, ≈4B 파라미터**. 효율을 위해 **MLLM 내 이미지 관련 히든 상태는 폐기**하고 **텍스트 토큰 히든만 유지**하여 조건으로 사용한다(중복/비용 절감).
- **학습/샘플링(개념)**: OmniGen2의 이미지 경로는 **Rectified Flow(RF)**로 학습·추론된다. RF는 확률 흐름 ODE의 **속도장 v_θ**를 학습해 노이즈→데이터로의 연속적 변환을 푼다("flow‑matching"류의 목적함수).

$\frac{dx_t}{dt} = v_{\theta}(x_t, t|c), \quad x_{t=0} \sim \mathcal{N}(0,I), \quad x_{t=1} \approx \text{data}$

$c$는 MLLM 히든 + VAE 특징으로 구성된 조건. (논문은 RF 사용을 명시)

#### 📌 Omni‑RoPE (Multimodal Rotary Position Embedding)
- **핵심 아이디어**: 포지션/정체성 정보를 **(id_seq, h, w)** 삼분하여 텍스트·여러 이미지·편집 대상/참조 대상 간 **명확한 구분**과 **위치 일관성**을 동시에 달성. 동일 이미지의 토큰들은 **공유** `id_seq`, 각 이미지에 대해 **지역 2D 좌표 (h,w)** 를 독립 계산해 대응 위치가 **동일 임베딩**을 갖도록 유도한다.

$\text{Omni-RoPE}(\text{token}) \leftarrow f(\text{id}_{\text{seq}}, h, w)$

텍스트만 있을 때는 1D RoPE로 **자연스럽게 축소**된다.

#### 📌 Refiner & Conditioning Interface
- **Refiner**: 여러 조건(텍스트 히든, 이미지 레퍼런스, 편집 마스크 등)을 **정렬·정규화**해 디퓨전 트랜스포머 입력으로 통합한다( Lumina‑Image 2.0 참조 ).
- **Condition Pruning**: **VAE 특징을 명시적으로 주입**하므로, MLLM의 **이미지 히든은 버리고** 텍스트 히든만 유지해 **계산량을 절감**한다.

### 입력/출력 흐름 (간단 요약)
1. 입력(텍스트/이미지) → **ViT→MLLM**으로 이해, 텍스트 생성은 **AR 헤드**가 직접 출력
2. 출력 시퀀스 중 `<|img|>` 등장 → **MLLM 히든(텍스트 기반)**을 조건으로 사용
3. **VAE 인코더**가 저수준 시각 특징 추출 → **디퓨전 트랜스포머**에 **Omni‑RoPE**와 함께 투입
4. **RF 기반 샘플링**으로 고품질 이미지 합성 → 필요 시 **반성(Reflection)** 루프를 통해 반복 개선(별도 섹션)

---

## 4. 학습 방법론 (Training Methodology)

### 📌 데이터셋 구성
- **X2I 데이터셋**: OmniGen2를 위해 이미지 편집 및 인컨텍스트 생성 데이터를 포함하는 종합적인 데이터 구축 파이프라인을 개발
- **멀티태스크 통합**: 다양한 생성 태스크(텍스트→이미지, 편집, 인컨텍스트 생성)를 하나의 통일된 포맷으로 구성
- **비디오 기반 Subject-Driven 데이터**: 비디오 데이터를 기반으로 한 특별히 설계된 학습 파이프라인을 통해 우수한 주체 일관성과 맥락적 통합을 구현

### 📌 학습 전략
- **단계적 학습**: MLLM 백본의 대부분 파라미터는 동결하고, 새로운 토큰 임베딩과 디퓨전 디코더만 학습
- **Rectified Flow 기반**: 확률 흐름 ODE를 활용한 효율적인 이미지 생성 학습
- **조건부 생성**: MLLM 히든 상태와 VAE 특징을 조건으로 하는 joint attention 학습

---

## 5. 반성 메커니즘 (Reflection Mechanism)

### 📌 자가 개선 시스템
- **내장 반성 능력**: OmniGen2의 특징적인 기능으로, 자체 출력을 평가하고 단점을 식별한 후 반복적 개선을 통해 향상된 결과를 생성하는 내장 반성 메커니즘
- **전용 반성 데이터셋**: OmniGen2 기반으로 큐레이션된 반성 전용 데이터셋 구축
- **반복적 개선**: 초기 생성 → 품질 평가 → 개선된 재생성의 순환 구조

### 📌 품질 향상 과정
$\text{Initial Generation} \rightarrow \text{Self-Evaluation} \rightarrow \text{Identify Issues} \rightarrow \text{Refined Generation}$

---

## 6. 성능 및 벤치마크 (Performance & Benchmarks)

### 📌 모델 규모 및 효율성
- **파라미터 수**: 38억 개 파라미터로 SD3 모델(127억 개)보다 3배 이상 작은 크기
- **아키텍처 간소화**: 기존 디퓨전 모델의 인코더-디코더 구조와 달리, 추가 텍스트 인코더 비용을 제거한 간소화된 구조
- **경쟁력 있는 성능**: 상대적으로 적은 파라미터 수에도 불구하고 여러 태스크 벤치마크에서 경쟁력 있는 결과 달성

### 📌 평가 벤치마크
- **OmniContext 벤치마크**: 일관된 인컨텍스트 생성을 위한 전용 평가 벤치마크 출시
- **다중 태스크 평가**: 텍스트→이미지, 이미지 편집, Subject-Driven 생성에서 종합 평가
- **주관적 품질**: 이전 오픈소스 모델들을 능가하는 주체 일관성과 맥락적 통합 성능

---

## 7. 주요 응용 분야 (Key Applications)

### 📌 핵심 기능들
1. **텍스트→이미지 생성**: 자연어 설명으로부터 고품질 이미지 합성
2. **지시 기반 이미지 편집**: 객체 조작, 스타일 변경, 모션 편집 등 세밀한 수정이 가능하며, 편집되지 않은 영역과 시각적 현실감 및 일관성을 유지
3. **인컨텍스트 생성**: 참조 이미지를 활용한 맥락적 이미지 생성
4. **Subject-Driven 생성**: 참조 이미지에서 주체를 추출하고 텍스트 프롬프트에 따라 새로운 장면에서 재렌더링

### 📌 실용적 장점
- **통합 솔루션**: 여러 모델이 필요했던 작업들을 단일 모델로 처리
- **ControlNet/IP-Adapter 불필요**: 추가 모듈 없이 다양한 제어 조건 처리 가능
- **유연한 입력**: 텍스트와 이미지가 임의로 섞인 인터리브 입력 지원

---

## 8. 기술적 혁신점 요약

### 📌 OmniGen v1 대비 개선사항
- **파라미터 분리**: 텍스트와 이미지 모달리티에 대해 별도의 디코딩 경로와 공유되지 않는 파라미터 사용
- **디커플링된 이미지 토크나이저**: 분리된 이미지 토크나이저로 기존 멀티모달 이해 모델 위에 구축 가능
- **원본 능력 보존**: VAE 입력을 재적응할 필요 없이 원래 텍스트 생성 능력을 보존

### 📌 설계상의 차별점
- **언어/이미지 파라미터 완전 분리**(공유 없음)로 상호 간섭 제거
- **VAE 특징은 생성 경로에만** 사용(MLLM에 주입하지 않음) → 언어 능력 보존/중복 인코딩 제거
- **쿼리 토큰 압축 대신 MLLM 히든 전량 활용** → 긴 프롬프트·복합 제약 대응력 향상
- **Omni‑RoPE**로 다중 이미지·편집 상황에서 위치/정체성 일관성 강화
- **내장 반성 메커니즘**으로 자가 품질 개선 능력 제공

## ⚖️ 기존 모델과의 비교

| 항목    | 본 논문 (OmniGen2) | 기존 방법1 (OmniGen v1) | 기존 방법2 (일반 unified MLLM+Diffusion) |
| ----- | ---------------- | ----------------------- | ---------------------------------------- |
| 구조    | Dual-path 디코딩 (텍스트: AR LM / 이미지: Diffusion Transformer), Decoupled VAE & ViT, Omni-RoPE | 단일 파라미터 공유 구조, 텍스트·이미지 경로 결합 | 단일 경로 또는 shared backbone, query token 기반 조건 주입 |
| 학습 방식 | 멀티태스크 학습 (T2I, 이미지 편집, 인컨텍스트, Reflection), OmniContext 벤치마크 기반 | T2I 중심 멀티태스크, Reflection 미지원 | T2I 또는 편집 단일 과제 중심, in-context 생성/Reflection 미지원 |
| 목적    | 언어·이미지 품질 동시 극대화, 멀티모달 전방위 생성, 제어 가능성 확보 | 텍스트-이미지 생성 통합, 초기 멀티모달 시도 | 단일 과제 최적화, 범용성 부족 |

---

## 📉 실험 및 결과

* **데이터셋**:
  - OmniContext (in-context 생성 평가)
  - 다양한 공개 T2I·편집 데이터셋 (COCO, EditBench 등)
  - Reflection용 자가생성 품질 개선 데이터
* **비교 모델**:
  - OmniGen v1
  - BAGEL, Lumina-Image 2.0 등
  - 기타 오픈소스 MLLM+Diffusion 모델
* **주요 성능 지표 및 결과**:
  - T2I: CLIP-Score, FID, IS
  - 편집: EditScore, SSIM
  - In-context: Consistency Score (OmniContext 기준)
  - Reflection: 개선 전/후 품질 지표

| 모델      | Accuracy / Consistency | F1 / CLIP-Score | FID ↓ | 기타 |
| ------- | ---------------------- | --------------- | ----- | ---- |
| 본 논문    | 최고 성능 (오픈소스 중)  | +상승폭 2~5%     | 개선   | Reflection로 품질 향상 |
| 기존 SOTA | 다소 낮음               | 기준치           | 기준치 | Reflection 미지원 |

> **실험 결과 요약 및 해석**  
> OmniGen2는 기존 오픈소스 멀티모달 생성 모델 대비 **in-context 생성 일관성**과 **편집 품질**에서 우위. Reflection 메커니즘 도입으로 동일 조건에서 평균 품질이 향상됨. Dual-path 구조가 언어/이미지 성능 간 trade-off를 줄임.

---

## ✅ 장점 및 한계

### **장점**:
* 언어와 이미지 디코딩 경로를 완전히 분리해 **상호 간섭 최소화**
* **Omni-RoPE**로 다중 이미지·편집 상황에서도 위치 일관성 유지
* **Reflection** 메커니즘으로 자동 품질 개선
* **OmniContext** 벤치마크로 in-context 생성 성능을 체계적으로 평가

### **한계 및 개선 가능성**:
* Reflection 반복 시 연산 비용 증가
* 파라미터 수(≈4B)와 훈련 데이터 규모로 인해 소규모 환경에서 학습 난이도 높음
* 텍스트 외 모달(예: 오디오, 비디오)로의 확장성은 후속 연구 필요

---

## 🧠 TL;DR – 한눈에 요약

> **Dual-path 구조 + Decoupled Tokenizer + Reflection 메커니즘**을 통해 언어·이미지 품질을 동시에 극대화한 차세대 멀티모달 생성 프레임워크.  
> OmniContext 벤치마크에서 오픈소스 최고 수준의 in-context 생성 일관성을 달성하며, 오픈소스로 활용 가능.

| 구성 요소  | 설명 |
| ------ | -- |
| 핵심 모듈  | Dual-path 디코딩, Decoupled VAE & ViT, Omni-RoPE, Reflection |
| 학습 전략  | 멀티태스크 학습 (T2I, 편집, in-context, Reflection) |
| 전이 방식  | Qwen2.5-VL-3B 기반, Lumina-Image 2.0 refiner 구조 차용 |
| 성능/효율성 | 언어·이미지 모두 고성능, in-context 일관성 최고, 오픈소스 활용 가능 |

---

## 🔗 참고 링크 (References)

* [📄 arXiv 논문](https://arxiv.org/abs/2506.18871)
* [💻 GitHub](https://github.com/VectorSpaceLab/OmniGen2)
* [📈 Papers with Code](https://huggingface.co/papers/2506.18871)


