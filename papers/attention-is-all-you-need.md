# 📘 Attention Is All You Need

## 1. 개요 (Overview)

* **제목**: Attention Is All You Need
* **저자**: Vaswani, Shazeer, Parmar, Uszkoreit, Jones, Gomez, Kaiser, Polosukhin
* **소속**: Google Brain
* **학회**: NeurIPS (NIPS) 2017
* **링크**: [arXiv](https://arxiv.org/abs/1706.03762), [GitHub](https://github.com/tensorflow/tensor2tensor)

> 논문 정리 첫 시작으로 트랜스포머 구조의 가장 기본이 된다고 생각하는 Attention Is All You Need를 선정함. 확실히 트랜스포머부터 개념이 약하다고 생각하여 여기서부터 시작해야겠다 생각함.
> 완전한 Self-Attention 기반 구조인 Transformer를 처음 제안한 논문. 순차처리 구조(RNN/CNN)를 제거하고, 병렬 처리 및 성능 향상을 모두 달성함.

---

## 2. 문제 정의 (Problem Formulation)

**문제 및 기존 한계**:

* Recurrent model에서는 입출력 시퀀스의 기호 위치에 따라 계산을 수행함. 계산 시간에 따라 위치를 정렬하면 이전 hidden state(h)와 위치(t)에 대한 hidden state sequence(h\_t)가 생성됨.
* 이러한 순차적 구조는 각 시퀀스 내부의 병렬 처리를 어렵게 만들며, 시퀀스 길이가 길어질수록 여러 시퀀스를 동시에 처리하는 데 메모리 한계로 인해 배치 구성이 제한되는 문제가 더욱 심각해짐.
* 이전 연구에서는 인수분해와 조건부 계산을 통해 효율을 개선했지만 sequential computation의 제약은 여전히 남아있음.

**제안 방식**:

* 순환 구조 없이 self-attention만으로 global dependency를 처리함
* 병렬화 가능하고, RNN/CNN 대비 단순하면서도 효과적인 구조를 가짐

> ※ **Global Dependency**: 입력 또는 출력 시퀀스 내의 멀리 떨어진 요소들 간의 의미적 연결. 문장 앞뒤에 있는 단어들 사이의 관계를 모델링하는 것.

---

## 3. 모델 구조 (Architecture)

### 전체 구조

![트랜스포머 구조](/papers/images/AIAYNmodel.png)

* Transformer는 **Encoder-Decoder 구조**로 구성되어 있음

* **인코더**: 각 레이어는 두 개의 서브레이어로 구성됨

  1. Multi-Head Self-Attention
  2. Position-wise Feed-Forward Network
     각 서브레이어에 대해 **Residual Connection + LayerNorm**을 적용함:

  $$
  \text{Output} = \text{LayerNorm}(x + \text{Sublayer}(x))
  $$

  모든 서브레이어 및 임베딩은 512차원의 출력을 생성함

* **디코더**: 인코더와 동일한 구조에 더해, 세 번째 서브레이어 추가:
  3\. Multi-Head Attention over Encoder Output

  디코더의 Self-Attention에는 \*\*마스킹(Masking)\*\*이 적용되어 **미래 위치의 정보 참조를 방지**함
  → i번째 위치의 예측은 오직 i보다 앞선 위치의 정보에만 의존함 (오토리그레시브)

> ✔️ **Residual Connection(잔차 연결)**:
>
> * 학습 안정화를 돕고, 기울기 소실을 방지
> * 원본 입력 정보를 그대로 유지하면서 새로운 표현만 추가함

---

## ✨ Attention 메커니즘 개요

* Attention은 query, key, value의 세 가지 벡터 쌍을 입력으로 받아 **가중합된 value**를 출력하는 구조

![어텐션 공식](/papers/images/attentioncalc.png)

---

### 💠 Scaled Dot-Product Attention

* Query와 모든 Key 간의 **내적 (dot product)** 을 계산하고, 각 결과를 \$\sqrt{d\_k}\$로 나눈 뒤
  **Softmax 함수를 적용하여 Value에 대한 가중치**를 얻는다.

* 여러 개의 Query를 \*\*행렬 \$Q$\*\*로 묶어 병렬적으로 계산하며, Key와 Value도 각각 \$K\$, \$V\$로 묶음

* 전체 Attention 계산 순서:

$$
\text{Query-Key 내적} \;\to\; \text{정규화} \;\to\; \text{Softmax} \;\to\; \text{Value 가중합}
$$

* 수식 표현:

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

---

## ⚖️ Additive Attention vs Dot-Product Attention

* 대표적인 어텐션 방식은 **Additive Attention**과 **Dot-Product (Multiplicative) Attention**
* Dot-Product 방식은 본 논문에서 제안한 방식과 거의 같으나, 기존 방식에는 스케일링이 없음

> Dot-Product Attention: 내적 후 softmax
> Additive Attention: FFN 기반 호환성 함수 사용

* Dot-Product는 **고속 행렬곱 기반으로 더 빠르고 메모리 효율이 높음**

---

## 📉 Scaling이 필요한 이유

* \$d\_k\$가 작을 때는 큰 차이가 없지만, **\$d\_k\$가 클 경우 Dot-Product의 크기가 너무 커져** softmax가 **gradient 소실 영역으로 들어감**

* 이를 방지하기 위해 **\$\frac{1}{\sqrt{d\_k}}\$로 스케일링하여 softmax 입력을 안정화**시킴

### 💠 Multi-Head Attention

- Query, Key, Value를 각각 $h$개의 선형 변환을 통해 $d_k$, $d_k$, $d_v$ 차원으로 투영
- 각 head에서 병렬로 Scaled Dot-Product Attention을 수행:

$$
\text{head}_i = \text{Attention}(Q W^Q_i, K W^K_i, V W^V_i)
$$

- 모든 head의 출력을 Concatenate 한 뒤, 또 한 번 선형 변환:

$$
\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, ..., \text{head}_h) W^O
$$

- 이 구조는 다양한 위치에서 다양한 표현 공간에 대해 동시에 주의를 집중하게 해 줌

#### 📌 논문 설정
- $h = 8$
- $d_k = d_v = \frac{d_{\text{model}}}{h} = 64$
- $W^Q_i, W^K_i, W^V_i \in \mathbb{R}^{d_{\text{model}} \times d_k}$
- $W^O \in \mathbb{R}^{hd_v \times d_{\text{model}}}$

> 여러 head를 사용하면 averaging에 의한 정보 손실을 방지하고, 더 다양한 관계를 동시에 포착 가능함.

### 💠 3.2.3 Attention의 활용 방식 

Transformer에서는 Multi-Head Attention이 **총 3가지 방식**으로 사용된다:

---

#### 1. 🧭 Encoder-Decoder Attention
- Query는 디코더에서, Key와 Value는 인코더의 출력에서 가져옴
- 디코더의 각 위치가 **입력 시퀀스 전체를 참조할 수 있게** 함
- 기존의 seq2seq 모델에서 사용되던 attention 방식과 유사함

---

#### 2. 🔁 Encoder Self-Attention
- Query, Key, Value 모두 **인코더의 동일한 위치 출력을 사용**
- 인코더의 각 위치는 **입력 시퀀스 전체 위치를 참조**할 수 있음

---

#### 3. ⏩ Decoder Self-Attention (Masked)
- Query, Key, Value 모두 **디코더의 이전 레이어 출력**
- 단, **i번째 위치는 i보다 크거나 같은 미래 위치를 참조하지 못하도록 마스킹 처리**
  - 마스킹은 softmax 입력에서 **불가능한 연결을 $-\\infty$로 설정하여 무효화**
- 이는 **오토리그레시브(autoregressive)** 성질을 유지하기 위함

> 요약:
> - Encoder-Decoder Attention → 디코더가 인코더 참조  
> - Encoder Self-Attention → 입력 내부 참조  
> - Decoder Self-Attention → 출력 내부 참조 + 마스킹

### 💠 3.3 Position-wise Feed-Forward Networks

- Encoder와 Decoder의 각 레이어에는 Attention 서브레이어 외에도 **Fully Connected Feed-Forward Network (FFN)**이 존재함.
- 이 FFN은 **각 위치(position)에 대해 독립적이고 동일하게 적용**됨.
- 구조는 다음과 같다:

$$
\text{FFN}(x) = \max(0, xW_1 + b_1) W_2 + b_2
$$

- ReLU 활성화 함수를 중심으로 **두 개의 선형 변환**으로 구성됨.
- 동일한 위치에 대해서는 동일한 FFN이 적용되지만, **레이어마다 다른 파라미터**를 사용함.
- **커널 크기 1의 convolution으로 해석할 수도 있음.**
- 논문에서는 입력 및 출력 차원 $d_{\text{model}} = 512$, 내부 차원 $d_{\text{ff}} = 2048$ 사용.

---

### 💠 3.4 Embeddings and Softmax

- 입력 토큰과 출력 토큰은 **학습된 임베딩(Embedding)**을 통해 $d_{\text{model}}$ 차원 벡터로 변환됨.
- 디코더의 출력은 **선형 변환 + Softmax**를 거쳐 다음 토큰에 대한 확률 분포를 생성함.

> 🎯 특징:
> - **입력 임베딩**, **출력 임베딩**, **pre-Softmax 선형 변환**에 **동일한 가중치 행렬(weight tying)** 사용  
> - 임베딩 행렬은 $\\sqrt{d_{\\text{model}}}$를 곱하여 스케일 조정됨

---

### 💠 3.5 Positional Encoding

- Transformer에는 **RNN이나 CNN이 없기 때문에**, 토큰 간 **순서 정보(position)**를 인코딩해야 함.
- 이를 위해 **Positional Encoding**을 입력 임베딩에 더해줌.
- Positional Encoding은 **임베딩과 동일한 차원($d_{\text{model}}$)**을 가지므로 더하기가 가능함.

> 논문에서는 **고정된(fixed)** 방식의 Positional Encoding 사용  
> (학습하는 방식도 실험했지만 큰 차이는 없음)

#### 📐 수식:

$$
PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)
$$

$$
PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)
$$

- $pos$는 위치, $i$는 임베딩의 차원 인덱스
- 각 차원은 서로 다른 주파수의 사인/코사인 함수를 사용
- 주파수는 $2\\pi$에서 $10000 \\cdot 2\\pi$까지 **기하급수적으로 증가**

> 이 방식은 **상대 위치에 따라 attention이 잘 작동하도록 유도**  
> (예: $PE_{pos+k}$는 $PE_{pos}$의 선형 함수로 표현 가능)

---

## 🧠 4. Why Self-Attention?

이 섹션에서는 Transformer가 기존의 RNN 또는 CNN 대신 **Self-Attention**을 사용하는 이유를 세 가지 측면에서 분석한다:

---

### 📌 1. 연산 복잡도 (Computational Complexity per Layer)

- **Self-Attention**: $\mathcal{O}(n^2 \cdot d)$
- **Recurrent Layer**: $\mathcal{O}(n \cdot d^2)$
- **Convolutional Layer**: $\mathcal{O}(k \cdot n \cdot d^2)$

> 여기서 $n$은 시퀀스 길이, $d$는 표현 차원, $k$는 커널 크기

- 대부분의 자연어 처리에서는 $n < d$이기 때문에 self-attention이 더 효율적
- **Separable convolution**을 사용해도, self-attention + FFN 조합과 연산량이 유사함

---

### 📌 2. 병렬화 가능성 (Parallelization)

- **Self-Attention**: 모든 위치에 대해 **동시에 연산 가능 (O(1) 순차 연산)**  
- **RNN**: 반드시 **이전 시점의 결과에 의존 (O(n) 순차 연산)**  
- **CNN**: 병렬화 가능하지만, **전체 위치 간 연결을 위해 여러 레이어 필요**

> ✅ 병렬화 관점에서 Self-Attention은 RNN보다 훨씬 유리함

---

### 📌 3. 장거리 의존성 학습 (Path Length for Long-Range Dependencies)

> 시퀀스의 위치 간 정보가 전달되기까지의 **최장 경로 길이**는 학습 난이도에 직접적인 영향을 줌

| 구조 | 최대 경로 길이 (Max Path Length) |
|------|-----------------------------------|
| Self-Attention | $\mathcal{O}(1)$ |
| RNN | $\mathcal{O}(n)$ |
| CNN | $\mathcal{O}(\log_k n)$ (dilated) 또는 $\mathcal{O}(n/k)$ (contiguous) |
| 제한된 Attention ($r$ 이웃) | $\mathcal{O}(n/r)$ |

- Self-Attention은 모든 위치 간 **직접 연결이 가능**하므로 **장거리 의존성 학습이 유리**
- RNN이나 CNN은 **여러 레이어 또는 시간 단계를 거쳐야만 정보가 전달**됨

---

### ✅ 부가적 장점: 해석 가능성 (Interpretability)

- 각 Attention Head가 **서로 다른 언어적 기능을 학습**하는 경향이 있음
- 문법적/의미적 구조에 따른 Attention 분포가 나타남 → **분석/시각화 용이**

---

## 🔍 요약

| 항목 | Self-Attention | RNN | CNN |
|------|----------------|-----|-----|
| 연산 복잡도 | $\mathcal{O}(n^2 \cdot d)$ | $\mathcal{O}(n \cdot d^2)$ | $\mathcal{O}(k \cdot n \cdot d^2)$ |
| 병렬화 가능성 | ✅ 매우 높음 (O(1)) | ❌ 낮음 (O(n)) | 🔶 중간 (병렬 가능) |
| 장거리 의존성 | ✅ 직접 연결 (O(1)) | ❌ 시간 순차 의존 | 🔶 여러 레이어 필요 |

> ✔️ 결론: Self-Attention은 **병렬성, 표현력, 연산 효율, 해석 가능성** 측면에서 RNN과 CNN을 모두 능가함


---

## 7. 한계 및 향후 연구 (Limitations & Future Work)

* Attention은 O(n^2) 메모리 복잡도 → 긴 시퀀스 비효율적
* 이후 연구에서는 Efficient Attention, Sparse Attention 등으로 확장


---

## 🧠 TL;DR – 한눈에 보는 핵심 요약

> **"Attention Is All You Need"**는 RNN/CNN 없이 **Self-Attention만으로 시퀀스를 처리**하는 새로운 모델 **Transformer**를 제안하였다.  
> 이 구조는 병렬 처리와 장거리 의존성 학습에 유리하며, 기존 방식 대비 연산 효율과 성능 모두 뛰어나다.

---

### 📌 핵심 구성 요소 요약

| 구성 요소 | 설명 |
|------------|----------------------------------------------------|
| Self-Attention | 입력 시퀀스 내 모든 위치 쌍의 관계를 한 번에 계산 |
| Multi-Head Attention | 여러 표현 공간에서 병렬적으로 attention 수행 |
| Feed-Forward Network | 각 위치에 독립적으로 적용되는 2-layer MLP |
| Positional Encoding | 순서 정보를 사인/코사인 함수로 인코딩해 입력에 추가 |
| Residual + LayerNorm | 학습 안정화 및 정보 보존 역할 수행 |

---

### ⚖️ 기존 구조와 비교 (RNN, CNN 대비)

| 항목 | Self-Attention | RNN | CNN |
|------|----------------|-----|-----|
| 병렬 처리 | ✅ 매우 우수 | ❌ 불가 | 🔶 가능 |
| 장거리 의존성 | ✅ 직접 연결 (O(1)) | ❌ O(n) | 🔶 여러 레이어 필요 |
| 해석 가능성 | ✅ 시각화 용이 | ❌ 낮음 | 🔶 제한적 |
| 연산 효율 | ✅ 효율적 (짧은 시퀀스) | ❌ 비효율적 | 🔶 커널 크기에 의존 |

---


### ✅ 결론

> Transformer는 **순차성 없는 병렬 구조**, **강력한 표현력**, **효율적인 계산**, **해석 가능성**이라는 네 마리 토끼를 모두 잡은 시퀀스 모델이다.  
> 이후 BERT, GPT, T5 등 대부분의 NLP 모델이 이 구조를 기반으로 발전하였다.

## 🔗 참고 링크 (References)

* [📄 arXiv 논문](https://arxiv.org/abs/1706.03762)
* [💻 공식 GitHub (Tensor2Tensor)](https://github.com/tensorflow/tensor2tensor)
* [📈 Papers with Code](https://paperswithcode.com/paper/attention-is-all-you-need)

##다음 논문: BERT
