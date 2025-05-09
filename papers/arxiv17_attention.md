# 📘 Attention Is All You Need 

## 1. 개요 (Overview)

* **제목**: Attention Is All You Need
* **저자**: Vaswani, Shazeer, Parmar, Uszkoreit, Jones, Gomez, Kaiser, Polosukhin
* **소속**: Google Brain
* **학회**: NeurIPS (NIPS) 2017
* **링크**: [arXiv](https://arxiv.org/abs/1706.03762), [GitHub](https://github.com/tensorflow/tensor2tensor)

> 논문 정리 첫 시작으로 트랜스포머 구조의 가장 기본이 된다고 생각하는 Attention Is All You Need를 선정함. 확실히 트랜스포머 부터 개념이 약하다고 생각하여 여기서부터 시작해야겠다 생각함.
> 완전한 Self-Attention 기반 구조인 Transformer를 처음 제안한 논문. 순차처리 구조(RNN/CNN)를 제거하고, 병렬 처리 및 성능 향상을 모두 달성함.

---

## 2. 문제 정의 (Problem Formulation)

* **문제 및 기존한계**: Reccurent model에서는 입출력 시퀀스의 기호 위치에 따라 계산을 수행함. 계산 시간에 따라 위치를 정렬하면 이전 hidden state(h)와 위치(t)에 대한 hidden state sequence(h_t)가 생성됨. 이러한 순차적 구조는 각 시퀀스 내부의 병렬 처리를 어렵게 만들며,
시퀀스 길이가 길어질수록 여러 시퀀스를 동시에 처리하는 데 메모리 한계로 인해 배치 구성이 제한되는 문제가 더욱 심각해짐. 이전 연구에서는 인수분해와 조건부 계산을 통해 효율을 개선했지만 sequential computation의 제약은 여전히 남아있음.
* 
* 
* **제안 방식**: 순환 없이 self-attention만으로 global dependency를 처리
* 
병렬화 가능, RNN/CNN 대비 단순하고 효과적인 구조
* 
※Global Dependency
입력 또는 출력 시퀀스 내의 멀리 떨어진 요소들 간의 의존 관계
즉, 문장이나 시퀀스의 처음과 끝처럼 서로 멀리 떨어진 위치 사이에 의미적인 연결이나 영향을 주고받는 관계를 의미.
* 


---

## 3. 모델 구조 (Architecture)

### 전체 구조

* Encoder-Decoder 구조
* 
* 인코더에는 두개의 하위 레이어가 존재. 각각 multi-head self attention과 단순 feed forward network로 구성
두개의 하위 레이어 각각에 residual connection을 사용한 후 layer norm진행. 즉, 각 서브 레이어의 출력은 $ \text{LayerNorm}(x + \text{Sublayer}(x)) $
이러한 residual connection 용이하게 하기 위해 모델의 모든 sub-layer와 embedding layer는 512차원의 출력 생성.

* 디코더의 경우 인코더와 마찬가지로 multi-head self attention과 단순 feed forward network를 가짐. 여기에 인코더 스택의 출력에 대해  multi-head self attention 수행하는 세번째 sub-layer 추가.
디코더의 self-attention 서브레이어는 특정 위치가 자기보다 뒤에 있는 위치를 참조하지 못하도록(masking) 변경됨. 이 마스킹은, 출력 임베딩이 한 칸 뒤로(offset) 밀려 있는 구조와 결합되어 i번째 위치의 예측이 오직 i보다 작은 위치들의 출력에만 의존하도록 만든다. -> 미래 단어를 미리 보면 안 되기 때문 (→ 오토리그레시브 방식 유지)

## ※Residual Connection(잔차 연결)
수식은:
$$
\text{Output} = \mathcal{F}(x) + x
$$
여기서 $\mathcal{F}(x)$는 레이어를 거친 출력이고, $x$는 입력값.
✔️ 학습 안정화
깊은 네트워크일수록 기울기 소실/폭주 문제가 생김
Residual 연결은 기울기가 역전파될 경로를 직접 만들어줌
✔️ 정보 보존
이전의 원본 입력 정보를 그대로 유지하면서 새로운 특징만 더하게 됨

*Attention은 query,key-value쌍을 출력에 매핑하는것으로 설명가능함.(여기서 모든 값들은 벡터) 출력은 가중치 합으로 계산됨.

* 각 블록은 Multi-Head Attention + Feed-Forward Network
* Residual Connection + Layer Normalization 포함

### 주요 컴포넌트 설명

#### 💠 Scaled Dot-Product Attention

$\text{Attention}(Q, K, V) = \text{softmax}(\frac{QK^T}{\sqrt{d_k}})V$

#### 💠 Multi-Head Attention

* 여러 개의 Attention Head를 병렬로 사용해 다양한 표현을 학습

#### 💠 Positional Encoding

* 순서를 알 수 없기 때문에, 입력에 위치 정보를 더함
* 수식:
  $PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{model}}}\right)$
  $PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{model}}}\right)$

---

## 4. 학습 방식 (Training & Optimization)

* **Loss**: Cross-Entropy Loss (Label Smoothing 적용)
* **Optimizer**: Adam ($\beta_1=0.9, \beta_2=0.98$)
* **Learning Rate Schedule**:
  $lr = d_{model}^{-0.5} \cdot \min(step^{-0.5}, step \cdot warmup^{-1.5})$
* **Dropout**: 0.1 적용
* **Parameter Initialization**: Xavier init

---

## 5. 실험 설정 (Experiment Settings)

* **데이터셋**:

  * WMT 2014 English-German (4.5M 문장쌍)
  * WMT 2014 English-French (36M 문장쌍)
* **Tokenization**: Byte-Pair Encoding (BPE)
* **배치 크기**: 25,000 token per batch

---

## 6. 결과 분석 (Results & Evaluation)

| 모델          | BLEU (En-De) | 학습 시간 | 파라미터 수 |
| ----------- | ------------ | ----- | ------ |
| Transformer | **28.4**     | 빠름    | 65M    |
| GNMT        | 24.6         | 느림    | 213M   |

* **Ablation**:

  * Head 수, Depth, Positional Encoding 유무 실험
* **성능 지표**: BLEU (Bilingual Evaluation Understudy Score)

---

## 7. 한계 및 향후 연구 (Limitations & Future Work)

* Attention은 O(n^2) 메모리 복잡도 → 긴 시퀀스 비효율적
* 이후 연구에서는 Efficient Attention, Sparse Attention 등으로 확장

---

## 8. 관련 연구와 비교 (Related Works)

| 모델           | 구조             | 병렬화 가능 | 장거리 의존성 |
| ------------ | -------------- | ------ | ------- |
| LSTM Seq2Seq | RNN 기반         | ❌      | 제한적     |
| GNMT         | LSTM+Attention | 🔶 부분적 | 보완적     |
| Transformer  | Attention-only | ✅      | 효과적     |

---

## 9. 논문 읽으며 메모한 포인트 (Notes)

* Attention 연산이 시각적으로 명확해 분석 용이
* Residual + LayerNorm이 학습 안정화에 큰 기여
* Positional Encoding은 사인-코사인 함수로 매우 우아하게 해결

---

## 10. 한 줄 요약 (TL;DR)

> "RNN 없이도 순차 학습이 가능하며, Self-Attention만으로 병렬성 + 성능을 모두 달성한 획기적 모델"

---

## 🔗 참고 링크 (References)

* [📄 arXiv 논문](https://arxiv.org/abs/1706.03762)
* [💻 공식 GitHub (Tensor2Tensor)](https://github.com/tensorflow/tensor2tensor)
* [📈 Papers with Code](https://paperswithcode.com/paper/attention-is-all-you-need)
