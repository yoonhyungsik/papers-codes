# 📘 Attention Is All You Need 

## 1. 개요 (Overview)

* **제목**: Attention Is All You Need
* **저자**: Vaswani, Shazeer, Parmar, Uszkoreit, Jones, Gomez, Kaiser, Polosukhin
* **소속**: Google Brain
* **학회**: NeurIPS (NIPS) 2017
* **링크**: [arXiv](https://arxiv.org/abs/1706.03762), [GitHub](https://github.com/tensorflow/tensor2tensor)

> 완전한 Self-Attention 기반 구조인 Transformer를 처음 제안한 논문. 순차처리 구조(RNN/CNN)를 제거하고, 병렬 처리 및 성능 향상을 모두 달성함.

---

## 2. 문제 정의 (Problem Formulation)

* **문제**: 기계 번역 (Sequence-to-Sequence)
* **기존 한계**: RNN/CNN 기반 모델은 병렬화가 어렵고, 장거리 의존성(long-range dependency) 학습이 비효율적임
* **제안 방식**: 모든 위치 간 관계를 Self-Attention으로 계산해, 장기 의존성 문제 해결 + 병렬 처리 가능

---

## 3. 모델 구조 (Architecture)

### 전체 구조

* Encoder-Decoder 구조
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
