# 📘 Neural Machine Translation by Jointly Learning to Align and Translate

## 1. 개요 (Overview)

* **제목**: Neural Machine Translation by Jointly Learning to Align and Translate
* **저자**: Dzmitry Bahdanau, Kyunghyun Cho, Yoshua Bengio
* **소속**: Université de Montréal
* **학회**: ICLR 2015
* **링크**: [arXiv:1409.0473](https://arxiv.org/abs/1409.0473)

> 최초로 "Attention" 메커니즘을 제안한 논문으로, 기존 RNN 기반 번역기의 성능 한계를 극복하기 위해 디코더가 인코더의 전체 출력을 **가변적으로 참조**할 수 있도록 한 방식이다.

---

## 2. 문제 정의 (Problem Formulation)

**문제 및 기존 한계**:

* 기존 RNN 기반 Encoder-Decoder는 **고정된 context 벡터**만으로 전체 입력 시퀀스를 요약
* 긴 문장일수록 정보가 손실되어 번역 성능이 떨어짐
* 모든 정보를 하나의 벡터로 압축하면 디코더가 모든 의미 정보를 반영하기 어려움

**제안 방식**:

* 디코더가 인코더의 **모든 hidden state를 동적으로 가중합**하여 context 벡터 생성
* 각 디코딩 단계마다 "어디에 집중할지" 결정하는 **soft attention mechanism** 도입

> ※ Attention은 alignment를 학습하며, 번역 중 어떤 입력 단어에 집중할지를 모델이 자동으로 선택함

---

## 3. 모델 구조 (Architecture)

### 전체 구조

* 전통적인 RNN Encoder-Decoder 구조를 기반으로 하되, **context vector를 고정하지 않고 동적으로 생성**
* 인코더는 양방향 RNN (Bi-RNN)으로 구성되어 각 입력 위치 \$i\$에서 hidden state \$h\_i\$를 생성
* 디코더는 이전 상태 \$s\_{t-1}\$, 이전 출력 \$y\_{t-1}\$, 동적 context \$c\_t\$를 이용하여 출력 생성

---

### 💠 Attention Mechanism (Additive Attention)

#### 📌 작동 과정 (단계별 설명)

1. **Score 계산**: 디코더의 현재 상태 \$s\_{t-1}\$와 인코더의 각 hidden state \$h\_i\$를 비교해 score \$e\_{ti}\$를 계산

$$
e_{ti} = v_a^T \tanh(W_a s_{t-1} + U_a h_i)
$$

2. **Softmax 정규화**: 각 score에 대해 softmax를 적용하여 가중치 \$\alpha\_{ti}\$를 계산

$$
\alpha_{ti} = \frac{\exp(e_{ti})}{\sum_j \exp(e_{tj})}
$$

3. **Context Vector 계산**: 가중치 \$\alpha\_{ti}\$와 hidden state \$h\_i\$의 가중합으로 context vector \$c\_t\$ 계산

$$
c_t = \sum_i \alpha_{ti} h_i
$$

4. **디코더 출력**: \$(c\_t, y\_{t-1})\$를 디코더 RNN의 입력으로 사용하여 새로운 hidden state \$s\_t\$를 생성하고, 최종 출력 생성

#### 📌 개념적으로 보면:

* Attention은 **query** (디코더 상태), **key/value** (인코더 상태)의 쌍으로 구성됨
* 디코더는 각 시점마다 “입력의 어떤 위치에 주목할지”를 학습함
* 이는 기계 번역 과정에서의 **단어 alignment** 문제를 유연하게 해결함

#### 📌 추가 개념: Alignment Model의 역할

* alignment 함수 \$a(s\_{t-1}, h\_i)\$는 디코더 상태와 인코더 상태 간의 \*\*유사도(score)\*\*를 측정함
* additive attention은 \$s\_{t-1}\$과 \$h\_i\$를 **concat → tanh → 선형 변환**하여 score 생성
* 이 구조는 **비선형성과 학습 가능한 파라미터**를 통해 soft attention weight를 유도함

#### 📌 시각적 해석

* Attention Weight를 시각화하면 **어떤 입력 단어가 어떤 출력 단어와 연결되는지**를 확인 가능
* 이는 모델의 해석 가능성 (interpretability)을 높여주는 중요한 장점
* 번역뿐 아니라 다른 시퀀스 학습 문제에서도 어디를 참조했는지 알 수 있음

#### 📌 핵심 차별점

* 기존 모델은 context를 압축해 손실을 유발했으나, attention은 **모든 입력에 soft하게 접근 가능**
* 위치에 따른 alignment 가중치를 직접 학습함으로써 **더 정밀한 정보 선택**이 가능

---

## 🧠 TL;DR – 한눈에 요약

> 디코더가 인코더의 hidden state 전체에 대해 가중치를 학습해 동적으로 집중하는 **Attention mechanism**을 처음 도입.
> 이 방식은 RNN의 정보 압축 한계를 극복하고, 이후 Transformer로 이어지는 Attention 시대의 문을 열었다.

| 구성 요소     | 설명                                 |
| --------- | ---------------------------------- |
| Encoder   | Bi-RNN (입력 시퀀스 인코딩)                |
| Attention | Additive attention (score = FFN)   |
| Decoder   | RNN (context + previous output 기반) |
| 핵심 기여     | soft alignment 학습 방식 제안            |

---

## 🔗 참고 링크 (References)

* [📄 arXiv 논문](https://arxiv.org/abs/1409.0473)
* [💻 GitHub 구현 (예시)](https://github.com/keon/seq2seq)
* [📈 Papers with Code](https://paperswithcode.com/paper/neural-machine-translation-by-jointly)

## 다음 논문: Attention Is All You Need
