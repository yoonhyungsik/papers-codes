# 🎯 Transformer 어텐션에서 Q, K, V의 역할

Transformer의 Self-Attention 연산은 세 가지 주요 구성요소인 **Query (Q)**, **Key (K)**, **Value (V)**로 이루어져 있습니다. 이들은 각각 **입력 토큰을 서로 어떻게 주목할 것인지 결정**하는 데에 사용됩니다.

---

## 🔍 1. Q / K / V의 직관적 의미

| 구성 요소 | 의미 | 역할 비유 |
|-----------|------|------------|
| **Query (Q)** | 주의를 줄 기준 | 질문지 |
| **Key (K)** | 각 항목의 특징 | 책 제목 |
| **Value (V)** | 정보의 실제 내용 | 책 내용 |

- Query는 "어디에 집중할까?"라는 질문
- Key는 "각 항목이 무엇을 말하고 있는지"
- Value는 "그 항목이 실제로 담고 있는 정보"

---

## 🧠 2. 예시로 이해하기

문장 예시:
> `"The animal didn't cross the street because it was too tired."`

- "it" → **Query** (무엇을 가리키는지 알고 싶음)
- "animal", "street" → **Key** (후보들)
- 각 단어의 의미 → **Value**

어텐션은 다음 과정을 거칩니다:
1. "it" (Query)와 모든 단어(Key)의 유사도 계산
2. 유사도에 따라 어떤 단어를 더 중요하게 여길지 결정
3. 선택된 Key에 대응하는 Value를 조합해 최종 출력 생성

---

## 🧮 3. 수식 관점

어텐션은 다음과 같이 정의됩니다:

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$



# 🔍 Transformer 연산 과정: NLP vs CV 비교

Transformer는 NLP와 Computer Vision(CV)에서 모두 폭넓게 사용되지만,  
**입력 구조와 처리 방식에는 차이점**이 있습니다.  
이 문서에서는 **입력 → 출력까지의 연산 과정을 NLP와 CV 관점에서 나눠서 설명**합니다.

---

## 🧠 1. 공통 개요

| 단계 | NLP (텍스트 입력) | CV (이미지 입력: ViT 등) |
|------|--------------------|---------------------------|
| **1. 입력 데이터** | 텍스트 시퀀스 (단어 or subword) | 이미지 (HxWxC) |
| **2. 임베딩 변환** | Token Embedding + Positional Encoding | Patch Embedding + Positional Encoding |
| **3. QKV 생성** | 단어 임베딩 → 선형 변환 | 패치 임베딩 → 선형 변환 |
| **4. 어텐션 계산** | 단어 간 문맥 관계 계산 | 패치 간 시각 관계 계산 |
| **5. FFN + Residual** | 위치별 비선형 변환 | 지역 패치 특징 강화 |
| **6. N층 반복** | 문맥 심화 | 시각 패턴 심화 |
| **7. 출력 사용** | 분류, 생성, 질의응답 등 | 분류, 감지, 세분화 등 |

---

## 📘 2. NLP에서의 Transformer 연산 흐름 (예: BERT)

입력 문장 → 토큰화 → 임베딩 → 포지셔널 인코딩 → N개의 인코더 → 문맥 임베딩 출력

### 🔹 주요 단계 설명 (NLP 기준: BERT 등)

#### 1. 토큰화 (Tokenization)
- 입력 문장 예: `"I love you"`
- 토큰화: `["I", "love", "you"]`
- 정수 인덱스 변환 (예: BERT tokenizer 기준):  
  `[101, 1045, 2293, 2017, 102]`  
  - `101`: [CLS], `102`: [SEP]

---

#### 2. Token Embedding + Positional Encoding
- 각 토큰을 고정 차원의 벡터로 임베딩 (예: 768차원)
- 순서 정보를 표현하기 위해 **포지셔널 인코딩(Position Encoding)** 추가
  - $\text{Input} = \text{TokenEmbedding} + \text{PositionEmbedding}$

---

#### 3. Multi-Head Self-Attention
- 각 토큰이 전체 시퀀스 내 다른 토큰과의 **의미적 관계**를 학습
- 입력 임베딩을 통해 Q, K, V 생성:
  - $Q = XW^Q$, $K = XW^K$, $V = XW^V$
- 어텐션 수식:
  $\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$

---

#### 4. Feed-Forward Network (FFN)
- 각 위치별로 독립적으로 작동하는 MLP
- 일반 구조:
  $\text{FFN}(x) = \text{GELU}(xW_1 + b_1)W_2 + b_2$
- 대부분 두 개의 Linear Layer + 활성화 함수 (GELU, ReLU 등)

---

#### 5. Residual Connection + LayerNorm
- 어텐션 및 FFN 연산 후 **원래 입력과 더해주는 skip 연결** 수행
- 이후 **Layer Normalization**으로 안정화
- 구성:
  x ← x + Attention(x)
  x ← LayerNorm(x)
  x ← x + FFN(x)
  x ← LayerNorm(x)


## 🖼️ 3. CV에서의 Transformer 연산 흐름 (예: ViT)

입력 이미지 → 패치 분할 → 패치 임베딩 → 포지셔널 인코딩 → N개의 인코더 → 출력 벡터

### 🔹 주요 단계 설명 (CV 기준: Vision Transformer)

#### 1. 이미지 입력
- 예: RGB 이미지 크기 $224 \times 224 \times 3$

---

#### 2. Patchify + Linear Embedding
- 이미지를 $16 \times 16$ 패치로 분할
- 총 패치 수: $(224 / 16)^2 = 196$
- 각 패치를 펼친 후, Linear Layer를 통해 고정 차원 임베딩 (예: 768차원)

---

#### 3. Positional Encoding
- 각 패치에 위치 정보 추가
- 방식:
  - 고정된 사인/코사인 포지셔널 인코딩
  - 또는 학습 가능한 위치 임베딩

---

#### 4. Multi-Head Self-Attention
- 각 패치가 다른 패치들과의 시각적 관계를 학습
- Q, K, V는 패치 임베딩에서 선형 변환하여 생성됨

---

#### 5. Feed-Forward Network (FFN)
- 각 패치 위치에 독립적인 MLP (두 개의 Linear Layer + 활성화 함수) 적용

---

#### 6. Residual Connection + LayerNorm
- 어텐션 및 FFN 출력에 대해:
  - Skip Connection 수행 (입력 + 출력)
  - Layer Normalization 적용

---

#### 7. 출력 활용
- 분류:
  - [CLS] 토큰 추출 → Linear Layer → Softmax
- 또는:
  - 모든 패치 출력 평균 풀링 (Mean Pooling)
  - 후속 디코더로 전달 (예: Detection, Segmentation 등)

---

### ✅ 4. NLP vs CV 핵심 비교 요약

| 요소 | NLP (BERT 등) | CV (ViT 등) |
|------|----------------|--------------|
| **입력 단위** | 단어, subword 토큰 | 이미지 패치 (예: $16 \times 16$) |
| **입력 길이** | 가변 (예: 128개 토큰) | 고정 (예: 196개 패치) |
| **위치 정보** | 1D 포지셔널 인코딩 | 2D 기반 인코딩 (평면 위치) |
| **Q, K, V 의미** | 단어 간 문맥 관계 계산 | 패치 간 시각 관계 계산 |
| **출력 목적** | 문장 분류, 생성, QA 등 | 이미지 분류, 객체 감지, 세분화 등 |

---




