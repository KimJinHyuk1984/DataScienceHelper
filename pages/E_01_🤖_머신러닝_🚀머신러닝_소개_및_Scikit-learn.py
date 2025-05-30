# pages/1_🚀_머신러닝_소개_및_Scikit-learn.py
import streamlit as st
import pandas as pd # Scikit-learn API 설명 시 DataFrame 예시용
import numpy as np  # Scikit-learn API 설명 시 NumPy 배열 예시용
# from utils_ml import get_dataset # 이 페이지에서는 직접 사용하지 않을 수 있음

st.header("1. 머신러닝 소개 및 Scikit-learn")

st.subheader("1.1 머신러닝이란?")
st.markdown("""
머신러닝(Machine Learning, 기계 학습)은 명시적인 프로그래밍 없이 컴퓨터가 데이터로부터 학습하여 특정 작업을 수행할 수 있도록 하는 인공 지능(AI)의 한 분야입니다.
데이터를 기반으로 패턴을 인식하고, 예측하며, 결정을 내리는 모델을 구축하는 것을 목표로 합니다.

**핵심 아이디어:** 데이터에 숨겨진 규칙이나 관계를 찾아내어, 새로운 데이터에 대한 예측이나 분류를 수행합니다.
""")

st.subheader("1.2 머신러닝의 주요 유형")
col1, col2 = st.columns(2)
with col1:
    st.markdown("""
    #### 📝 지도 학습 (Supervised Learning)
    - **정의:** 입력 데이터(특성, Features)와 해당 데이터에 대한 정답(레이블, Label 또는 Target)이 함께 주어진 상태에서 학습합니다.
    - **목표:** 입력 데이터와 정답 간의 관계를 모델링하여, 새로운 입력 데이터에 대한 정답을 예측합니다.
    - **주요 작업:**
        - **회귀 (Regression):** 연속적인 값을 예측합니다. (예: 주택 가격 예측, 주가 예측)
        - **분류 (Classification):** 데이터를 미리 정의된 범주(클래스) 중 하나로 분류합니다. (예: 스팸 메일 분류, 이미지 분류 - 개/고양이)
    """)
with col2:
    st.markdown("""
    #### 🎨 비지도 학습 (Unsupervised Learning)
    - **정의:** 정답(레이블)이 없는 입력 데이터만으로 학습합니다.
    - **목표:** 데이터 자체의 숨겨진 구조, 패턴, 관계를 발견합니다.
    - **주요 작업:**
        - **군집화 (Clustering):** 유사한 특성을 가진 데이터들을 그룹으로 묶습니다. (예: 고객 세분화)
        - **차원 축소 (Dimensionality Reduction):** 데이터의 특성 수를 줄이면서 중요한 정보는 최대한 유지합니다. (예: 고차원 데이터 시각화, 노이즈 제거)
        - **연관 규칙 학습 (Association Rule Learning):** 데이터 항목 간의 흥미로운 관계를 찾습니다. (예: 장바구니 분석 - 맥주와 기저귀)
    """)
st.markdown("""
#### 🦾 강화 학습 (Reinforcement Learning) - 간략 소개
- **정의:** 에이전트(Agent)가 환경(Environment)과 상호작용하며, 특정 행동(Action)에 대한 보상(Reward) 또는 벌점(Penalty)을 통해 학습합니다.
- **목표:** 누적 보상을 최대화하는 최적의 행동 정책(Policy)을 학습합니다.
- **예시:** 게임 AI (알파고), 로봇 제어, 자율 주행 등.
- *이 도우미 앱에서는 강화 학습을 상세히 다루지 않습니다.*
""")
st.markdown("---")

st.subheader("1.3 머신러닝 주요 용어")
st.markdown("""
- **특성 (Features / Attributes / Variables):** 모델 학습에 사용되는 입력 데이터의 개별 측정 가능한 속성입니다. (X, 독립 변수)
- **레이블 (Label / Target / Class):** 지도 학습에서 모델이 예측하려는 대상 값입니다. (y, 종속 변수)
- **샘플 (Sample / Instance / Observation):** 데이터셋의 각 행, 즉 하나의 관측치를 의미합니다.
- **훈련 데이터셋 (Training Set):** 머신러닝 모델을 학습시키는 데 사용되는 데이터입니다.
- **테스트 데이터셋 (Test Set):** 학습된 모델의 성능을 평가하기 위해 사용되는, 모델 학습에 사용되지 않은 데이터입니다.
- **모델 (Model):** 데이터로부터 학습된 특정 패턴이나 규칙의 표현입니다. (예: 선형 회귀 모델, 결정 트리 모델)
- **과적합 (Overfitting):** 모델이 훈련 데이터에는 매우 잘 맞지만, 새로운 데이터(테스트 데이터)에는 성능이 낮은 현상입니다. 모델이 데이터의 노이즈까지 학습한 경우 발생합니다.
- **과소적합 (Underfitting):** 모델이 훈련 데이터의 패턴조차 제대로 학습하지 못하여 성능이 낮은 현상입니다. 모델이 너무 단순한 경우 발생합니다.
- **편향-분산 트레이드오프 (Bias-Variance Tradeoff):**
    - **편향(Bias):** 실제 값과 모델 예측값 간의 차이. 높은 편향은 과소적합을 의미합니다.
    - **분산(Variance):** 훈련 데이터셋이 약간 변경될 때 모델 예측값이 얼마나 변하는지 나타내는 척도. 높은 분산은 과적합을 의미합니다.
    - 일반적으로 편향과 분산은 서로 상충 관계에 있어, 둘 사이의 적절한 균형을 찾는 것이 중요합니다.
""")
st.markdown("---")

st.subheader("1.4 Scikit-learn 소개 및 기본 API")
st.markdown("""
**Scikit-learn**은 파이썬에서 가장 널리 사용되는 머신러닝 라이브러리 중 하나입니다.
다양한 알고리즘과 사용하기 쉬운 API를 제공하여 머신러닝 파이프라인을 효율적으로 구축할 수 있도록 돕습니다.
""")

st.markdown("#### Scikit-learn의 주요 특징 및 철학:")
st.markdown("""
- **일관성 (Consistency):** 대부분의 알고리즘(Estimator)이 유사한 인터페이스를 가집니다.
- **검사 (Inspection):** 모델의 하이퍼파라미터와 학습된 파라미터를 쉽게 확인할 수 있습니다.
- **제한된 객체 계층 구조 (Limited Object Hierarchy):** 알고리즘은 파이썬 클래스로, 데이터셋은 NumPy 배열, Pandas DataFrame, SciPy 희소 행렬 등으로 표현됩니다. 파라미터는 일반적인 파이썬 문자열이나 숫자를 사용합니다.
- **구성 (Composition):** 여러 빌딩 블록(전처리기, 모델 등)을 조합하여 복잡한 머신러닝 파이프라인을 쉽게 구성할 수 있습니다. (예: `Pipeline` 객체)
- **합리적인 기본값 (Sensible Defaults):** 대부분의 모델 파라미터에 합리적인 기본값이 설정되어 있어, 빠르게 프로토타이핑하고 결과를 확인할 수 있습니다.
""")

st.markdown("#### Scikit-learn의 핵심 API - Estimator 인터페이스")
st.markdown("""
Scikit-learn의 모델들은 **Estimator**라는 공통 인터페이스를 따릅니다. 주요 메소드는 다음과 같습니다.
""")
code_sklearn_api = """
# from sklearn.some_module import SomeEstimator # 예시 Estimator
# from sklearn.model_selection import train_test_split
# import numpy as np
# import pandas as pd

# # 0. 데이터 준비 (X: 특성, y: 레이블)
# # X_data = np.array([[1,2],[3,4],[5,6], [7,8]]) # 특성 데이터 (샘플 수 x 특성 수)
# # y_data = np.array([0,1,0,1])                   # 레이블 데이터 (샘플 수)

# # # 0.1 훈련 데이터와 테스트 데이터 분리 (선택 사항이지만 일반적)
# # X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.25, random_state=42)


# # 1. Estimator 객체 생성 (모델 초기화)
# # model = SomeEstimator(hyperparameter1=value1, hyperparameter2=value2)
# # 예: model = LogisticRegression(C=1.0, penalty='l2')


# # 2. 모델 학습 (fit 메소드)
# # model.fit(X_train, y_train) # 훈련 데이터(X_train)와 해당 레이블(y_train)로 모델 학습


# # 3. 예측 (predict 메소드) - 분류 또는 회귀
# # y_pred = model.predict(X_test) # 테스트 데이터(X_test)에 대한 예측 수행

# # (분류 모델의 경우) 예측 확률 (predict_proba 메소드)
# # y_pred_proba = model.predict_proba(X_test) # 각 클래스에 속할 확률 반환


# # 4. 데이터 변환 (transform 메소드) - 전처리기 또는 차원 축소 모델
# # (만약 SomeEstimator가 전처리기라면)
# # X_transformed = model.transform(X_data_to_transform)

# # (학습과 변환을 동시에: fit_transform 메소드)
# # X_train_transformed = model.fit_transform(X_train) # 훈련 데이터에 학습 및 변환 동시 적용
# # X_test_transformed = model.transform(X_test)     # 테스트 데이터에는 학습 없이 변환만 적용


# # 5. 모델 평가 (score 메소드 또는 별도 평가 함수 사용)
# # accuracy = model.score(X_test, y_test) # 분류 모델의 경우 정확도, 회귀 모델은 R^2 점수 반환
# # from sklearn.metrics import mean_squared_error
# # mse = mean_squared_error(y_test, y_pred) # 회귀 모델의 MSE 계산
"""
st.code(code_sklearn_api, language='python')
st.markdown("""
**주요 메소드 요약:**
- `fit(X, y)`: 모델을 데이터 `X`와 레이블 `y`(지도학습의 경우)로 학습시킵니다. 전처리기의 경우, 데이터 `X`에서 필요한 통계량(예: 평균, 표준편차)을 계산합니다.
- `predict(X)`: 학습된 모델을 사용하여 새로운 데이터 `X`에 대한 예측값을 반환합니다. (분류: 클래스 레이블, 회귀: 연속적인 값)
- `predict_proba(X)`: (분류 모델) 각 클래스에 속할 확률을 반환합니다.
- `transform(X)`: (전처리기, 차원 축소 모델) 데이터 `X`를 변환합니다. (예: 스케일링, 인코딩, 차원 축소)
- `fit_transform(X, y=None)`: `fit()`과 `transform()`을 순차적으로 적용합니다. 훈련 데이터에 주로 사용됩니다. (주의: 테스트 데이터에는 `transform()`만 사용해야 함)
- `score(X, y)`: (지도학습 모델) 테스트 데이터 `X`와 실제 레이블 `y`를 사용하여 모델의 기본 평가 지표(분류: 정확도, 회귀: R²)를 반환합니다.
- `get_params()`: 모델의 하이퍼파라미터를 딕셔너리 형태로 반환합니다.
- `set_params(**params)`: 모델의 하이퍼파라미터를 설정합니다.

이러한 일관된 API 덕분에 Scikit-learn에서는 다양한 모델과 전처리기를 쉽게 교체하고 실험해볼 수 있습니다.
""")
st.markdown("---")
st.markdown("다음 페이지부터는 데이터 전처리, 주요 머신러닝 모델, 모델 평가 방법에 대해 구체적인 예시와 함께 살펴보겠습니다.")