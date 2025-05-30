# pages/3_📈_회귀_모델_Regression.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer # 데이터셋에 따라 결측치 처리용
from utils.utils_ml import get_dataset, display_regression_metrics # 유틸리티 함수 사용

st.header("3. 회귀 모델 (Regression Models)")
st.markdown("""
회귀는 하나 이상의 독립 변수(특성)를 사용하여 연속적인 종속 변수(타겟)의 값을 예측하는 지도 학습 기법입니다.
예를 들어, 주택의 특징(크기, 방 개수, 위치 등)을 바탕으로 주택 가격을 예측하거나, 광고비 지출에 따른 매출액을 예측하는 데 사용될 수 있습니다.
""")

# --- 예제 데이터셋 로드 (캘리포니아 주택 가격 데이터) ---
st.subheader("캘리포니아 주택 가격 데이터셋을 사용한 회귀 예제")
try:
    df_housing, _, feature_names_housing = get_dataset("california_housing")
except Exception as e:
    st.error(f"캘리포니아 주택 가격 데이터셋 로드 중 오류 발생: {e}")
    st.info("Scikit-learn 데이터셋 서버에 일시적인 문제가 있을 수 있습니다. 나중에 다시 시도해주세요.")
    df_housing = None # 오류 발생 시 df_housing을 None으로 설정

if df_housing is not None:
    if st.checkbox("캘리포니아 주택 가격 데이터셋 미리보기 (상위 5행)", key="show_housing_df_page_3"):
        st.dataframe(df_housing.head())
        st.write(f"데이터 형태: {df_housing.shape}")
        st.write("특성 (Features):", feature_names_housing)
        st.write("타겟 (Target): 주택 가격 중앙값 (MedHouseVal)")
        st.write("결측치 확인:")
        st.dataframe(df_housing.isnull().sum().rename("결측치 수"))

    st.markdown("---")

    # --- 3.1 데이터 전처리 (회귀용) ---
    st.subheader("3.1 데이터 전처리 (회귀용)")
    st.markdown("""
    회귀 모델, 특히 선형 모델의 경우 특성 스케일링이 성능에 영향을 줄 수 있습니다.
    여기서는 간단히 특성(X)과 타겟(y)을 분리하고, 훈련/테스트 데이터로 나눈 후, 특성을 표준화합니다.
    """)

    # 특성(X)과 타겟(y) 분리
    X_housing = df_housing.drop('target', axis=1)
    y_housing = df_housing['target']

    # 훈련 데이터와 테스트 데이터 분리
    X_train_h, X_test_h, y_train_h, y_test_h = train_test_split(X_housing, y_housing, test_size=0.2, random_state=42)

    # 특성 스케일링 (StandardScaler 사용)
    # 참고: 실제로는 훈련 데이터에 fit_transform을, 테스트 데이터에는 transform만 적용해야 데이터 유출(data leakage)을 방지합니다.
    scaler_housing = StandardScaler()
    X_train_scaled_h = scaler_housing.fit_transform(X_train_h) # 훈련 데이터로 fit 및 transform
    X_test_scaled_h = scaler_housing.transform(X_test_h)     # 테스트 데이터는 transform만

    if st.checkbox("전처리된 데이터 일부 확인", key="show_preprocessed_housing_data_page_3"):
        st.write("훈련용 특성 데이터 (스케일링 후, 상위 3행):")
        st.dataframe(pd.DataFrame(X_train_scaled_h, columns=X_housing.columns).head(3).round(3))
        st.write("테스트용 특성 데이터 (스케일링 후, 상위 3행):")
        st.dataframe(pd.DataFrame(X_test_scaled_h, columns=X_housing.columns).head(3).round(3))

    st.markdown("---")

    # --- 3.2 선형 회귀 (Linear Regression) ---
    st.subheader("3.2 선형 회귀 (`LinearRegression`)")
    st.markdown("""
    선형 회귀는 특성과 타겟 간의 선형 관계를 가정하고, 이 관계를 가장 잘 나타내는 직선(또는 초평면)을 찾는 모델입니다.
    $y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + ... + \beta_n x_n$ 형태의 방정식을 학습합니다.
    - $\beta_0$: 절편 (intercept)
    - $\beta_1, ..., \beta_n$: 각 특성에 대한 계수 (coefficients)
    Scikit-learn의 `LinearRegression`을 사용합니다.
    """)
    code_linear_regression = """
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
# from utils_ml import display_regression_metrics (평가용)

# # 데이터 로드 및 전처리 (X_train_scaled_h, X_test_scaled_h, y_train_h, y_test_h 준비 가정)
# # ... (위의 전처리 코드와 동일) ...

# 1. 선형 회귀 모델 객체 생성
lr_model = LinearRegression()

# 2. 모델 학습 (스케일링된 훈련 데이터 사용)
lr_model.fit(X_train_scaled_h, y_train_h)

# 3. 테스트 데이터로 예측
y_pred_lr = lr_model.predict(X_test_scaled_h)

# 4. 모델 파라미터 확인
# print(f"절편 (Intercept): {lr_model.intercept_:.4f}")
# print("계수 (Coefficients):")
# for feature, coef in zip(X_housing.columns, lr_model.coef_):
#     print(f"  - {feature}: {coef:.4f}")

# 5. 모델 평가 (utils_ml의 display_regression_metrics 사용)
# display_regression_metrics(y_test_h, y_pred_lr, title="선형 회귀 모델 평가 결과")
    """
    st.code(code_linear_regression, language='python')

    if st.checkbox("선형 회귀 모델 실행 및 평가 보기", key="linear_regression_page_3"):
        st.markdown("#### 선형 회귀 모델 학습 및 예측")
        lr_model_ex = LinearRegression()
        lr_model_ex.fit(X_train_scaled_h, y_train_h)
        y_pred_lr_ex = lr_model_ex.predict(X_test_scaled_h)

        st.write(f"**절편 (Intercept, $\\beta_0$):** `{lr_model_ex.intercept_:.4f}`")
        st.write("**계수 (Coefficients, $\\beta_i$):**")
        coef_df = pd.DataFrame({'Feature': X_housing.columns, 'Coefficient': lr_model_ex.coef_}).round(4)
        st.dataframe(coef_df)
        st.caption("계수의 절대값이 클수록 해당 특성이 타겟 변수에 더 큰 영향을 미친다고 (선형적으로) 해석할 수 있습니다 (스케일링된 데이터 기준).")
        
        # 평가지표 표시
        display_regression_metrics(y_test_h, y_pred_lr_ex, title="선형 회귀 모델 평가 결과")

    st.markdown("---")

    # --- 3.3 기타 회귀 모델 (간략 소개) ---
    st.subheader("3.3 기타 주요 회귀 모델 (간략 소개)")
    st.markdown("""
    선형 회귀 외에도 다양한 회귀 알고리즘이 있으며, 데이터의 특성이나 문제 상황에 따라 더 적합한 모델을 선택할 수 있습니다.
    """)

    # --- Ridge 회귀 ---
    st.markdown("#### 릿지 회귀 (`Ridge`)")
    st.markdown("""
    선형 회귀에 L2 규제(regularization)를 추가한 모델입니다. 계수의 크기를 줄여 과적합을 방지하는 데 도움이 됩니다.
    `alpha` 파라미터로 규제 강도를 조절합니다 (클수록 규제가 강해짐).
    """)
    code_ridge = """
from sklearn.linear_model import Ridge

# Ridge 모델 객체 생성 (alpha 값은 튜닝 필요)
ridge_model = Ridge(alpha=1.0) 

# 모델 학습
# ridge_model.fit(X_train_scaled_h, y_train_h)

# 예측
# y_pred_ridge = ridge_model.predict(X_test_scaled_h)
# display_regression_metrics(y_test_h, y_pred_ridge, title="Ridge 회귀 모델 평가 결과 (alpha=1.0)")
    """
    st.code(code_ridge, language='python')
    if st.checkbox("Ridge 회귀 예시 실행 (alpha=1.0)", key="ridge_page_3"):
        ridge_model_ex = Ridge(alpha=1.0)
        ridge_model_ex.fit(X_train_scaled_h, y_train_h)
        y_pred_ridge_ex = ridge_model_ex.predict(X_test_scaled_h)
        display_regression_metrics(y_test_h, y_pred_ridge_ex, title="Ridge 회귀 (alpha=1.0) 평가 결과")


    # --- Lasso 회귀 ---
    st.markdown("#### 라쏘 회귀 (`Lasso`)")
    st.markdown("""
    선형 회귀에 L1 규제를 추가한 모델입니다. 일부 특성의 계수를 정확히 0으로 만들어 특성 선택(feature selection) 효과를 가집니다.
    `alpha` 파라미터로 규제 강도를 조절합니다.
    """)
    code_lasso = """
from sklearn.linear_model import Lasso

# Lasso 모델 객체 생성 (alpha 값은 튜닝 필요)
lasso_model = Lasso(alpha=0.1) # alpha가 너무 크면 많은 계수가 0이 될 수 있음

# 모델 학습
# lasso_model.fit(X_train_scaled_h, y_train_h)

# 예측
# y_pred_lasso = lasso_model.predict(X_test_scaled_h)
# display_regression_metrics(y_test_h, y_pred_lasso, title="Lasso 회귀 모델 평가 결과 (alpha=0.1)")
# print("Lasso 계수 중 0이 아닌 것의 개수:", np.sum(lasso_model.coef_ != 0))
    """
    st.code(code_lasso, language='python')
    if st.checkbox("Lasso 회귀 예시 실행 (alpha=0.01)", key="lasso_page_3"): # alpha 조정
        lasso_model_ex = Lasso(alpha=0.01) # alpha를 조금 작게 설정하여 너무 많은 계수가 0이 되는 것을 방지
        lasso_model_ex.fit(X_train_scaled_h, y_train_h)
        y_pred_lasso_ex = lasso_model_ex.predict(X_test_scaled_h)
        display_regression_metrics(y_test_h, y_pred_lasso_ex, title="Lasso 회귀 (alpha=0.01) 평가 결과")
        st.write(f"Lasso 모델 계수 (0이 아닌 것 개수: {np.sum(lasso_model_ex.coef_ != 0)} / 총 {len(lasso_model_ex.coef_)} 개):")
        lasso_coef_df = pd.DataFrame({'Feature': X_housing.columns, 'Coefficient': lasso_model_ex.coef_}).round(4)
        st.dataframe(lasso_coef_df[lasso_coef_df['Coefficient'] != 0]) # 0이 아닌 계수만 표시


    # --- 결정 트리 회귀 ---
    st.markdown("#### 결정 트리 회귀 (`DecisionTreeRegressor`)")
    st.markdown("""
    데이터를 트리 구조로 분할하며 예측을 수행합니다. 비선형 관계도 학습할 수 있지만, 과적합되기 쉬운 단점이 있습니다.
    주요 하이퍼파라미터: `max_depth` (트리 최대 깊이), `min_samples_split` (분할을 위한 최소 샘플 수), `min_samples_leaf` (리프 노드의 최소 샘플 수).
    """)
    code_dt_reg = """
from sklearn.tree import DecisionTreeRegressor

# Decision Tree Regressor 모델 객체 생성
dt_reg_model = DecisionTreeRegressor(max_depth=5, random_state=42)

# 모델 학습
# dt_reg_model.fit(X_train_scaled_h, y_train_h) # 스케일링은 트리기반 모델에 필수 아님

# 예측
# y_pred_dt_reg = dt_reg_model.predict(X_test_scaled_h)
# display_regression_metrics(y_test_h, y_pred_dt_reg, title="결정 트리 회귀 모델 평가 결과 (max_depth=5)")
    """
    st.code(code_dt_reg, language='python')
    if st.checkbox("결정 트리 회귀 예시 실행 (max_depth=5)", key="dt_reg_page_3"):
        dt_reg_model_ex = DecisionTreeRegressor(max_depth=5, random_state=42)
        # 트리 기반 모델은 스케일링이 필수는 아니지만, 일관성을 위해 스케일링된 데이터 사용
        dt_reg_model_ex.fit(X_train_scaled_h, y_train_h)
        y_pred_dt_reg_ex = dt_reg_model_ex.predict(X_test_scaled_h)
        display_regression_metrics(y_test_h, y_pred_dt_reg_ex, title="결정 트리 회귀 (max_depth=5) 평가 결과")


    # --- 랜덤 포레스트 회귀 ---
    st.markdown("#### 랜덤 포레스트 회귀 (`RandomForestRegressor`)")
    st.markdown("""
    여러 개의 결정 트리를 앙상블(ensemble)하여 예측 성능을 높이고 과적합을 줄인 모델입니다.
    주요 하이퍼파라미터: `n_estimators` (트리 개수), `max_depth`, `min_samples_split`, `min_samples_leaf`.
    """)
    code_rf_reg = """
from sklearn.ensemble import RandomForestRegressor

# Random Forest Regressor 모델 객체 생성
rf_reg_model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1) # n_jobs=-1: 모든 CPU 코어 사용

# 모델 학습
# rf_reg_model.fit(X_train_scaled_h, y_train_h)

# 예측
# y_pred_rf_reg = rf_reg_model.predict(X_test_scaled_h)
# display_regression_metrics(y_test_h, y_pred_rf_reg, title="랜덤 포레스트 회귀 모델 평가 결과 (n_estimators=100, max_depth=10)")
    """
    st.code(code_rf_reg, language='python')
    if st.checkbox("랜덤 포레스트 회귀 예시 실행 (n_estimators=50, max_depth=8)", key="rf_reg_page_3"): # 파라미터 약간 줄임
        st.caption("랜덤 포레스트는 학습에 다소 시간이 걸릴 수 있습니다...")
        rf_reg_model_ex = RandomForestRegressor(n_estimators=50, max_depth=8, random_state=42, n_jobs=-1)
        rf_reg_model_ex.fit(X_train_scaled_h, y_train_h)
        y_pred_rf_reg_ex = rf_reg_model_ex.predict(X_test_scaled_h)
        display_regression_metrics(y_test_h, y_pred_rf_reg_ex, title="랜덤 포레스트 회귀 (n_estimators=50, max_depth=8) 평가 결과")

    st.markdown("---")
    st.markdown("이 외에도 Support Vector Regressor (SVR), Gradient Boosting Regressor, XGBoost, LightGBM 등 다양한 고급 회귀 모델들이 있습니다. 각 모델의 특성을 이해하고 데이터에 적합한 모델을 선택하는 것이 중요합니다.")

else: # df_housing이 None일 경우 (데이터 로드 실패)
    st.error("데이터셋을 로드할 수 없어 회귀 모델 예제를 진행할 수 없습니다.")
    st.markdown("인터넷 연결을 확인하거나, Scikit-learn 데이터셋 서버가 정상적으로 동작하는지 확인해주세요. (`fetch_california_housing`)")