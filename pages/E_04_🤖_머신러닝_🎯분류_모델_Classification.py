# pages/4_🎯_분류_모델_Classification.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.impute import SimpleImputer # 데이터셋에 따라 결측치 처리용
from utils.utils_ml import get_dataset, display_classification_metrics # 유틸리티 함수 사용

st.header("4. 분류 모델 (Classification Models)")
st.markdown("""
분류는 미리 정의된 여러 범주(클래스 또는 레이블) 중 하나로 입력 데이터를 할당하는 지도 학습 기법입니다.
예를 들어, 이메일이 스팸인지 아닌지, 이미지가 고양이인지 강아지인지, 또는 고객의 이탈 여부 등을 예측하는 데 사용됩니다.
""")

# --- 예제 데이터셋 로드 (와인 데이터 또는 유방암 데이터) ---
st.subheader("와인 또는 유방암 데이터셋을 사용한 분류 예제")
dataset_options = ["wine", "breast_cancer", "iris"] # iris 추가
chosen_dataset_cls = st.selectbox(
    "분류 예제에 사용할 데이터셋을 선택하세요:",
    dataset_options,
    key="cls_dataset_selector"
)

try:
    df_cls, target_names_cls, feature_names_cls = get_dataset(chosen_dataset_cls)
except Exception as e:
    st.error(f"{chosen_dataset_cls} 데이터셋 로드 중 오류 발생: {e}")
    st.info("Scikit-learn 데이터셋 서버에 일시적인 문제가 있을 수 있습니다. 나중에 다시 시도해주세요.")
    df_cls = None

if df_cls is not None:
    if st.checkbox(f"{chosen_dataset_cls} 데이터셋 미리보기 (상위 5행)", key=f"show_{chosen_dataset_cls}_df_page_4"):
        st.dataframe(df_cls.head())
        st.write(f"데이터 형태: {df_cls.shape}")
        st.write("특성 (Features):", feature_names_cls)
        st.write("타겟 (Target) 클래스 이름:", target_names_cls if target_names_cls is not None else "단일 타겟 (회귀) 또는 이름 없음")
        st.write("타겟 변수 고유값:", df_cls['target'].unique())
        st.write("결측치 확인:")
        st.dataframe(df_cls.isnull().sum().rename("결측치 수"))


    st.markdown("---")

    # --- 4.1 데이터 전처리 (분류용) ---
    st.subheader("4.1 데이터 전처리 (분류용)")
    st.markdown("""
    분류 모델 학습 전, 결측치 처리, 특성 스케일링, 데이터 분할 등의 전처리 과정이 필요할 수 있습니다.
    타겟 변수가 문자열인 경우, `LabelEncoder` 등을 사용하여 숫자형으로 변환해야 합니다.
    """)

    # 특성(X)과 타겟(y) 분리
    X_cls = df_cls.drop('target', axis=1)
    y_cls = df_cls['target']

    # (참고) 만약 y_cls가 문자열 레이블이라면 LabelEncoder 사용
    # le = LabelEncoder()
    # y_cls_encoded = le.fit_transform(y_cls)
    # target_names_cls = le.classes_ # 인코딩된 클래스 이름 저장

    # 훈련 데이터와 테스트 데이터 분리
    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
        X_cls, y_cls,
        test_size=0.3,     # 테스트 데이터 비율 30%
        random_state=42,   # 결과 재현을 위한 시드
        stratify=y_cls     # y의 클래스 비율을 유지하며 분할
    )

    # 특성 스케일링 (StandardScaler 사용)
    scaler_cls = StandardScaler()
    X_train_scaled_c = scaler_cls.fit_transform(X_train_c)
    X_test_scaled_c = scaler_cls.transform(X_test_c)

    if st.checkbox("전처리된 분류 데이터 일부 확인", key="show_preprocessed_cls_data_page_4"):
        st.write("훈련용 특성 데이터 (스케일링 후, 상위 3행):")
        st.dataframe(pd.DataFrame(X_train_scaled_c, columns=X_cls.columns).head(3).round(3))
        st.write("테스트용 타겟 데이터 (상위 10개):", y_test_c.head(10).values)


    st.markdown("---")

    # --- 4.2 로지스틱 회귀 (Logistic Regression) ---
    st.subheader("4.2 로지스틱 회귀 (`LogisticRegression`)")
    st.markdown("""
    로지스틱 회귀는 이름에 '회귀'가 들어가지만, 실제로는 **분류** 알고리즘입니다.
    시그모이드(Sigmoid) 함수를 사용하여 각 클래스에 속할 확률을 예측하고, 이를 바탕으로 분류합니다.
    이진 분류에 주로 사용되지만, 다중 클래스 분류로 확장될 수 있습니다 (예: OvR - One-vs-Rest 방식).
    선형 모델의 일종으로 해석이 용이하고, 비교적 학습 속도가 빠릅니다.
    주요 하이퍼파라미터: `C` (규제 강도의 역수, 작을수록 강한 규제), `penalty` (`'l1'`, `'l2'`, `'elasticnet'`).
    """)
    code_logistic_regression = """
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
# from utils_ml import display_classification_metrics (평가용)

# # 데이터 로드 및 전처리 (X_train_scaled_c, X_test_scaled_c, y_train_c, y_test_c 준비 가정)
# # ... (위의 전처리 코드와 동일) ...

# 1. 로지스틱 회귀 모델 객체 생성
log_reg_model = LogisticRegression(solver='liblinear', random_state=42) # solver는 데이터셋 크기나 penalty에 따라 선택

# 2. 모델 학습
log_reg_model.fit(X_train_scaled_c, y_train_c)

# 3. 테스트 데이터로 예측
y_pred_log_reg = log_reg_model.predict(X_test_scaled_c)
# y_pred_proba_log_reg = log_reg_model.predict_proba(X_test_scaled_c) # 각 클래스별 예측 확률

# 4. 모델 평가
# display_classification_metrics(y_test_c, y_pred_log_reg, target_names=target_names_cls, title="로지스틱 회귀 모델 평가")
    """
    st.code(code_logistic_regression, language='python')

    if st.checkbox("로지스틱 회귀 모델 실행 및 평가 보기", key="logistic_regression_page_4"):
        st.markdown("#### 로지스틱 회귀 모델 학습 및 예측")
        # solver='liblinear'는 작은 데이터셋에 적합하고 L1, L2 규제 모두 지원
        # 다중 클래스 경우 solver='lbfgs'(기본값) 또는 'saga' 등이 사용됨
        solver_option = 'liblinear' if len(df_cls['target'].unique()) == 2 else 'lbfgs'

        log_reg_model_ex = LogisticRegression(solver=solver_option, random_state=42, max_iter=200) # max_iter 증가
        log_reg_model_ex.fit(X_train_scaled_c, y_train_c)
        y_pred_log_reg_ex = log_reg_model_ex.predict(X_test_scaled_c)
        
        display_classification_metrics(y_test_c, y_pred_log_reg_ex, target_names=target_names_cls, title="로지스틱 회귀 모델 평가 결과")
        
        if hasattr(log_reg_model_ex, 'coef_'):
            st.write("**모델 계수 (Coefficients):**")
            # 이진 분류의 경우 coef_는 (1, n_features), 다중 클래스는 (n_classes, n_features)
            if log_reg_model_ex.coef_.shape[0] == 1: # 이진 분류
                 coef_df_log = pd.DataFrame({'Feature': X_cls.columns, 'Coefficient': log_reg_model_ex.coef_[0]}).round(4)
            else: # 다중 클래스
                coef_df_log = pd.DataFrame(log_reg_model_ex.coef_.T, index=X_cls.columns, columns=[f'Class_{i}' for i in log_reg_model_ex.classes_]).round(4)
            st.dataframe(coef_df_log)

    st.markdown("---")

    # --- 4.3 K-최근접 이웃 (K-Nearest Neighbors, KNN) ---
    st.subheader("4.3 K-최근접 이웃 (`KNeighborsClassifier`)")
    st.markdown("""
    KNN은 새로운 데이터 포인트가 주어졌을 때, 가장 가까운 `K`개의 훈련 데이터 포인트를 찾아 다수결 원칙에 따라 클래스를 예측하는 비모수적(non-parametric) 알고리즘입니다.
    - 단순하고 직관적이지만, 데이터가 많을 경우 예측 속도가 느릴 수 있고, 특성 스케일링에 민감합니다.
    - `K` 값 선택이 중요하며, 너무 작으면 노이즈에 민감하고, 너무 크면 경계가 모호해질 수 있습니다.
    주요 하이퍼파라미터: `n_neighbors` (K 값), `weights` (`'uniform'`, `'distance'`), `metric` (거리 측정 방식, 예: `'minkowski'`, `'euclidean'`).
    """)
    code_knn = """
from sklearn.neighbors import KNeighborsClassifier

# # 데이터 로드 및 전처리 (X_train_scaled_c, X_test_scaled_c, y_train_c, y_test_c 준비 가정)

# 1. KNN 모델 객체 생성
knn_model = KNeighborsClassifier(n_neighbors=5) # K=5로 설정

# 2. 모델 학습 (KNN은 실제로 학습 단계에서 많은 계산을 하지 않고 데이터를 저장)
knn_model.fit(X_train_scaled_c, y_train_c)

# 3. 테스트 데이터로 예측
y_pred_knn = knn_model.predict(X_test_scaled_c)

# 4. 모델 평가
# display_classification_metrics(y_test_c, y_pred_knn, target_names=target_names_cls, title="KNN 분류 모델 평가 (K=5)")
    """
    st.code(code_knn, language='python')

    if st.checkbox("KNN 분류 모델 실행 및 평가 보기 (K=5)", key="knn_page_4"):
        st.markdown("#### KNN 분류 모델 학습 및 예측 (K=5)")
        knn_model_ex = KNeighborsClassifier(n_neighbors=5)
        knn_model_ex.fit(X_train_scaled_c, y_train_c)
        y_pred_knn_ex = knn_model_ex.predict(X_test_scaled_c)
        display_classification_metrics(y_test_c, y_pred_knn_ex, target_names=target_names_cls, title="KNN 분류 모델 평가 결과 (K=5)")

    st.markdown("---")

    # --- 4.4 결정 트리 (Decision Tree Classifier) ---
    st.subheader("4.4 결정 트리 (`DecisionTreeClassifier`)")
    st.markdown("""
    결정 트리는 데이터의 특성들을 기반으로 스무고개와 같은 질문을 연속적으로 던져 데이터를 분할하고, 각 최종 분할 영역(리프 노드)에 해당하는 클래스를 예측하는 모델입니다.
    - 해석이 용이하고 시각화가 가능하지만, 과적합되기 쉬운 경향이 있습니다.
    - 특성 스케일링이 필수는 아닙니다.
    주요 하이퍼파라미터: `criterion` (`'gini'`, `'entropy'`), `max_depth`, `min_samples_split`, `min_samples_leaf`.
    """)
    code_dt_cls = """
from sklearn.tree import DecisionTreeClassifier

# # 데이터 로드 및 전처리 (X_train_c, X_test_c, y_train_c, y_test_c 준비 가정 - 스케일링 안된 데이터도 사용 가능)

# 1. 결정 트리 모델 객체 생성
dt_cls_model = DecisionTreeClassifier(max_depth=4, random_state=42)

# 2. 모델 학습
dt_cls_model.fit(X_train_c, y_train_c) # 스케일링 안된 원본 X_train_c 사용 가능

# 3. 테스트 데이터로 예측
y_pred_dt_cls = dt_cls_model.predict(X_test_c)

# 4. 모델 평가
# display_classification_metrics(y_test_c, y_pred_dt_cls, target_names=target_names_cls, title="결정 트리 분류 모델 평가 (max_depth=4)")

# 특성 중요도 확인
# feature_importances = dt_cls_model.feature_importances_
# importance_df = pd.DataFrame({'Feature': X_cls.columns, 'Importance': feature_importances})
# importance_df = importance_df.sort_values(by='Importance', ascending=False)
# print("\\n특성 중요도:\\n", importance_df)
    """
    st.code(code_dt_cls, language='python')

    if st.checkbox("결정 트리 분류 모델 실행 및 평가 보기 (max_depth=4)", key="dt_cls_page_4"):
        st.markdown("#### 결정 트리 분류 모델 학습 및 예측 (max_depth=4)")
        dt_cls_model_ex = DecisionTreeClassifier(max_depth=4, random_state=42)
        # 트리 모델은 스케일링에 영향을 받지 않으므로, 스케일링 안 된 X_train_c 사용 가능
        dt_cls_model_ex.fit(X_train_c, y_train_c)
        y_pred_dt_cls_ex = dt_cls_model_ex.predict(X_test_c)
        display_classification_metrics(y_test_c, y_pred_dt_cls_ex, target_names=target_names_cls, title="결정 트리 분류 모델 평가 결과 (max_depth=4)")

        st.write("**특성 중요도 (Feature Importances):**")
        importances = dt_cls_model_ex.feature_importances_
        importance_df = pd.DataFrame({'Feature': X_cls.columns, 'Importance': importances})
        importance_df = importance_df.sort_values(by='Importance', ascending=False).reset_index(drop=True)
        st.dataframe(importance_df.round(4))
        st.caption("특성 중요도는 해당 특성이 모델의 예측에 얼마나 기여하는지를 나타냅니다.")

    st.markdown("---")
    st.markdown("""
    이 외에도 **랜덤 포레스트(`RandomForestClassifier`)**, **서포트 벡터 머신(`SVC`)**, **나이브 베이즈(`GaussianNB` 등)**, **그래디언트 부스팅 계열(`GradientBoostingClassifier`, `XGBClassifier`, `LGBMClassifier`)** 등 다양한 강력한 분류 알고리즘들이 있습니다.
    각 모델의 장단점과 하이퍼파라미터를 이해하고, 교차 검증 등을 통해 문제에 가장 적합한 모델을 선택하고 튜닝하는 것이 중요합니다.
    """)

else: # df_cls가 None일 경우 (데이터 로드 실패)
    st.error(f"{chosen_dataset_cls} 데이터셋을 로드할 수 없어 분류 모델 예제를 진행할 수 없습니다.")
    st.markdown("인터넷 연결을 확인하거나, Scikit-learn 데이터셋 서버가 정상적으로 동작하는지 확인해주세요.")