# pages/7_⚙️_모델_평가_및_개선.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, KFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression # 예제용 모델
from sklearn.tree import DecisionTreeClassifier # GridSearchCV 예제용 모델
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer # Pipeline 예제용
from utils.utils_ml import get_dataset, display_classification_metrics # 유틸리티 함수 사용

st.header("7. 모델 평가 및 개선")
st.markdown("""
머신러닝 모델을 구축한 후에는 그 성능을 객관적으로 평가하고, 더 나은 성능을 위해 모델을 개선하는 과정이 필수적입니다.
이 페이지에서는 교차 검증, 하이퍼파라미터 튜닝, 파이프라인 구축 등 모델 평가 및 개선에 사용되는 주요 기법들을 소개합니다.
""")

# --- 예제 데이터셋 로드 (분류용 데이터 사용) ---
st.subheader("모델 평가 및 개선 예제용 데이터셋 (유방암 진단 데이터)")
try:
    df_eval, target_names_eval, feature_names_eval = get_dataset("breast_cancer")
    X_eval = df_eval.drop('target', axis=1)
    y_eval = df_eval['target']
except Exception as e:
    st.error(f"유방암 데이터셋 로드 중 오류 발생: {e}")
    df_eval = None
    X_eval = None
    y_eval = None

if df_eval is not None and X_eval is not None:
    if st.checkbox("유방암 데이터셋 미리보기 (상위 5행)", key="show_eval_df_page_7"):
        st.dataframe(X_eval.head())
        st.write(f"데이터 형태 (특성): {X_eval.shape}")
        st.write(f"타겟 클래스 이름: {target_names_eval}")

    # --- 데이터 전처리 (스케일링 및 분할) ---
    # 이 페이지의 모든 예제에서 사용할 공통 전처리된 데이터
    scaler_eval = StandardScaler()
    X_eval_scaled = scaler_eval.fit_transform(X_eval)

    # 훈련/테스트 데이터 분할은 각 예제 섹션 내에서 필요에 따라 수행하거나,
    # 여기서는 교차검증 등을 위해 전체 X_eval_scaled, y_eval을 주로 사용합니다.
    # GridSearchCV 예제에서는 내부적으로 분할되거나, 별도 분할된 데이터를 사용합니다.

    st.markdown("---")

    # --- 7.1 교차 검증 (Cross-Validation) ---
    st.subheader("7.1 교차 검증 (Cross-Validation)")
    st.markdown("""
    모델의 일반화 성능을 보다 안정적으로 평가하기 위한 방법입니다.
    훈련 데이터를 여러 개의 작은 부분집합(fold)으로 나누어, 각 부분집합이 한 번씩 테스트 세트 역할을 하고 나머지 부분집합들이 훈련 세트 역할을 하도록 모델을 여러 번 학습하고 평가합니다.
    - **K-Fold 교차 검증 (`KFold`, `cross_val_score`):** 데이터를 K개의 폴드로 나누어 K번 검증을 수행합니다.
        - `cross_val_score(estimator, X, y, cv, scoring)`: 교차 검증을 수행하고 각 폴드에서의 평가 점수를 반환합니다.
    """)
    code_cross_validation = """
from sklearn.model_selection import cross_val_score, KFold
from sklearn.linear_model import LogisticRegression
# from sklearn.preprocessing import StandardScaler (X_eval_scaled가 이미 준비되었다고 가정)
# import numpy as np

# # 데이터 준비 (X_eval_scaled, y_eval 사용 가정)

# 1. 모델 객체 생성
log_reg_cv_model = LogisticRegression(solver='liblinear', random_state=42, max_iter=200)

# 2. K-Fold 교차 검증 설정
# n_splits: 폴드 개수
# shuffle=True: 데이터를 섞음 (클래스 분포가 불균형할 때 유용, stratify와 함께 사용 권장)
# random_state: shuffle=True일 때 재현성을 위한 시드
kfold = KFold(n_splits=5, shuffle=True, random_state=42)

# 3. cross_val_score 수행
# scoring: 평가 지표 (예: 'accuracy', 'f1', 'roc_auc', 'neg_mean_squared_error' 등)
# cv: 교차 검증 분할기 객체 또는 폴드 수(정수)
scores = cross_val_score(log_reg_cv_model, X_eval_scaled, y_eval, cv=kfold, scoring='accuracy')

# print(f"각 폴드별 정확도: {np.round(scores, 4)}")
# print(f"평균 정확도: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})") # 평균 +/- 2*표준편차
    """
    st.code(code_cross_validation, language='python')

    if st.checkbox("교차 검증 (`cross_val_score`) 예시 실행", key="cross_val_page_7"):
        st.markdown("#### 로지스틱 회귀 모델 K-Fold 교차 검증 (K=5)")
        log_reg_cv_model_ex = LogisticRegression(solver='liblinear', random_state=42, max_iter=200)
        kfold_ex = KFold(n_splits=5, shuffle=True, random_state=42)
        
        # cross_val_score는 내부적으로 모델을 복제하여 각 폴드마다 새로 학습시킴
        cv_scores = cross_val_score(log_reg_cv_model_ex, X_eval_scaled, y_eval, cv=kfold_ex, scoring='accuracy')
        
        st.write(f"**각 폴드별 정확도:** `{np.round(cv_scores, 4)}`")
        st.metric(label="평균 정확도", value=f"{cv_scores.mean():.4f}")
        st.caption(f"(표준편차: {cv_scores.std():.4f}, 95% 신뢰구간 근사: +/- {cv_scores.std() * 2:.4f})")
        st.markdown("교차 검증은 모델이 특정 데이터 분할에만 과도하게 적합되는 것을 방지하고, 일반화 성능을 더 신뢰성 있게 측정하는 데 도움을 줍니다.")

    st.markdown("---")

    # --- 7.2 하이퍼파라미터 튜닝 (Hyperparameter Tuning) ---
    st.subheader("7.2 하이퍼파라미터 튜닝")
    st.markdown("""
    머신러닝 모델의 성능은 사용자가 직접 설정하는 **하이퍼파라미터**에 따라 크게 달라질 수 있습니다.
    최적의 하이퍼파라미터 조합을 찾는 과정을 하이퍼파라미터 튜닝이라고 합니다.
    - **`GridSearchCV`:** 사용자가 지정한 하이퍼파라미터 값들의 모든 조합에 대해 교차 검증을 수행하여 최적의 조합을 찾습니다.
    - **`RandomizedSearchCV`:** 지정된 범위 또는 분포에서 하이퍼파라미터 값을 무작위로 샘플링하여 교차 검증을 수행합니다. 탐색 공간이 넓을 때 유용합니다.
    """)
    code_grid_search = """
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.tree import DecisionTreeClassifier # 예제 모델
# from sklearn.preprocessing import StandardScaler (X_eval_scaled, y_eval 사용 가정)

# # 예시를 위해 훈련/검증 데이터 분리 (GridSearchCV는 내부적으로 교차검증하지만, 최종 평가용 테스트셋은 별도)
# X_train_gs, X_test_gs, y_train_gs, y_test_gs = train_test_split(
#     X_eval_scaled, y_eval, test_size=0.2, random_state=42, stratify=y_eval
# )

# 1. 기본 모델 객체 생성
dt_model_gs = DecisionTreeClassifier(random_state=42)

# 2. 탐색할 하이퍼파라미터 그리드 정의
param_grid = {
    'max_depth': [3, 5, 7, None], # None은 제한 없음
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'criterion': ['gini', 'entropy']
}

# 3. GridSearchCV 객체 생성
# estimator: 튜닝할 모델
# param_grid: 탐색할 하이퍼파라미터 그리드
# cv: 교차 검증 폴드 수
# scoring: 평가 지표
# n_jobs=-1: 모든 CPU 코어 사용
grid_search = GridSearchCV(
    estimator=dt_model_gs,
    param_grid=param_grid,
    cv=3, # 교차 검증 폴드 수 (예시에서는 간단히 3)
    scoring='accuracy',
    n_jobs=-1,
    verbose=1 # 진행 상황 출력 레벨
)

# 4. GridSearchCV 학습 (최적 하이퍼파라미터 탐색)
# grid_search.fit(X_train_gs, y_train_gs)

# 5. 최적 결과 확인
# print(f"최적 하이퍼파라미터: {grid_search.best_params_}")
# print(f"최적 교차 검증 정확도: {grid_search.best_score_:.4f}")

# # 6. 최적 모델로 테스트 데이터 예측 및 평가
# best_model = grid_search.best_estimator_
# y_pred_gs = best_model.predict(X_test_gs)
# # (display_classification_metrics 등으로 평가)
    """
    st.code(code_grid_search, language='python')

    if st.checkbox("`GridSearchCV` 예시 실행 (결정 트리)", key="grid_search_page_7"):
        st.markdown("#### 결정 트리 모델 `GridSearchCV` (간단한 파라미터 그리드)")
        st.caption("`GridSearchCV`는 탐색할 조합이 많으면 시간이 오래 걸릴 수 있습니다. 예제에서는 파라미터 범위를 줄였습니다.")

        # GridSearchCV는 학습 시간이 걸리므로, 예시에서는 파라미터 범위를 축소
        param_grid_ex = {
            'max_depth': [3, 5, None],
            'min_samples_split': [2, 10],
            'criterion': ['gini'] # criterion 하나로 고정하여 조합 수 줄임
        }
        
        # 훈련/검증 데이터 분리 (최종 평가용 테스트셋은 여기서 사용 안 함)
        # GridSearchCV는 내부적으로 훈련 데이터를 다시 교차검증 폴드로 나눔
        X_train_gs_ex, _, y_train_gs_ex, _ = train_test_split(
             X_eval_scaled, y_eval, test_size=0.2, random_state=42, stratify=y_eval
        )


        dt_model_gs_ex = DecisionTreeClassifier(random_state=42)
        grid_search_ex = GridSearchCV(
            estimator=dt_model_gs_ex,
            param_grid=param_grid_ex,
            cv=2, # 폴드 수도 줄임
            scoring='accuracy',
            n_jobs=-1
        )
        
        with st.spinner("`GridSearchCV` 실행 중... (몇 초에서 몇 분 소요될 수 있습니다)"):
            grid_search_ex.fit(X_train_gs_ex, y_train_gs_ex)

        st.success("`GridSearchCV` 실행 완료!")
        st.write("**최적 하이퍼파라미터:**")
        st.json(grid_search_ex.best_params_)
        st.metric(label="최적 교차 검증 정확도 (훈련 데이터 내에서)", value=f"{grid_search_ex.best_score_:.4f}")

        st.write("`GridSearchCV`가 찾은 최적 모델 (`best_estimator_`):")
        st.text(grid_search_ex.best_estimator_)
        st.markdown("이 `best_estimator_`를 사용하여 **별도의 테스트 데이터셋**에 대한 최종 성능을 평가해야 합니다.")


    st.markdown("---")

    # --- 7.3 파이프라인 (Pipeline) ---
    st.subheader("7.3 파이프라인 (`Pipeline`)")
    st.markdown("""
    데이터 전처리 단계와 모델 학습 단계를 하나로 연결하여 코드의 간결성과 재현성을 높입니다.
    `Pipeline` 객체는 여러 변환기(transformer)와 마지막에 추정기(estimator)를 순차적으로 실행합니다.
    - 교차 검증이나 하이퍼파라미터 튜닝 시 데이터 유출(data leakage)을 방지하는 데 매우 유용합니다.
    """)
    code_pipeline = """
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer # 결측치 처리기 추가
# from sklearn.model_selection import train_test_split (X_train_pipe, ... 준비 가정)

# # 데이터 준비 (결측치가 있을 수 있는 원본 X_eval, y_eval 사용)
# X_train_pipe, X_test_pipe, y_train_pipe, y_test_pipe = train_test_split(
#     X_eval, y_eval, test_size=0.2, random_state=42, stratify=y_eval
# )


# 1. 파이프라인 단계 정의 (리스트 형태로 (이름, 객체) 튜플 전달)
pipeline_steps = [
    ('imputer', SimpleImputer(strategy='median')), # 1. 결측치 중앙값으로 대치
    ('scaler', StandardScaler()),                  # 2. 표준화 스케일링
    ('classifier', LogisticRegression(solver='liblinear', random_state=42)) # 3. 로지스틱 회귀 모델
]

# 2. Pipeline 객체 생성
model_pipeline = Pipeline(pipeline_steps)

# 3. 파이프라인 학습 (전체 데이터 또는 훈련 데이터에 fit)
# model_pipeline.fit(X_train_pipe, y_train_pipe)
# # 내부적으로 X_train_pipe가 imputer.fit_transform -> scaler.fit_transform -> classifier.fit 순으로 전달됨

# 4. 파이프라인으로 예측 (테스트 데이터에 predict)
# y_pred_pipeline = model_pipeline.predict(X_test_pipe)
# # 내부적으로 X_test_pipe가 imputer.transform -> scaler.transform -> classifier.predict 순으로 전달됨

# 5. 파이프라인 평가
# # (display_classification_metrics 등으로 평가)

# # GridSearchCV와 함께 사용 시 파이프라인 단계의 하이퍼파라미터 지정:
# # param_grid_pipe = {
# #     'classifier__C': [0.1, 1.0, 10.0], # '추정기이름__파라미터이름' 형태
# #     'imputer__strategy': ['mean', 'median']
# # }
# # grid_search_pipe = GridSearchCV(model_pipeline, param_grid_pipe, cv=3, scoring='accuracy')
# # grid_search_pipe.fit(X_train_pipe, y_train_pipe)
    """
    st.code(code_pipeline, language='python')

    if st.checkbox("`Pipeline` 예시 실행", key="pipeline_page_7"):
        st.markdown("#### `Pipeline`을 사용한 전처리 및 모델 학습")
        # 파이프라인 예시에서는 원본 X_eval을 사용하여 전처리부터 시작
        X_train_p, X_test_p, y_train_p, y_test_p = train_test_split(
            X_eval, y_eval, test_size=0.2, random_state=42, stratify=y_eval
        )

        # 1. 파이프라인 단계 정의
        pipeline_steps_ex = [
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
            ('classifier', LogisticRegression(solver='liblinear', random_state=42, C=0.1, max_iter=200))
        ]
        # 2. Pipeline 객체 생성
        model_pipeline_ex = Pipeline(pipeline_steps_ex)

        # 3. 파이프라인 학습
        with st.spinner("파이프라인 학습 중..."):
            model_pipeline_ex.fit(X_train_p, y_train_p)
        st.success("파이프라인 학습 완료!")

        # 4. 파이프라인으로 예측 및 평가
        y_pred_pipeline_ex = model_pipeline_ex.predict(X_test_p)
        display_classification_metrics(y_test_p, y_pred_pipeline_ex, target_names=target_names_eval, title="파이프라인 모델 평가 결과")
        
        st.markdown("##### 파이프라인 내 각 단계의 파라미터 확인 (예시)")
        st.write("Imputer 학습 결과 (중앙값):")
        # 파이프라인 단계 접근: model_pipeline_ex.named_steps['imputer'] 또는 model_pipeline_ex['imputer']
        st.text(model_pipeline_ex.named_steps['imputer'].statistics_)
        
        st.write("Scaler 학습 결과 (평균):")
        st.text(model_pipeline_ex.named_steps['scaler'].mean_)


    st.markdown("---")
    st.markdown("""
    모델 평가는 한 번으로 끝나는 것이 아니라, 다양한 지표와 방법을 통해 모델의 강점과 약점을 파악하고, 지속적으로 개선해나가는 반복적인 과정입니다.
    이 외에도 모델 저장 및 불러오기 (`joblib` 또는 `pickle` 사용), 고급 앙상블 기법, 최신 AutoML 도구 등 머신러닝 모델의 성능을 높이고 운영을 효율화하는 다양한 방법들이 있습니다.
    """)

else: # df_eval이 None일 경우 (데이터 로드 실패)
    st.error(f"{chosen_dataset_pca} 데이터셋을 로드할 수 없어 모델 평가/개선 예제를 진행할 수 없습니다.")