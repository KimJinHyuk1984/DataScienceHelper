# utils_ml.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris, fetch_california_housing, load_wine, load_breast_cancer, make_blobs
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns

@st.cache_data # 데이터 로딩은 캐싱
def get_dataset(name="iris"):
    """
    Scikit-learn 내장 데이터셋을 로드하여 DataFrame으로 반환합니다.
    Args:
        name (str): 로드할 데이터셋 이름 ("iris", "california_housing", "wine", "breast_cancer", "blobs_for_clustering")
    Returns:
        tuple: (pd.DataFrame, target_names or None, feature_names)
               target_names는 분류 데이터셋의 클래스 이름입니다.
    """
    if name == "iris":
        data = load_iris()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df['target'] = data.target
        return df, data.target_names, data.feature_names
    elif name == "california_housing":
        data = fetch_california_housing()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df['target'] = data.target
        return df, None, data.feature_names
    elif name == "wine":
        data = load_wine()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df['target'] = data.target
        return df, data.target_names, data.feature_names
    elif name == "breast_cancer":
        data = load_breast_cancer()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df['target'] = data.target
        return df, data.target_names, data.feature_names
    elif name == "blobs_for_clustering":
        # 군집화 예제용 데이터 생성
        X, y = make_blobs(n_samples=300, centers=4, cluster_std=0.8, random_state=42)
        df = pd.DataFrame(X, columns=['feature1', 'feature2'])
        df['true_label'] = y # 실제 레이블 (군집화에서는 사용하지 않지만, 비교용으로 포함)
        return df, None, ['feature1', 'feature2'] # target_names 없음
    return None, None, None

def display_regression_metrics(y_true, y_pred, title="📈 회귀 모델 평가 지표"):
    """회귀 모델의 주요 평가지표를 Streamlit에 표시합니다."""
    st.subheader(title)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric(label="Mean Squared Error (MSE)", value=f"{mse:.4f}")
    with col2:
        st.metric(label="Root Mean Squared Error (RMSE)", value=f"{rmse:.4f}")
    st.metric(label="R-squared (R²)", value=f"{r2:.4f}")
    
    # 예측값과 실제값 비교 시각화 (샘플)
    fig, ax = plt.subplots(figsize=(8,6))
    ax.scatter(y_true, y_pred, alpha=0.5, edgecolors='w', s=50)
    ax.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'k--', lw=2) # y=x 직선
    ax.set_xlabel("실제 값 (Actual Values)")
    ax.set_ylabel("예측 값 (Predicted Values)")
    ax.set_title("실제 값 vs. 예측 값 비교")
    st.pyplot(fig)
    plt.clf() # 이전 그림 지우기
    st.markdown("---")

def display_classification_metrics(y_true, y_pred, target_names=None, title="🎯 분류 모델 평가 지표"):
    """분류 모델의 주요 평가지표와 Confusion Matrix를 Streamlit에 표시합니다."""
    st.subheader(title)
    accuracy = accuracy_score(y_true, y_pred)
    # 다중 클래스일 경우 'weighted' 평균이 적절할 수 있음
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)

    col1, col2 = st.columns(2)
    with col1:
        st.metric(label="정확도 (Accuracy)", value=f"{accuracy:.4f}")
        st.metric(label="정밀도 (Precision - Weighted)", value=f"{precision:.4f}")
    with col2:
        st.metric(label="재현율 (Recall - Weighted)", value=f"{recall:.4f}")
        st.metric(label="F1 점수 (F1-Score - Weighted)", value=f"{f1:.4f}")

    st.markdown("#### 혼동 행렬 (Confusion Matrix):")
    cm = confusion_matrix(y_true, y_pred)
    
    # 클래스 레이블이 주어지면 사용, 아니면 기본 숫자 사용
    if target_names is not None and len(target_names) == cm.shape[0]:
        cm_df = pd.DataFrame(cm, index=target_names, columns=target_names)
    else:
        cm_df = pd.DataFrame(cm)
    
    fig, ax = plt.subplots(figsize=(cm.shape[1]*1.5, cm.shape[0]*1.2)) # 크기 동적 조절
    sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False, annot_kws={"size": 14})
    ax.set_xlabel('예측된 레이블 (Predicted Label)', fontsize=12)
    ax.set_ylabel('실제 레이블 (True Label)', fontsize=12)
    ax.set_title('Confusion Matrix', fontsize=15)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    st.pyplot(fig)
    plt.clf() # 이전 그림 지우기
    st.markdown("---")

def display_clustering_metrics(X, labels, title="🔗 군집화 모델 평가 지표 (Silhouette Score)"):
    """군집화 모델의 실루엣 점수를 Streamlit에 표시합니다. (실제 레이블 없이 평가)"""
    st.subheader(title)
    # 레이블이 모두 동일하거나(-1 등) 하나만 있는 경우 실루엣 점수 계산 불가
    if len(np.unique(labels)) > 1 :
        try:
            score = silhouette_score(X, labels)
            st.metric(label="실루엣 계수 (Silhouette Coefficient)", value=f"{score:.4f}")
            st.caption("""
            실루엣 계수는 -1에서 1 사이의 값을 가집니다.
            - 1에 가까울수록: 군집들이 잘 분리되어 있고, 군집 내 데이터들이 조밀하게 모여있음을 의미합니다.
            - 0에 가까울수록: 군집들이 서로 겹쳐있음을 의미합니다.
            - 음수 값은: 데이터 포인트가 잘못된 군집에 할당되었을 가능성을 시사합니다.
            """)
        except ValueError as e:
            st.warning(f"실루엣 점수 계산 중 오류: {e}. (예: 단일 군집)")
    else:
        st.warning("군집이 하나만 형성되어 실루엣 점수를 계산할 수 없습니다.")
    st.markdown("---")