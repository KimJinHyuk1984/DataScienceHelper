# pages/5_🔗_군집화_Clustering.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from utils.utils_ml import get_dataset, display_clustering_metrics # 유틸리티 함수 사용
import seaborn as sns

st.header("5. 군집화 (Clustering)")
st.markdown("""
군집화는 비지도 학습의 한 종류로, 레이블(정답)이 없는 데이터 내에서 유사한 특성을 가진 데이터 포인트들을 그룹(군집 또는 클러스터)으로 묶는 과정입니다.
데이터의 숨겨진 구조를 발견하거나, 고객 세분화, 이상치 탐지 등 다양한 분야에 활용됩니다.
""")

# --- 예제 데이터셋 로드 (군집화용 가상 데이터) ---
st.subheader("군집화 예제용 데이터셋 (`make_blobs`)")
try:
    # utils_ml.get_dataset에서 'true_label' 컬럼도 반환하지만, 군집화는 비지도 학습이므로 사용하지 않음
    df_blobs, _, feature_names_blobs = get_dataset("blobs_for_clustering")
    X_blobs = df_blobs[['feature1', 'feature2']].copy() # 특성만 사용
except Exception as e:
    st.error(f"군집화용 데이터셋 로드 중 오류 발생: {e}")
    df_blobs = None
    X_blobs = None

if df_blobs is not None and X_blobs is not None:
    if st.checkbox("군집화용 데이터셋 시각화 (2D Scatter Plot)", key="show_blobs_df_page_5"):
        fig, ax = plt.subplots()
        ax.scatter(X_blobs['feature1'], X_blobs['feature2'], alpha=0.7, edgecolors='w')
        ax.set_xlabel("Feature 1")
        ax.set_ylabel("Feature 2")
        ax.set_title("군집화 전 원본 데이터 분포")
        st.pyplot(fig)
        plt.clf()
        st.write(f"데이터 형태: {X_blobs.shape}")
        st.write("특성 (Features):", feature_names_blobs)

    st.markdown("---")

    # --- 5.1 데이터 전처리 (군집화용) ---
    st.subheader("5.1 데이터 전처리 (군집화용)")
    st.markdown("""
    K-평균과 같은 거리 기반 군집화 알고리즘은 특성 스케일링에 민감합니다. 모든 특성이 군집화 과정에 동등하게 기여하도록 스케일링을 적용하는 것이 좋습니다.
    """)

    scaler_cluster = StandardScaler()
    X_blobs_scaled = scaler_cluster.fit_transform(X_blobs)
    
    df_blobs_scaled = pd.DataFrame(X_blobs_scaled, columns=feature_names_blobs)

    if st.checkbox("스케일링된 군집화 데이터 일부 확인", key="show_preprocessed_blobs_data_page_5"):
        st.write("특성 데이터 (스케일링 후, 상위 5행):")
        st.dataframe(df_blobs_scaled.head().round(3))

    st.markdown("---")

    # --- 5.2 K-평균 군집화 (K-Means Clustering) ---
    st.subheader("5.2 K-평균 군집화 (`KMeans`)")
    st.markdown("""
    K-평균은 가장 널리 사용되는 군집화 알고리즘 중 하나입니다.
    1.  사용자가 미리 군집의 개수(K)를 지정합니다.
    2.  임의의 K개 중심점(centroid)을 선택합니다.
    3.  각 데이터 포인트를 가장 가까운 중심점에 할당합니다.
    4.  각 군집의 중심점을 해당 군집에 속한 데이터 포인트들의 평균값으로 업데이트합니다.
    5.  중심점의 변화가 거의 없을 때까지 3, 4번 과정을 반복합니다.

    주요 하이퍼파라미터:
    - `n_clusters`: 군집의 개수 (K).
    - `init`: 초기 중심점 선택 방법 (`'k-means++'` (기본값), `'random'`).
    - `n_init`: 다른 초기 중심점으로 K-평균 알고리즘을 몇 번 실행할지 결정합니다. 가장 좋은 결과를 선택합니다. (`'auto'` 또는 정수)
    - `max_iter`: 한 번의 실행에서 최대 반복 횟수.
    - `random_state`: 결과 재현을 위한 시드.
    """)

    # --- 엘보우 방법 (Elbow Method)으로 적절한 K 찾기 ---
    st.markdown("#### 엘보우 방법 (Elbow Method)으로 K 선택하기")
    st.markdown("""
    K-평균은 사용자가 군집의 개수(K)를 미리 지정해야 합니다. '엘보우 방법'은 적절한 K를 찾는 데 도움을 줄 수 있는 휴리스틱 기법입니다.
    다양한 K 값에 대해 K-평균을 실행하고, 각 K에 대한 군집 내 제곱합(WCSS - Within-Cluster Sum of Squares, 또는 `inertia_`)을 계산합니다.
    WCSS 값을 K에 대해 그래프로 그렸을 때, 그래프가 팔꿈치처럼 급격히 꺾이는 지점의 K를 선택합니다. 이 지점 이후로는 K를 증가시켜도 WCSS 감소폭이 작아집니다.
    """)

    if st.checkbox("엘보우 방법 그래프 보기", key="elbow_method_page_5"):
        st.caption("K값을 변경하며 WCSS(inertia)를 계산 중입니다. 잠시 기다려주세요...")
        wcss = []
        k_range = range(1, 11) # 1부터 10까지의 K 테스트
        for i in k_range:
            kmeans_elbow = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=42)
            kmeans_elbow.fit(X_blobs_scaled)
            wcss.append(kmeans_elbow.inertia_) # inertia_는 WCSS 값을 제공
        
        fig_elbow, ax_elbow = plt.subplots(figsize=(8,5))
        ax_elbow.plot(k_range, wcss, marker='o', linestyle='--')
        ax_elbow.set_title('엘보우 방법 (Elbow Method)')
        ax_elbow.set_xlabel('군집의 수 (K)')
        ax_elbow.set_ylabel('WCSS (Inertia)')
        st.pyplot(fig_elbow)
        plt.clf()
        st.markdown("위 그래프에서 '팔꿈치'에 해당하는 K 값을 선택합니다. (예: 이 데이터에서는 K=4 부근)")

# pages/5_🔗_군집화_Clustering.py
# ... (다른 import 및 코드 부분은 동일하게 유지) ...

    st.markdown("#### K-평균 모델 학습 및 결과 시각화")
    k_selected = st.slider(
        "K-평균 군집화에 사용할 군집의 수(K)를 선택하세요:",
        min_value=2, max_value=8, value=4, key="k_slider_page_5_v2" # 키 변경 (이전과 다르게)
    )

    
    code_kmeans = """
from sklearn.cluster import KMeans
# from utils_ml import display_clustering_metrics (평가용)
# import matplotlib.pyplot as plt (시각화용)
# import numpy as np (시각화용 unique_labels 등에서 사용될 수 있음)

# # 데이터 준비 (X_blobs_scaled 사용 가정)

# 1. K-평균 모델 객체 생성
kmeans_model = KMeans(n_clusters={k_val}, init='k-means++', n_init=10, random_state=42) # {k_val}은 format으로 채워짐

# 2. 모델 학습 및 레이블 예측 (fit_predict 사용)
cluster_labels = kmeans_model.fit_predict(X_blobs_scaled)

# 3. 군집 중심점 확인
centroids = kmeans_model.cluster_centers_
# print(f"군집 중심점 (K={k_val}):\\n", centroids) # 여기도 k_val 사용

# 4. 각 데이터 포인트에 할당된 군집 레이블 확인
# print(f"각 데이터의 군집 레이블 (상위 10개): {{cluster_labels[:10]}}") # 내부 f-string의 중괄호는 {{}}로 이스케이프 처리

# 5. 모델 평가 (실루엣 계수 등)
# display_clustering_metrics(X_blobs_scaled, cluster_labels, title=f"K-평균 군집화 평가 (K={k_val})")

# 6. 결과 시각화 (2D 데이터의 경우)
# fig, ax = plt.subplots(figsize=(8,6))
# # 원본 스케일링된 데이터에 군집 레이블별로 색상 부여
# unique_labels = np.unique(cluster_labels) # np를 사용하려면 import numpy as np 필요
# for i in unique_labels:
#     ax.scatter(X_blobs_scaled[cluster_labels == i, 0], X_blobs_scaled[cluster_labels == i, 1], label=f'Cluster {{i}}', alpha=0.7, edgecolors='w') # 내부 f-string 중괄호 이스케이프
# ax.scatter(centroids[:, 0], centroids[:, 1], s=200, c='red', marker='X', label='Centroids') # 중심점 표시
# ax.set_title(f'K-Means Clustering (K={k_val})')
# ax.set_xlabel('Feature 1 (Scaled)')
# ax.set_ylabel('Feature 2 (Scaled)')
# ax.legend()
# st.pyplot(fig)
# plt.clf()
    """.format(k_val=k_selected) # k_selected 값을 {k_val} 플레이스홀더에 삽입

    st.code(code_kmeans, language='python')

    if st.button(f"K={k_selected}로 K-평균 군집화 실행", key="run_kmeans_page_5_v2"): # 키 변경
        # ... (이하 실행 코드는 이전과 동일하게 유지) ...
        st.markdown(f"#### K-평균 군집화 결과 (K={k_selected})")
        kmeans_model_ex = KMeans(n_clusters=k_selected, init='k-means++', n_init=10, random_state=42, tol=1e-4)
        cluster_labels_ex = kmeans_model_ex.fit_predict(X_blobs_scaled)
        centroids_ex = kmeans_model_ex.cluster_centers_

        # 결과 시각화
        fig_clusters, ax_clusters = plt.subplots(figsize=(8,6))
        unique_labels_ex = np.unique(cluster_labels_ex)
        
        palette = sns.color_palette("deep", len(unique_labels_ex))

        for i, label_val in enumerate(unique_labels_ex): # label 대신 label_val 사용 (외부 변수와 충돌 방지)
            ax_clusters.scatter(
                X_blobs_scaled[cluster_labels_ex == label_val, 0], 
                X_blobs_scaled[cluster_labels_ex == label_val, 1], 
                label=f'군집 {label_val}', 
                alpha=0.8, 
                edgecolors='k',
                s=70,
                color=palette[i % len(palette)]
            )
        ax_clusters.scatter(
            centroids_ex[:, 0], centroids_ex[:, 1], 
            s=250, c='black', marker='X', label='중심점 (Centroids)',
            edgecolor='white', linewidth=1.5
        )
        
        ax_clusters.set_title(f'K-평균 군집화 결과 (K={k_selected})', fontsize=15)
        ax_clusters.set_xlabel('Feature 1 (Scaled)', fontsize=12)
        ax_clusters.set_ylabel('Feature 2 (Scaled)', fontsize=12)
        ax_clusters.legend(title="범례", title_fontsize='13', fontsize='11', loc='upper right')
        ax_clusters.grid(True, linestyle='--', alpha=0.6)
        st.pyplot(fig_clusters)
        plt.clf()

        display_clustering_metrics(X_blobs_scaled, cluster_labels_ex, title=f"K-평균 군집화 평가 (K={k_selected})")

    
    st.markdown("---")

    # --- 5.3 기타 군집화 알고리즘 (간략 소개) ---
    st.subheader("5.3 기타 주요 군집화 알고리즘")
    st.markdown("""
    K-평균 외에도 다양한 군집화 알고리즘이 있습니다:
    - **계층적 군집화 (Hierarchical Clustering, 예: `AgglomerativeClustering`):** 데이터 포인트들을 계층적인 트리 구조(덴드로그램)로 연결하며 군집을 형성합니다. K를 미리 정할 필요는 없지만, 덴드로그램을 보고 적절한 수준에서 잘라내어 군집 수를 결정할 수 있습니다.
    - **DBSCAN (`DBSCAN` - Density-Based Spatial Clustering of Applications with Noise):** 밀도 기반 군집화 알고리즘으로, 데이터가 밀집된 영역을 찾아내어 군집을 형성하고, 어떤 군집에도 속하지 않는 데이터는 노이즈(noise)로 처리합니다. K-평균과 달리 군집의 모양이 복잡하거나 불규칙해도 잘 찾아낼 수 있고, K를 미리 정할 필요가 없습니다. (주요 파라미터: `eps`, `min_samples`)
    """)
    st.markdown("각 알고리즘은 데이터의 특성과 분포, 그리고 분석 목적에 따라 장단점을 가지므로, 적절한 알고리즘을 선택하는 것이 중요합니다.")

else: # df_blobs가 None일 경우 (데이터 로드 실패)
    st.error("군집화용 데이터셋을 로드할 수 없어 예제를 진행할 수 없습니다.")