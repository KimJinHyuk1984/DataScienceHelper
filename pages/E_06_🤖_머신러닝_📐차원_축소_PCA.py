# pages/6_📐_차원_축소_PCA.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from utils.utils_ml import get_dataset # 유틸리티 함수 사용
import seaborn as sns

st.header("6. 차원 축소 (Dimensionality Reduction) - PCA")
st.markdown("""
차원 축소는 데이터의 특성(feature) 수를 줄이면서 원래 데이터의 중요한 정보는 최대한 유지하는 과정입니다.
고차원 데이터는 시각화가 어렵고, '차원의 저주(curse of dimensionality)'로 인해 모델 성능이 저하될 수 있습니다.
차원 축소를 통해 데이터를 더 잘 이해하고, 모델 학습 속도를 높이며, 과적합을 줄이는 데 도움이 될 수 있습니다.

여기서는 대표적인 차원 축소 기법인 **주성분 분석 (Principal Component Analysis, PCA)**에 대해 알아봅니다.
""")

# --- 예제 데이터셋 로드 (와인 또는 유방암 데이터 - 다수의 특성을 가짐) ---
st.subheader("PCA 예제용 데이터셋 (와인 또는 유방암 데이터)")
dataset_options_pca = ["wine", "breast_cancer", "iris"] # Iris도 비교적 적은 특성이지만 가능
chosen_dataset_pca = st.selectbox(
    "PCA 예제에 사용할 데이터셋을 선택하세요:",
    dataset_options_pca,
    index=0, # 기본값 'wine'
    key="pca_dataset_selector"
)

try:
    df_pca_orig, target_names_pca, feature_names_pca = get_dataset(chosen_dataset_pca)
    X_pca_orig = df_pca_orig.drop('target', axis=1)
    y_pca_orig = df_pca_orig['target']
except Exception as e:
    st.error(f"{chosen_dataset_pca} 데이터셋 로드 중 오류 발생: {e}")
    df_pca_orig = None
    X_pca_orig = None
    y_pca_orig = None

if df_pca_orig is not None and X_pca_orig is not None:
    if st.checkbox(f"{chosen_dataset_pca} 원본 데이터셋 미리보기 (상위 5행)", key=f"show_{chosen_dataset_pca}_df_page_6"):
        st.dataframe(X_pca_orig.head())
        st.write(f"원본 데이터 형태 (특성): {X_pca_orig.shape}")
        st.write("특성 (Features):", feature_names_pca)
        st.write("결측치 확인:")
        st.dataframe(X_pca_orig.isnull().sum().rename("결측치 수"))

    st.markdown("---")

    # --- 6.1 데이터 스케일링 (PCA 전처리) ---
    st.subheader("6.1 데이터 스케일링 (PCA 전처리)")
    st.markdown("""
    PCA는 데이터의 분산(variance)을 기반으로 주성분을 찾으므로, 특성들의 스케일이 다르면 분산이 큰 특성이 주성분에 과도한 영향을 미칠 수 있습니다.
    따라서 PCA 적용 전에 각 특성을 **표준화(Standardization)**하는 것이 일반적입니다.
    """)

    scaler_pca = StandardScaler()
    X_pca_scaled = scaler_pca.fit_transform(X_pca_orig)
    
    df_pca_scaled = pd.DataFrame(X_pca_scaled, columns=feature_names_pca)

    if st.checkbox("스케일링된 PCA 입력 데이터 일부 확인", key="show_preprocessed_pca_data_page_6"):
        st.write("특성 데이터 (스케일링 후, 상위 5행):")
        st.dataframe(df_pca_scaled.head().round(3))
        st.caption(f"스케일링 후 각 특성의 평균: {df_pca_scaled.mean().mean():.2e}, 표준편차: {df_pca_scaled.std().mean():.2f} (각각 0과 1에 가까워야 함)")


    st.markdown("---")

    # --- 6.2 주성분 분석 (PCA) ---
    st.subheader("6.2 주성분 분석 (`PCA`)")
    st.markdown(f"""
    PCA는 데이터의 분산이 가장 큰 방향을 첫 번째 주성분(PC1)으로, 그 다음으로 분산이 큰 (PC1과 직교하는) 방향을 두 번째 주성분(PC2)으로 찾는 방식으로 주성분들을 계산합니다.
    원래 특성 공간을 새로운 (더 낮은 차원의) 주성분 공간으로 변환합니다.

    주요 하이퍼파라미터:
    - `n_components`: 유지할 주성분의 개수.
        - 정수: 주성분 개수를 직접 지정.
        - `0 < n_components < 1` (실수): 유지할 총 분산의 비율을 지정 (예: `0.95`는 분산의 95%를 설명하는 주성분 개수 자동 선택).
        - `None`: 모든 주성분을 유지 (`min(n_samples, n_features)`).
    - `random_state`: (솔버가 확률적일 경우) 결과 재현을 위한 시드.
    """)



    n_features = X_pca_scaled.shape[1]
    pca_n_components = st.slider(
        "PCA에서 유지할 주성분(n_components) 개수를 선택하세요:",
        min_value=1, 
        max_value=n_features, 
        value=min(2, n_features),
        key="pca_n_components_slider_page_6_v2" # 키 변경
    )

    # f-string 대신 .format()과 이중 중괄호 사용으로 변경
    code_pca = """
from sklearn.decomposition import PCA
# from sklearn.preprocessing import StandardScaler (스케일링은 이미 완료 가정)
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt

# # 데이터 준비 (X_pca_scaled 사용 가정)

# 1. PCA 모델 객체 생성
# n_components: 유지할 주성분의 개수. 여기서는 {n_comps_val}로 설정.
pca = PCA(n_components={n_comps_val}, random_state=42)

# 2. PCA 모델 학습 및 데이터 변환
# fit_transform()은 주성분을 찾고(fit), 데이터를 해당 주성분 공간으로 변환(transform)합니다.
X_pca_transformed = pca.fit_transform(X_pca_scaled)

# # 변환된 데이터 확인
# print(f"원본 데이터 형태: {{X_pca_scaled.shape}}") # 이중 중괄호로 이스케이프
# print(f"PCA 변환 후 데이터 형태: {{X_pca_transformed.shape}}") # 이중 중괄호로 이스케이프
# df_pca_result = pd.DataFrame(X_pca_transformed, columns=[f'PC{{i+1}}' for i in range(X_pca_transformed.shape[1])]) # 내부 f-string의 i 변수 부분도 이스케이프
# print("\\nPCA 변환 데이터 (상위 5행):\\n", df_pca_result.head().round(3))

# # 설명된 분산 (Explained Variance)
# # 각 주성분이 원본 데이터의 분산을 얼마나 설명하는지를 나타냅니다.
explained_variance_ratio = pca.explained_variance_ratio_
# print("\\n각 주성분의 설명된 분산 비율:", np.round(explained_variance_ratio, 4))
# print(f"선택된 {n_comps_val}개 주성분으로 설명되는 총 분산 비율: {{np.sum(explained_variance_ratio):.4f}}") # 이중 중괄호 및 n_comps_val 사용

# # (선택 사항) 주성분 로딩 (Principal Component Loadings)
# # 각 주성분이 원래 특성들과 어떤 관계를 가지는지 보여줍니다. (pca.components_)
# # feature_names_pca 변수는 이 코드 블록 바깥 스코프에 있어야 함 (실제로는 있음)
# loadings = pd.DataFrame(pca.components_.T, columns=[f'PC{{i+1}}' for i in range(pca.n_components_)], index=feature_names_pca) # 내부 f-string의 i 변수 부분 이스케이프
# print("\\n주성분 로딩:\\n", loadings.round(3))
    """.format(n_comps_val=pca_n_components) # .format()으로 외부 변수 주입

    st.code(code_pca, language='python')

    if st.button(f"PCA 실행 (n_components={pca_n_components})", key="run_pca_page_6_v2"): # 키 변경
        # ... (이하 실행 코드는 이전과 동일하게 유지) ...
        st.markdown(f"#### PCA 결과 (n_components={pca_n_components})")
        
        pca_ex = PCA(n_components=pca_n_components, random_state=42)
        X_pca_transformed_ex = pca_ex.fit_transform(X_pca_scaled)

        df_pca_result_ex = pd.DataFrame(
            X_pca_transformed_ex, 
            columns=[f'PC{i+1}' for i in range(pca_n_components)]
        )
        st.write(f"PCA 변환 후 데이터 형태: `{X_pca_transformed_ex.shape}`")
        st.write("PCA 변환 데이터 (상위 5행):")
        st.dataframe(df_pca_result_ex.head().round(3))

        st.markdown("##### 설명된 분산 (Explained Variance)")
        explained_variance_ratio_ex = pca_ex.explained_variance_ratio_
        explained_variance_df = pd.DataFrame({
            '주성분 (Principal Component)': [f'PC{i+1}' for i in range(pca_n_components)],
            '설명된 분산 비율 (Explained Variance Ratio)': explained_variance_ratio_ex
        }).set_index('주성분 (Principal Component)')
        
        st.dataframe(explained_variance_df.style.format({'설명된 분산 비율 (Explained Variance Ratio)': "{:.4f}"}))
        st.metric(
            label=f"선택된 {pca_n_components}개 주성분으로 설명되는 총 분산 비율",
            value=f"{np.sum(explained_variance_ratio_ex):.4f}"
        )

        # ... (이하 누적 설명 분산 그래프 및 2D 시각화 코드는 이전과 동일하게 유지) ...
        if pca_n_components > 1 and pca_n_components <= n_features :
            pca_full = PCA(n_components=None, random_state=42)
            pca_full.fit(X_pca_scaled)
            
            fig_explained_var, ax_explained_var = plt.subplots(figsize=(8,5))
            ax_explained_var.plot(
                np.arange(1, len(pca_full.explained_variance_ratio_) + 1),
                np.cumsum(pca_full.explained_variance_ratio_),
                marker='o', linestyle='--'
            )
            ax_explained_var.set_xlabel("주성분 개수")
            ax_explained_var.set_ylabel("누적 설명 분산 비율 (Cumulative Explained Variance)")
            ax_explained_var.set_title("주성분 개수에 따른 누적 설명 분산")
            ax_explained_var.grid(True, linestyle=':', alpha=0.7)
            if pca_n_components <= len(pca_full.explained_variance_ratio_):
                ax_explained_var.axvline(x=pca_n_components, color='red', linestyle=':', label=f'선택된 개수: {pca_n_components}')
                ax_explained_var.axhline(y=np.sum(explained_variance_ratio_ex), color='green', linestyle=':', label=f'현재 설명 분산: {np.sum(explained_variance_ratio_ex):.2f}')
                ax_explained_var.legend()
            st.pyplot(fig_explained_var)
            plt.clf()

        if pca_n_components >= 2 and y_pca_orig is not None:
            st.markdown("##### PCA 변환 데이터 2D 시각화 (PC1 vs PC2)")
            df_pca_plot = pd.DataFrame(X_pca_transformed_ex[:, :2], columns=['PC1', 'PC2'])
            df_pca_plot['target'] = y_pca_orig

            fig_pca_scatter, ax_pca_scatter = plt.subplots(figsize=(9,7))
            if target_names_pca is not None:
                unique_targets = sorted(df_pca_plot['target'].unique())
                palette = sns.color_palette("deep", len(unique_targets))
                for i, target_val in enumerate(unique_targets): # 변수 이름 target_val로 변경
                    subset = df_pca_plot[df_pca_plot['target'] == target_val] # target_val 사용
                    ax_pca_scatter.scatter(subset['PC1'], subset['PC2'], label=target_names_pca[target_val], alpha=0.8, s=70, edgecolor='k', color=palette[i % len(palette)])
                ax_pca_scatter.legend(title=f"{chosen_dataset_pca.capitalize()} Classes", title_fontsize='13')
            else:
                ax_pca_scatter.scatter(df_pca_plot['PC1'], df_pca_plot['PC2'], alpha=0.7, s=50, edgecolor='k')

            ax_pca_scatter.set_xlabel("주성분 1 (Principal Component 1)")
            ax_pca_scatter.set_ylabel("주성분 2 (Principal Component 2)")
            ax_pca_scatter.set_title(f"{chosen_dataset_pca.capitalize()} 데이터의 PCA 2D 시각화")
            ax_pca_scatter.grid(True, linestyle=':', alpha=0.6)
            st.pyplot(fig_pca_scatter)
            plt.clf()

    


    st.markdown("---")
    st.markdown("""
    PCA는 데이터 시각화, 노이즈 제거, 특성 공학, 머신러닝 모델의 입력 차원 축소 등 다양하게 활용될 수 있습니다.
    적절한 주성분 개수를 선택하는 것이 중요하며, 이는 보통 '설명된 분산'을 고려하여 결정합니다.
    이 외에도 t-SNE (`TSNE`), LDA (`LinearDiscriminantAnalysis` - 지도학습 기반 차원축소) 등 다양한 차원 축소 기법이 있습니다.
    """)

else: # df_pca_orig가 None일 경우 (데이터 로드 실패)
    st.error(f"{chosen_dataset_pca} 데이터셋을 로드할 수 없어 PCA 예제를 진행할 수 없습니다.")