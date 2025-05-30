# pages/2_🛠️_데이터_전처리.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, LabelEncoder
# from utils_ml import get_dataset # 이 페이지에서는 아래 자체 데이터 생성 함수를 사용합니다.
                                 # 필요에 따라 utils_ml의 다른 함수를 사용한다면 주석 해제 가능.

st.header("2. 데이터 전처리 (Data Preprocessing)")
st.markdown("""
머신러닝 모델의 성능은 입력 데이터의 품질에 크게 좌우됩니다. 데이터 전처리는 원시(raw) 데이터를 모델 학습에 적합한 형태로 가공하는 과정입니다.
주요 전처리 작업으로는 결측치 처리, 특성 스케일링, 범주형 데이터 인코딩, 데이터 분할 등이 있습니다.
""")

# --- 예제 DataFrame 생성 (전처리 시연용) ---
@st.cache_data # 데이터프레임 생성을 캐싱하여 반복 실행 방지
def load_preprocessing_data():
    """전처리 예제용 샘플 데이터프레임을 생성합니다."""
    data = {
        'Age': [25, 30, np.nan, 35, 22, 28, 40, np.nan, 33, 50],
        'Salary': [50000, 60000, 75000, np.nan, 45000, 55000, 120000, 85000, 70000, 150000],
        'City': ['New York', 'Paris', 'London', 'Tokyo', 'Seoul', 'Berlin', 'Paris', 'New York', 'London', 'Tokyo'],
        'Gender': ['Male', 'Female', 'Male', 'Female', 'Male', 'Male', 'Female', 'Female', 'Male', 'Female'],
        'Experience': [2, 5, 3, 8, 1, 4, 15, 6, 7, 20],
        'Purchased': [0, 1, 1, 0, 1, 0, 1, 1, 0, 1] # Target variable
    }
    df = pd.DataFrame(data)
    return df

sample_df_prep = load_preprocessing_data()

st.subheader("데이터 전처리 예제용 DataFrame")
if st.checkbox("전처리 예제 DataFrame 보기", key="show_prep_base_df_page_2"): # 페이지별 키 구분
    st.dataframe(sample_df_prep)
    st.write("결측치 확인 (열 별 개수):")
    st.dataframe(sample_df_prep.isnull().sum().rename("결측치 수"))

st.markdown("---")

# --- 2.1 결측치 처리 (Handling Missing Values) ---
st.subheader("2.1 결측치 처리")
st.markdown("""
결측치는 모델 성능에 부정적인 영향을 줄 수 있으므로 적절히 처리해야 합니다.
- **제거:** 결측치가 포함된 행이나 열을 삭제합니다 (`dropna()`). 데이터 손실이 발생할 수 있습니다.
- **대치 (Imputation):** 결측치를 특정 값(예: 평균, 중앙값, 최빈값)으로 채웁니다. `SimpleImputer`를 사용할 수 있습니다.
""")
code_imputation = """
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer

# 예제 데이터 (sample_df_prep의 'Age', 'Salary' 사용 가정)
# df_to_impute = sample_df_prep[['Age', 'Salary']].copy()

# 평균값으로 결측치 대치
# strategy: 'mean', 'median', 'most_frequent', 'constant' (fill_value 지정 필요)
imputer_mean = SimpleImputer(strategy='mean')

# fit_transform은 학습(평균 계산 등)과 변환을 동시에 수행
# 결과는 NumPy 배열이므로 다시 DataFrame으로 만들어주는 것이 좋음
# df_to_impute[['Age', 'Salary']] = imputer_mean.fit_transform(df_to_impute[['Age', 'Salary']])
# print("평균값으로 결측치 대치 후:\\n", df_to_impute)

# 다른 예: 'Salary'는 중앙값으로, 'Age'는 상수(예: -1)로 대치
# imputer_median_salary = SimpleImputer(strategy='median')
# df_to_impute['Salary'] = imputer_median_salary.fit_transform(df_to_impute[['Salary']])

# imputer_constant_age = SimpleImputer(strategy='constant', fill_value=-1)
# df_to_impute['Age'] = imputer_constant_age.fit_transform(df_to_impute[['Age']])
"""
st.code(code_imputation, language='python')

if st.checkbox("결측치 대치 (`SimpleImputer`) 예시 보기", key="imputation_page_2"):
    df_impute_ex = sample_df_prep[['Age', 'Salary']].copy() # 원본 변경 방지
    st.write("결측치 대치 전 ('Age', 'Salary' 컬럼):")
    st.dataframe(df_impute_ex)

    # 평균값으로 대치
    imputer_mean_ex = SimpleImputer(strategy='mean')
    df_impute_ex[['Age', 'Salary']] = imputer_mean_ex.fit_transform(df_impute_ex[['Age', 'Salary']])
    
    st.write("평균값으로 결측치 대치 후:")
    st.dataframe(df_impute_ex.round(2)) # 소수점 2자리로 반올림하여 표시
    # imputer_mean_ex.statistics_는 각 열에 대해 계산된 평균값을 담고 있음
    st.caption(f"Age 평균: {imputer_mean_ex.statistics_[0]:.2f}, Salary 평균: {imputer_mean_ex.statistics_[1]:.2f} (이 값들로 NaN이 채워짐)")

    # 중앙값/최빈값 대치 예시 (새로운 DataFrame에서)
    df_impute_median_mode = sample_df_prep[['Age', 'Salary']].copy()
    imputer_median = SimpleImputer(strategy='median')
    imputer_most_frequent = SimpleImputer(strategy='most_frequent')

    df_impute_median_mode['Age_median_imputed'] = imputer_median.fit_transform(df_impute_median_mode[['Age']])
    df_impute_median_mode['Salary_mode_imputed'] = imputer_most_frequent.fit_transform(df_impute_median_mode[['Salary']])

    st.write("Age는 중앙값, Salary는 최빈값(여기서는 데이터 특성상 평균/중앙값과 유사할 수 있음)으로 대치:")
    st.dataframe(df_impute_median_mode[['Age', 'Age_median_imputed', 'Salary', 'Salary_mode_imputed']].round(2))


st.markdown("---")

# --- 2.2 특성 스케일링 (Feature Scaling) ---
st.subheader("2.2 특성 스케일링")
st.markdown("""
서로 다른 범위의 값을 가진 특성들이 모델 학습에 미치는 영향을 균등하게 만들기 위해 스케일을 조정합니다.
거리 기반 알고리즘(KNN, SVM)이나 경사 하강법 기반 알고리즘(선형 회귀, 로지스틱 회귀, 신경망)에 특히 중요합니다.
- **`StandardScaler` (표준화):** 각 특성의 평균을 0, 표준편차를 1로 변환합니다. ($z = (x - \mu) / \sigma$)
- **`MinMaxScaler` (정규화):** 각 특성의 값을 0과 1 사이의 범위로 변환합니다. ($x' = (x - min) / (max - min)$)
- **`RobustScaler`:** 중앙값(median)과 사분위 범위(IQR)를 사용하여 스케일링합니다. 이상치(outlier)에 덜 민감합니다.
""")
code_scaling = """
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer # 결측치 처리를 먼저 해야 함

# sample_df_prep의 'Age', 'Salary', 'Experience' 사용 가정
# numeric_cols = ['Age', 'Salary', 'Experience']
# df_numeric = sample_df_prep[numeric_cols].copy()

# 결측치 대치 (예: 중앙값)
# imputer = SimpleImputer(strategy='median')
# df_numeric_imputed = pd.DataFrame(imputer.fit_transform(df_numeric), columns=numeric_cols)

# StandardScaler 사용
# scaler_standard = StandardScaler()
# scaled_standard_features = scaler_standard.fit_transform(df_numeric_imputed)
# df_standard_scaled = pd.DataFrame(scaled_standard_features, columns=numeric_cols)
# print("StandardScaler 적용 후:\\n", df_standard_scaled.head().round(2))

# MinMaxScaler 사용
# scaler_minmax = MinMaxScaler()
# scaled_minmax_features = scaler_minmax.fit_transform(df_numeric_imputed)
# df_minmax_scaled = pd.DataFrame(scaled_minmax_features, columns=numeric_cols)
# print("\\nMinMaxScaler 적용 후:\\n", df_minmax_scaled.head().round(2))
"""
st.code(code_scaling, language='python')

if st.checkbox("특성 스케일링 예시 보기", key="scaling_page_2"):
    # 결측치를 먼저 채운 후 스케일링 적용
    numeric_cols_for_scaling = ['Age', 'Salary', 'Experience']
    df_scale_ex_orig = sample_df_prep[numeric_cols_for_scaling].copy()
    
    imputer_for_scaling = SimpleImputer(strategy='median') # 중앙값으로 결측치 대치
    df_scale_ex_imputed = pd.DataFrame(
        imputer_for_scaling.fit_transform(df_scale_ex_orig),
        columns=numeric_cols_for_scaling
    )
    st.write("스케일링 전 데이터 (결측치는 중앙값으로 대치됨):")
    st.dataframe(df_scale_ex_imputed.round(2))

    # StandardScaler
    scaler_std = StandardScaler()
    scaled_std_data = scaler_std.fit_transform(df_scale_ex_imputed) # NumPy 배열 반환
    df_scaled_std_result = pd.DataFrame(scaled_std_data, columns=numeric_cols_for_scaling)
    st.write("`StandardScaler` 적용 후 (평균 0, 표준편차 1 근사):")
    st.dataframe(df_scaled_std_result.round(2))
    st.caption(f"StandardScaler 적용 후 평균: {df_scaled_std_result.mean().round(2).to_dict()}, 표준편차: {df_scaled_std_result.std().round(2).to_dict()}")


    # MinMaxScaler
    scaler_mm = MinMaxScaler()
    scaled_mm_data = scaler_mm.fit_transform(df_scale_ex_imputed) # NumPy 배열 반환
    df_scaled_mm_result = pd.DataFrame(scaled_mm_data, columns=numeric_cols_for_scaling)
    st.write("`MinMaxScaler` 적용 후 (0과 1 사이로 스케일링):")
    st.dataframe(df_scaled_mm_result.round(2))
    st.caption(f"MinMaxScaler 적용 후 최소값: {df_scaled_mm_result.min().round(2).to_dict()}, 최대값: {df_scaled_mm_result.max().round(2).to_dict()}")


st.markdown("---")

# --- 2.3 범주형 데이터 인코딩 (Categorical Data Encoding) ---
st.subheader("2.3 범주형 데이터 인코딩")
st.markdown("""
대부분의 머신러닝 모델은 숫자형 입력을 가정하므로, 문자열로 된 범주형 데이터를 숫자형으로 변환해야 합니다.
- **`LabelEncoder`:** 범주형 값을 0부터 (클래스 수 - 1)까지의 정수로 변환합니다. 주로 타겟 변수(y) 인코딩에 사용되거나, 순서가 있는 특성에 제한적으로 사용됩니다. (주의: 특성 간 순서 관계가 없는 명목형 특성에 사용 시 모델이 잘못된 순서를 학습할 수 있음)
- **`OneHotEncoder`:** 각 범주를 새로운 이진(0 또는 1) 특성(컬럼)으로 변환합니다. 더미 변수(dummy variable)를 생성합니다. 명목형 특성에 주로 사용됩니다.
- **`pandas.get_dummies()`:** Pandas에서 원-핫 인코딩을 쉽게 수행하는 함수입니다.
""")
code_encoding = """
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
# sample_df_prep의 'City', 'Gender' 사용 가정
# df_to_encode = sample_df_prep[['City', 'Gender']].copy()

# LabelEncoder 사용 (예: 'Gender' 컬럼)
# le = LabelEncoder()
# df_to_encode['Gender_LabelEncoded'] = le.fit_transform(df_to_encode['Gender'])
# print("LabelEncoder 적용 후 ('Gender'):\\n", df_to_encode[['Gender', 'Gender_LabelEncoded']].head())
# print("LabelEncoder 클래스:", le.classes_) # ['Female', 'Male'] -> 0, 1 매핑 확인

# OneHotEncoder 사용 (예: 'City' 컬럼)
# ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore') # sparse_output=False는 NumPy 배열 반환
                                                                # handle_unknown='ignore'는 테스트 데이터에 처음 본 범주 나오면 모두 0으로 처리
# city_one_hot_encoded = ohe.fit_transform(df_to_encode[['City']]) # 2D 배열 형태로 입력
# # 생성된 컬럼 이름 가져오기 (ohe.categories_ 또는 ohe.get_feature_names_out 사용)
# city_encoded_cols = ohe.get_feature_names_out(['City']) # 입력 특성 이름을 기반으로 새 컬럼명 생성
# df_city_encoded = pd.DataFrame(city_one_hot_encoded, columns=city_encoded_cols, index=df_to_encode.index)
# # 원본 DataFrame과 병합 (City 열은 제거)
# df_encoded_final = pd.concat([df_to_encode.drop('City', axis=1), df_city_encoded], axis=1)
# print("\\nOneHotEncoder 적용 후 ('City'):\\n", df_encoded_final.head())


# pandas.get_dummies() 사용 (가장 간편한 방법 중 하나)
# df_original = sample_df_prep.copy()
# df_dummies = pd.get_dummies(df_original, columns=['City', 'Gender'], prefix=['CityIs', 'GenderIs'], drop_first=True, dtype=int)
# # drop_first=True는 다중공선성 방지를 위해 첫 번째 범주에 대한 더미 변수 제거
# # dtype=int로 결과 타입을 정수로 지정
# print("\\npd.get_dummies() 적용 후 (drop_first=True):\\n", df_dummies.head())
"""
st.code(code_encoding, language='python')

if st.checkbox("범주형 데이터 인코딩 예시 보기", key="encoding_page_2"):
    df_encode_ex_orig = sample_df_prep[['City', 'Gender']].copy()
    st.write("인코딩 전 데이터:")
    st.dataframe(df_encode_ex_orig)

    # LabelEncoder
    df_le_ex = df_encode_ex_orig.copy()
    le = LabelEncoder()
    df_le_ex['Gender_LabelEncoded'] = le.fit_transform(df_le_ex['Gender'])
    st.write("`LabelEncoder` 적용 후 ('Gender'):")
    st.dataframe(df_le_ex[['Gender', 'Gender_LabelEncoded']])
    st.caption(f"LabelEncoder 클래스 매핑 (`le.classes_`): `{list(le.classes_)}` -> `{list(range(len(le.classes_)))}`")

    # OneHotEncoder
    df_ohe_ex_base = df_encode_ex_orig.copy()
    ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore') 
    city_one_hot_data = ohe.fit_transform(df_ohe_ex_base[['City']]) # 2D 형태로 입력
    city_encoded_cols_names = ohe.get_feature_names_out(['City'])
    df_city_ohe_result = pd.DataFrame(city_one_hot_data, columns=city_encoded_cols_names, index=df_ohe_ex_base.index)
    
    st.write("`OneHotEncoder` 적용 후 ('City'):")
    st.dataframe(df_city_ohe_result)
    st.caption("`handle_unknown='ignore'`는 테스트 시 처음 보는 범주가 나오면 모든 원핫인코딩 열을 0으로 만듭니다.")


    # pandas.get_dummies
    st.write("`pd.get_dummies()` 적용 후 ('City', 'Gender' 동시 변환, `drop_first=True`):")
    df_get_dummies_result = pd.get_dummies(
        sample_df_prep[['City', 'Gender']], 
        columns=['City', 'Gender'], 
        prefix={'City':'City', 'Gender':'Is'}, # 접두사 지정 (딕셔너리 형태 가능)
        drop_first=True, # 다중공선성 방지를 위해 첫 번째 범주 컬럼 제거
        dtype=int # 결과 타입을 정수로
    )
    st.dataframe(df_get_dummies_result)
    st.caption("`drop_first=True`는 범주가 N개일 때 N-1개의 더미 변수만 생성하여 다중공선성 문제를 완화합니다.")


st.markdown("---")

# --- 2.4 데이터 분할 (Train-Test Split) ---
st.subheader("2.4 데이터 분할 (`train_test_split`)")
st.markdown("""
머신러닝 모델을 학습시키고 평가하기 위해 전체 데이터셋을 훈련 데이터셋(training set)과 테스트 데이터셋(test set)으로 분리합니다.
- 모델은 훈련 데이터셋으로 학습됩니다.
- 학습된 모델의 성능은 테스트 데이터셋으로 평가됩니다 (모델이 보지 못한 새로운 데이터에 대한 일반화 성능 측정).
`sklearn.model_selection.train_test_split()` 함수를 사용합니다.
- `*arrays`: 분할할 특성 데이터(X)와 레이블 데이터(y).
- `test_size`: 테스트 데이터셋의 비율 (0.0 ~ 1.0 사이 실수) 또는 개수 (정수). (기본값: 0.25)
- `train_size`: 훈련 데이터셋의 비율 또는 개수. `test_size`와 둘 중 하나만 지정.
- `random_state`: 난수 시드. 동일한 시드를 사용하면 항상 같은 방식으로 데이터가 분할되어 결과 재현 가능.
- `stratify`: (분류 문제에서) 지정된 배열(보통 y 레이블)의 클래스 비율을 훈련/테스트 데이터셋 모두에 유사하게 유지. 불균형 데이터셋에 유용.
""")
code_train_test_split = """
import pandas as pd
from sklearn.model_selection import train_test_split
# sample_df_prep DataFrame이 있다고 가정 (X: 특성들, y: 'Purchased' 컬럼)

# # 전처리 과정이 선행되어야 함 (결측치 처리, 인코딩 등)
# df_processed = sample_df_prep.copy()
# # 예시: 간단한 전처리
# for col in ['Age', 'Salary']: # 숫자형 컬럼 결측치 평균으로
#     df_processed[col].fillna(df_processed[col].mean(), inplace=True)
# df_processed = pd.get_dummies(df_processed, columns=['City', 'Gender'], drop_first=True, dtype=int)

# # 특성(X)과 타겟(y) 분리
# X = df_processed.drop('Purchased', axis=1) # 'Purchased' 열을 제외한 모든 열
# y = df_processed['Purchased']              # 'Purchased' 열

# # 훈련 데이터와 테스트 데이터 분리
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y,
#     test_size=0.3,     # 테스트 데이터 비율 30%
#     random_state=42,   # 결과 재현을 위한 시드
#     stratify=y         # y의 클래스 비율을 유지하며 분할 (분류 문제에서 중요)
# )

# print("훈련 데이터 X 형태:", X_train.shape)
# print("테스트 데이터 X 형태:", X_test.shape)
# print("훈련 데이터 y 형태:", y_train.shape)
# print("테스트 데이터 y 형태:", y_test.shape)
# print("\\n훈련 데이터 y 클래스 비율:\\n", y_train.value_counts(normalize=True).round(2))
# print("테스트 데이터 y 클래스 비율:\\n", y_test.value_counts(normalize=True).round(2))
"""
st.code(code_train_test_split, language='python')

if st.checkbox("`train_test_split` 예시 보기", key="train_test_split_page_2"):
    df_split_ex_orig = sample_df_prep.copy()
    
    # 전처리 (예시: 결측치 채우고 범주형 인코딩)
    # 숫자형 컬럼 결측치: 중앙값으로 대치
    for col in ['Age', 'Salary', 'Experience']: 
        df_split_ex_orig[col].fillna(df_split_ex_orig[col].median(), inplace=True)
    # 범주형 컬럼: 원-핫 인코딩
    df_split_ex_processed = pd.get_dummies(df_split_ex_orig, columns=['City', 'Gender'], drop_first=True, dtype=int)

    X = df_split_ex_processed.drop('Purchased', axis=1)
    y = df_split_ex_processed['Purchased'] # 타겟 변수
    
    st.write("전처리 완료된 특성 데이터 `X` (상위 5행):")
    st.dataframe(X.head())
    st.write("타겟 데이터 `y` (상위 5행):")
    st.dataframe(y.head())

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.3,  # 테스트셋 비율 30%
        random_state=123, # 재현성을 위한 시드
        stratify=y      # 분류 문제에서 타겟 변수의 클래스 비율을 유지하며 분할
    )
    st.markdown("#### 분할 결과:")
    st.write(f"- `X_train` 형태: `{X_train.shape}`")
    st.write(f"- `X_test` 형태: `{X_test.shape}`")
    st.write(f"- `y_train` 형태: `{y_train.shape}`")
    st.write(f"- `y_test` 형태: `{y_test.shape}`")
    
    st.write("`y_train` 클래스 분포 (비율):")
    st.dataframe(y_train.value_counts(normalize=True).rename("비율").round(3))
    st.write("`y_test` 클래스 분포 (비율) (stratify=y로 인해 y_train과 유사):")
    st.dataframe(y_test.value_counts(normalize=True).rename("비율").round(3))

st.markdown("---")
st.markdown("데이터 전처리는 반복적이고 실험적인 과정일 수 있습니다. 데이터와 모델에 가장 적합한 전처리 방법을 찾기 위해 다양한 기법을 시도해보는 것이 중요합니다. Scikit-learn의 `Pipeline`을 사용하면 이러한 전처리 단계를 체계적으로 관리할 수 있습니다 (이후 페이지에서 소개).")