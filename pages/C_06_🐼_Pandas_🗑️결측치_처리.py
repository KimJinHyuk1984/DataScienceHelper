# pages/6_🗑️_결측치_처리.py
import streamlit as st
import pandas as pd
import numpy as np
import scipy
from utils.utils_pandas import display_dataframe_info

st.sidebar.title("라이브러리 버전 (Streamlit 환경)")
st.sidebar.info(f"""
- Pandas: {pd.__version__}
- NumPy: {np.__version__}
- SciPy: {scipy.__version__}
- Streamlit: {st.__version__}
""")
# --- 페이지 헤더 ---

st.header("6. 결측치(Missing Data) 처리")
st.markdown("""
실제 데이터에는 값이 누락된 경우가 많습니다. Pandas는 이러한 결측치(주로 `NaN` - Not a Number로 표시됨)를 효과적으로 다루는 다양한 방법을 제공합니다.
결측치를 제대로 처리하지 않으면 분석 결과가 왜곡될 수 있습니다.
""")

# --- 예제 DataFrame 생성 (결측치 포함) ---
@st.cache_data
def create_sample_missing_df():
    data = {
        'A': [1, 2, np.nan, 4, 5, np.nan],
        'B': [np.nan, 7, 8, 9, np.nan, 11], # <--- 이 부분이 숫자와 np.nan으로만 되어 있는지 확인!
        'C': ['x', 'y', 'z', 'x', np.nan, 'z'],
        'D': [10.0, 20.0, 30.0, 40.0, 50.0, 60.0]
    }
    return pd.DataFrame(data, index=[f'R{i}' for i in range(6)])

sample_df_missing = create_sample_missing_df()

st.subheader("결측치 처리 예제용 DataFrame 확인")
if st.checkbox("결측치 예제 DataFrame 보기", key="show_missing_base_df_page"):
    display_dataframe_info(sample_df_missing, "예제 DataFrame (sample_df_missing)", max_rows_to_display=6, show_describe=False)

st.markdown("---")

# --- 6.1 결측치 식별 ---
st.subheader("6.1 결측치 식별")
st.markdown("""
- `df.isnull()` 또는 `df.isna()`: 각 요소가 결측치인지 여부를 불리언 DataFrame으로 반환합니다 (`True`이면 결측치).
- `df.notnull()` 또는 `df.notna()`: 각 요소가 결측치가 아닌지 여부를 불리언 DataFrame으로 반환합니다 (`True`이면 유효한 값).
- `df.isnull().sum()`: 각 열별 결측치 개수를 반환합니다.
- `df.isnull().any()`: 각 열별로 결측치가 하나라도 있는지 여부를 반환합니다.
""")
code_identify_missing = """
import pandas as pd
import numpy as np
# sample_df_missing DataFrame이 이미 있다고 가정

# 결측치 여부 확인 (True/False DataFrame)
is_null_df = sample_df_missing.isnull()
# print("결측치 여부 (isnull()):\\n", is_null_df)

# 각 열별 결측치 개수
null_counts_per_column = sample_df_missing.isnull().sum()
# print("\\n각 열별 결측치 개수:\\n", null_counts_per_column)

# 전체 결측치 개수
total_null_count = sample_df_missing.isnull().sum().sum()
# print(f"\\n전체 결측치 개수: {total_null_count}")

# 유효한 값 여부 확인 (notnull())
not_null_df = sample_df_missing.notnull()
# print("\\n유효한 값 여부 (notnull()):\\n", not_null_df)
"""
st.code(code_identify_missing, language='python')

if st.checkbox("결측치 식별 예시 보기", key="identify_missing_page"):
    st.write("`sample_df_missing.isnull()` (결측치면 True):")
    st.dataframe(sample_df_missing.isnull())
    st.markdown("---")
    st.write("`sample_df_missing.isnull().sum()` (열별 결측치 개수):")
    st.dataframe(sample_df_missing.isnull().sum().rename("결측치 수"))
    st.markdown("---")
    st.write(f"`sample_df_missing.isnull().sum().sum()` (전체 결측치 개수): {sample_df_missing.isnull().sum().sum()}")
    st.markdown("---")
    st.write("`sample_df_missing.notnull()` (유효한 값이면 True):")
    st.dataframe(sample_df_missing.notnull())


st.markdown("---")

# --- 6.2 결측치 제거 (`.dropna()`) ---
st.subheader("6.2 결측치 제거 (`.dropna()`)")
st.markdown("""
결측치가 포함된 행 또는 열을 제거합니다.
- `axis`: 제거할 축 (0은 행, 1은 열. 기본값 0).
- `how`: `'any'` (하나라도 NaN이면 제거, 기본값), `'all'` (모든 값이 NaN이면 제거).
- `thresh`: 정수값. 해당 행/열에서 유효한 값(NaN이 아닌 값)의 최소 개수를 지정. 이보다 적으면 제거.
- `subset`: 특정 열(또는 행, `axis=1`일 때)을 기준으로 NaN을 검사하고 제거할 때 사용. 리스트 형태로 컬럼명 전달.
- `inplace`: 원본 DataFrame을 직접 수정할지 여부 (기본값 `False`).
""")
code_dropna = """
import pandas as pd
import numpy as np
# sample_df_missing DataFrame이 이미 있다고 가정

# NaN이 하나라도 포함된 행 제거 (기본 동작)
df_dropped_any_row = sample_df_missing.dropna() # how='any', axis=0 기본값
# display_dataframe_info(df_dropped_any_row, "NaN 포함 행 제거 (dropna())")

# 모든 값이 NaN인 행 제거 (이 예제에서는 해당 없음)
df_dropped_all_row = sample_df_missing.dropna(how='all')
# display_dataframe_info(df_dropped_all_row, "모든 값이 NaN인 행 제거 (dropna(how='all'))")

# NaN이 포함된 열 제거
df_dropped_any_col = sample_df_missing.dropna(axis=1) # 또는 axis='columns'
# display_dataframe_info(df_dropped_any_col, "NaN 포함 열 제거 (dropna(axis=1))")

# 'A' 또는 'B' 열에 NaN이 있는 행 제거
df_dropped_subset = sample_df_missing.dropna(subset=['A', 'B'])
# display_dataframe_info(df_dropped_subset, "A 또는 B 열에 NaN 있는 행 제거 (dropna(subset=['A', 'B']))")

# 유효한 값이 3개 미만인 행 제거
df_dropped_thresh = sample_df_missing.dropna(thresh=3)
# display_dataframe_info(df_dropped_thresh, "유효한 값 3개 미만인 행 제거 (dropna(thresh=3))")
"""
st.code(code_dropna, language='python')

if st.checkbox("`.dropna()` 예시 보기", key="dropna_page"):
    st.write("원본 DataFrame (sample_df_missing):")
    st.dataframe(sample_df_missing)

    st.write("`sample_df_missing.dropna()` (NaN 있는 행 모두 제거):")
    display_dataframe_info(sample_df_missing.dropna(), "dropna() 결과", max_rows_to_display=6, show_describe=False)

    st.write("`sample_df_missing.dropna(axis=1)` (NaN 있는 열 모두 제거):")
    display_dataframe_info(sample_df_missing.dropna(axis=1), "dropna(axis=1) 결과", max_rows_to_display=6, show_describe=False)

    st.write("`sample_df_missing.dropna(subset=['A', 'C'])` ('A' 또는 'C' 열에 NaN 있는 행 제거):")
    display_dataframe_info(sample_df_missing.dropna(subset=['A', 'C']), "dropna(subset=['A','C']) 결과", max_rows_to_display=6, show_describe=False)
    
    st.write("`sample_df_missing.dropna(thresh=3)` (유효한 값이 3개 미만인 행 제거):")
    display_dataframe_info(sample_df_missing.dropna(thresh=3), "dropna(thresh=3) 결과", max_rows_to_display=6, show_describe=False)


st.markdown("---")

# --- 6.3 결측치 채우기 (`.fillna()`) ---
st.subheader("6.3 결측치 채우기 (`.fillna()`)")
st.markdown("""
결측치를 특정 값 또는 방법으로 채웁니다.
- `value`: NaN을 채울 스칼라 값, 딕셔너리, Series, 또는 DataFrame. 딕셔너리를 사용하면 열마다 다른 값으로 채울 수 있음.
- `method`: 채우기 방법.
  - `'ffill'` 또는 `'pad'`: 앞의 유효한 값으로 채움 (Forward fill).
  - `'bfill'` 또는 `'backfill'`: 뒤의 유효한 값으로 채움 (Backward fill).
- `axis`: 채우기를 적용할 축 (0은 열 방향, 1은 행 방향).
- `limit`: 연속된 NaN을 채울 최대 개수 (ffill/bfill 사용 시).
- `inplace`: 원본 DataFrame을 직접 수정할지 여부 (기본값 `False`).
""")
code_fillna = """
import pandas as pd
import numpy as np
# sample_df_missing DataFrame이 이미 있다고 가정

# 모든 NaN을 0으로 채우기
df_filled_zero = sample_df_missing.fillna(0)
# display_dataframe_info(df_filled_zero, "모든 NaN을 0으로 채움 (fillna(0))")

# 각 열마다 다른 값으로 채우기 (딕셔너리 사용)
fill_values = {'A': sample_df_missing['A'].mean(), # A열은 평균값으로
               'B': 0,                             # B열은 0으로
               'C': 'Unknown'}                     # C열은 'Unknown'으로
df_filled_specific = sample_df_missing.fillna(value=fill_values)
# display_dataframe_info(df_filled_specific, "열마다 다른 값으로 채움")

# ffill (앞의 값으로 채우기)
df_filled_ffill = sample_df_missing.fillna(method='ffill')
# display_dataframe_info(df_filled_ffill, "ffill로 채움")

# bfill (뒤의 값으로 채우기), 최대 1개만
df_filled_bfill_limit = sample_df_missing.fillna(method='bfill', limit=1)
# display_dataframe_info(df_filled_bfill_limit, "bfill로 최대 1개 채움")
"""
st.code(code_fillna, language='python')

if st.checkbox("`.fillna()` 예시 보기", key="fillna_page"):
    st.write("원본 DataFrame (sample_df_missing):")
    st.dataframe(sample_df_missing)

    st.write("`sample_df_missing.fillna(-1)` (모든 NaN을 -1로 채움):")
    display_dataframe_info(sample_df_missing.fillna(-1), "fillna(-1) 결과", max_rows_to_display=6, show_describe=False)

    st.write("`sample_df_missing.fillna(method='ffill')` (앞의 값으로 채움):")
    display_dataframe_info(sample_df_missing.fillna(method='ffill'), "fillna(method='ffill') 결과", max_rows_to_display=6, show_describe=False)

    fill_values_ex = {'A': 99, 'B': sample_df_missing['B'].median(), 'C': '정보 없음'}
    st.write(f"`sample_df_missing.fillna(value={fill_values_ex})` (열별 특정 값으로 채움):")
    display_dataframe_info(sample_df_missing.fillna(value=fill_values_ex), "열별 특정 값 채우기 결과", max_rows_to_display=6, show_describe=False)


st.markdown("---")

# --- 6.4 보간법 (`.interpolate()`) ---
st.subheader("6.4 보간법 (`.interpolate()`)")
st.markdown("""
결측치를 주변 값들을 이용하여 추정된 값으로 채웁니다. 주로 숫자형 데이터나 시계열 데이터에 사용됩니다.
- `method`: 보간 방법.
  - `'linear'`: 선형 보간 (기본값).
  - `'polynomial'`, `'spline'`: 다항식 또는 스플라인 보간 (차수 `order` 지정 필요).
  - `'time'`: 시계열 데이터의 경우 시간 간격을 고려하여 보간. (인덱스가 DatetimeIndex여야 함)
  - `'nearest'`, `'quadratic'`, `'cubic'` 등 다양한 방법 지원.
- `limit_direction`: `'forward'`, `'backward'`, `'both'` 중 선택하여 보간 방향 및 한계 지정.
- `limit`: 연속된 NaN을 보간할 최대 개수.
""")
code_interpolate = """
import pandas as pd
import numpy as np
# sample_df_missing DataFrame이 이미 있다고 가정 (숫자형 컬럼에 대해 주로 사용)

# 'A' 컬럼에 대해 선형 보간 (기본값)
df_interpolated_A = sample_df_missing.copy() # 원본 변경 방지
df_interpolated_A['A_linear'] = df_interpolated_A['A'].interpolate(method='linear')
# display_dataframe_info(df_interpolated_A[['A', 'A_linear']], "'A' 컬럼 선형 보간")

# 'B' 컬럼에 대해 다항식 보간 (order=2, 2차 다항식)
df_interpolated_B = sample_df_missing.copy()
df_interpolated_B['B_poly'] = df_interpolated_B['B'].interpolate(method='polynomial', order=2)
# display_dataframe_info(df_interpolated_B[['B', 'B_poly']], "'B' 컬럼 2차 다항식 보간")
"""
st.code(code_interpolate, language='python')

# pages/6_🗑️_결측치_처리.py 의 .interpolate() 예시 부분

# ... (파일 상단 및 다른 코드는 동일) ...

# pages/6_🗑️_결측치_처리.py 의 .interpolate() 예시 부분
# ... (파일 상단 및 다른 코드는 동일) ...

if st.checkbox("`.interpolate()` 예시 보기", key="interpolate_page"):
    st.write("--- 원본 DataFrame (sample_df_missing)의 숫자형 컬럼 ---")
    st.dataframe(sample_df_missing[['A','B','D']])

    df_interpolated_ex = sample_df_missing.copy()

    # 'A' 컬럼 보간 (선형)
    st.write("--- 'A' 컬럼 선형 보간 ---")
    # ... (기존 'A' 컬럼 보간 코드는 정상 작동하므로 유지) ...
    try:
        df_interpolated_ex['A_linear_interp'] = df_interpolated_ex['A'].interpolate(method='linear')
        st.write("'A' 컬럼 선형 보간 후 ('A_linear_interp'):")
        st.dataframe(df_interpolated_ex[['A', 'A_linear_interp']])
    except Exception as e:
        st.error(f"'A' 컬럼 보간 중 오류: {e}")


    st.info("ℹ️ 'nearest', 'polynomial', 'spline' 등의 보간 방법을 사용하려면 `scipy` 라이브러리가 필요합니다.\n설치: `pip install scipy`")
    
    # 'B' 컬럼 보간 - 'nearest' 대신 'linear'로 변경하고 안내문 추가
    st.write("--- 'B' 컬럼 보간 시연 (원래 `method='nearest'` 예시) ---")
    st.markdown("""
    **참고:** `method='nearest'`는 특정 환경(라이브러리 버전 조합, Streamlit 실행) 및 데이터 패턴에서 예기치 않은 `TypeError`를 발생시키는 경우가 보고되고 있습니다. 
    이 예제에서는 안정적인 시연을 위해 `method='linear'`를 대신 사용합니다. 
    개념적으로 `nearest`는 가장 가까운 유효한 값으로 NaN을 채우는 방식입니다.
    """)
    
    st.write(f"Debug: 'B' 컬럼 dtype 보간 전: `{df_interpolated_ex['B'].dtype}`")
    st.text("보간 전 'B' 컬럼 내용:\n" + df_interpolated_ex['B'].to_string())
    
    interpolated_b_series = None
    try:
        # 'nearest' 대신 'linear' 사용 또는 'nearest'를 시도하되 에러 발생 시 안내
        # 여기서는 'linear'로 대체하여 보여주는 것을 제안합니다.
        st.write("Debug: `df_interpolated_ex['B'].interpolate(method='linear')` 실행 (원래 'nearest' 자리)")
        interpolated_b_series = df_interpolated_ex['B'].interpolate(method='linear', limit_direction='both') # linear도 limit_direction 사용 가능
        st.success("`interpolate(method='linear')` 실행 완료 (원래 'nearest' 자리).")
        
        st.write(f"Debug: 보간된 'B' 시리즈 (`interpolated_b_series` with linear) dtype: `{interpolated_b_series.dtype}`")
        st.text("보간된 'B' 시리즈 내용 (linear):\n" + interpolated_b_series.to_string())

        df_interpolated_ex['B_interp_demo'] = interpolated_b_series # 컬럼 이름 변경
        
        df_to_display = df_interpolated_ex[['B', 'B_interp_demo']]
        st.write("--- `df_to_display` (원본 B, 보간된 B) 정보 (linear) ---")
        st.dataframe(df_to_display)

    except Exception as e: 
        st.error(f"보간 중 오류 발생: {e}")
        st.write("원본 'B' 컬럼 내용:\n" + df_interpolated_ex['B'].to_string())

    st.caption("선형 보간은 NaN 양쪽의 값들을 사용하여 직선적으로 값을 채웁니다. Nearest는 가장 가까운 유효한 값으로 채웁니다 (이 예제에서는 linear로 대체 시연).")

st.markdown("---")
st.markdown("결측치 처리는 데이터의 특성과 분석 목적에 따라 적절한 방법을 선택하는 것이 중요합니다. 무조건 제거하거나 특정 값으로 채우기보다는 데이터 손실과 왜곡을 최소화하는 방향으로 신중하게 접근해야 합니다.")