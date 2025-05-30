# pages/7_✨_데이터_변환.py
import streamlit as st
import pandas as pd
import numpy as np
from utils.utils_pandas import display_dataframe_info

st.header("7. 데이터 변환 (Transformation)")
st.markdown("""
Pandas는 기존 데이터를 새로운 형태로 가공하거나, 특정 함수를 적용하여 값을 변환하는 다양한 기능을 제공합니다.
이는 데이터 분석을 위한 전처리 과정에서 매우 중요합니다.
""")

# --- 예제 DataFrame 생성 ---
@st.cache_data
def create_sample_transform_df():
    data = {
        'Name': ['  alice  ', ' BOB  ', 'Charlie', ' david ', 'Eve'],
        'Age': [25, 30, 22, 35, 28], # 이전 수정으로 'Age' 컬럼 추가됨
        'City': ['New York', 'Paris', 'London', 'Tokyo', 'Seoul'], # <<-- 이 'City' 컬럼을 다시 추가!
        'BirthDate': ['1990-05-15', '1985-11-20', '1992-07-01', '1988-01-30', '1995-09-10'],
        'Score_Math': [85, 92, 78, 95, 88],
        'Score_English': [70, 81, 90, 75, 85],
        'Department': ['HR', 'IT', 'Sales', 'IT', 'HR'],
        'Salary': [60000, 90000, 75000, 92000, 62000]
    }
    df = pd.DataFrame(data)
    df['BirthDate'] = pd.to_datetime(df['BirthDate']) # 날짜 타입으로 변환
    return df

sample_df_transform = create_sample_transform_df()

st.subheader("데이터 변환 예제용 DataFrame 확인")
if st.checkbox("데이터 변환 예제 DataFrame 보기", key="show_transform_base_df_page"):
    display_dataframe_info(sample_df_transform, "예제 DataFrame (sample_df_transform)", max_rows_to_display=5)

st.markdown("---")

# --- 7.1 함수 적용 (`.apply()`, `.applymap()`, `.map()`) ---
st.subheader("7.1 함수 적용 (`.apply()`, `.applymap()`, `.map()`)")
st.markdown("""
- **`df.apply(func, axis=0)`**: DataFrame의 행 또는 열 전체에 함수(`func`)를 적용합니다.
  - `axis=0` (기본값): 각 열(Series)에 함수 적용.
  - `axis=1`: 각 행(Series)에 함수 적용.
- **`df.applymap(func)`**: DataFrame의 각 요소별로 함수를 적용합니다 (DataFrame에만 사용 가능, Series에는 없음). 함수는 스칼라 값을 받아 스칼라 값을 반환해야 합니다.
- **`Series.map(arg, na_action=None)`**: Series의 각 요소별로 함수를 적용하거나, 딕셔너리/Series를 사용하여 값을 매핑합니다.
""")
code_apply_map = """
import pandas as pd
import numpy as np
# sample_df_transform DataFrame이 이미 있다고 가정

# apply() 예시: 각 숫자형 열의 최대값과 최소값의 차이 계산
numeric_cols = ['Score_Math', 'Score_English', 'Salary']
range_func = lambda x: x.max() - x.min() # x는 각 열(Series)
col_ranges = sample_df_transform[numeric_cols].apply(range_func, axis=0)
# print("각 숫자 열의 범위 (max-min):\\n", col_ranges)

# apply() 예시: 각 행의 점수 총합 계산 (새로운 컬럼 추가)
# df_copy = sample_df_transform.copy() # 원본 변경 방지
# df_copy['TotalScore'] = df_copy[['Score_Math', 'Score_English']].apply(np.sum, axis=1)
# display_dataframe_info(df_copy, "TotalScore가 추가된 DataFrame", show_describe=False)


# applymap() 예시: 숫자형 데이터에 100 더하기 (DataFrame 전체 요소에 적용)
# (주의: applymap은 문자열 등 다른 타입 컬럼에 적용 시 에러 발생 가능. 숫자형 컬럼만 선택 후 적용)
# numeric_df = sample_df_transform[numeric_cols]
# df_plus_100 = numeric_df.applymap(lambda x: x + 100 if pd.notnull(x) else x)
# print("\\n숫자형 컬럼에 100 더하기 (applymap):\\n", df_plus_100.head())


# map() 예시: 'Department' 열의 값을 코드로 매핑
dept_mapping = {'HR': 1, 'IT': 2, 'Sales': 3}
# sample_df_transform['Dept_Code'] = sample_df_transform['Department'].map(dept_mapping)
# print("\\n'Department'를 코드로 매핑 (map):\\n", sample_df_transform[['Department', 'Dept_Code']].head())
"""
st.code(code_apply_map, language='python')

if st.checkbox("함수 적용 예시 보기", key="apply_map_page"):
    st.write("`sample_df_transform[['Score_Math', 'Salary']].apply(np.sqrt)` (제곱근 계산, 열별 적용):")
    display_dataframe_info(sample_df_transform[['Score_Math', 'Salary']].apply(np.sqrt).round(2), "제곱근 계산 결과", max_rows_to_display=5, show_describe=False, show_dtypes=False)

    st.markdown("---")
    df_applymap_ex = sample_df_transform[['Age','Score_Math']].copy() # 숫자형 컬럼만 선택
    st.write("`df_applymap_ex.applymap(lambda x: x * 2)` (모든 요소에 2 곱하기):")
    display_dataframe_info(df_applymap_ex.applymap(lambda x: x*2 if pd.notnull(x) else x), "applymap 결과", max_rows_to_display=5, show_describe=False, show_dtypes=False)

    st.markdown("---")
    city_to_region = {'New York': 'America', 'Paris': 'Europe', 'London': 'Europe', 'Tokyo': 'Asia', 'Seoul': 'Asia'}
    st.write(f"`sample_df_transform['City'].map({city_to_region})` (도시를 지역으로 매핑):")
    region_series = sample_df_transform['City'].map(city_to_region)
    st.dataframe(pd.concat([sample_df_transform['City'], region_series.rename('Region')], axis=1))


st.markdown("---")

# --- 7.2 인덱스/컬럼 이름 변경 (`.rename()`) ---
st.subheader("7.2 인덱스/컬럼 이름 변경 (`.rename()`)")
st.markdown("""
행 인덱스 또는 열 이름을 변경합니다. 딕셔너리를 사용하여 특정 이름만 변경하거나, 함수를 적용하여 모든 이름을 일괄 변경할 수 있습니다.
- `mapper`: 변경할 이름 매핑 (딕셔너리 또는 함수).
- `index` 또는 `columns`: `mapper`를 적용할 대상. `index=mapper` 또는 `columns=mapper` 형태로 사용.
- `axis`: `mapper`를 적용할 축 ('rows' 또는 0, 'columns' 또는 1).
- `inplace`: 원본 DataFrame 직접 수정 여부.
""")
code_rename = """
import pandas as pd
# sample_df_transform DataFrame이 이미 있다고 가정

# 열 이름 변경
df_renamed_cols = sample_df_transform.rename(columns={
    'Score_Math': 'Math_Score',
    'Score_English': 'Eng_Score'
})
# display_dataframe_info(df_renamed_cols, "열 이름 변경 후", show_describe=False)

# 인덱스 이름 변경 (예시 DataFrame 인덱스가 0,1,2.. 라면 함수로 변경)
# current_index = sample_df_transform.index
# df_renamed_index = sample_df_transform.rename(index=lambda x: f"EMP_{x+100}")
# display_dataframe_info(df_renamed_index, "인덱스 이름 변경 후", show_describe=False)
"""
st.code(code_rename, language='python')

if st.checkbox("`.rename()` 예시 보기", key="rename_page"):
    st.write("원본 컬럼명:", list(sample_df_transform.columns))
    df_renamed_ex = sample_df_transform.rename(
        columns={'Name': 'Employee_Name', 'BirthDate': 'DOB', 'Salary': 'Annual_Salary'},
        index={0: 'emp_00', 1: 'emp_01'} # 예제 DataFrame 인덱스가 RangeIndex(0,1,2..)라고 가정
    )
    st.write("일부 컬럼명 및 인덱스명 변경 후:")
    display_dataframe_info(df_renamed_ex, "rename 결과", max_rows_to_display=5, show_describe=False)

    st.write("모든 컬럼명을 대문자로 변경 (함수 사용):")
    df_upper_cols_ex = sample_df_transform.rename(columns=str.upper)
    display_dataframe_info(df_upper_cols_ex, "컬럼명 대문자 변경", max_rows_to_display=5, show_describe=False, show_dtypes=False)


st.markdown("---")

# --- 7.3 인덱스 설정 및 리셋 (`.set_index()`, `.reset_index()`) ---
st.subheader("7.3 인덱스 설정 및 리셋 (`.set_index()`, `.reset_index()`)")
st.markdown("""
- **`.set_index(keys, drop=True, append=False, inplace=False)`**: 기존 열(들)을 DataFrame의 인덱스로 설정합니다.
  - `keys`: 인덱스로 사용할 열 이름 또는 열 이름의 리스트.
  - `drop`: 인덱스로 사용된 열을 기존 컬럼에서 삭제할지 여부 (기본값 `True`).
  - `append`: 기존 인덱스를 유지하고 새로운 인덱스를 추가할지 여부 (기본값 `False`, 멀티인덱스 생성).
- **`.reset_index(level=None, drop=False, inplace=False, col_level=0, col_fill='')`**: 현재 인덱스를 일반 열로 변환하고, 기본 정수 인덱스(RangeIndex)로 리셋합니다.
  - `drop`: 인덱스를 열로 추가하지 않고 그냥 삭제할지 여부 (기본값 `False`).
""")
code_set_reset_index = """
import pandas as pd
# sample_df_transform DataFrame이 이미 있다고 가정

# 'Name' 열을 인덱스로 설정
df_name_indexed = sample_df_transform.set_index('Name')
# display_dataframe_info(df_name_indexed, "'Name'을 인덱스로 설정", show_describe=False)

# 인덱스를 리셋하여 일반 열로 변환 (기존 인덱스는 'index' 열로 추가됨)
# df_reset = df_name_indexed.reset_index()
# display_dataframe_info(df_reset, "인덱스 리셋 후", show_describe=False)

# 인덱스를 리셋하되, 기존 인덱스는 삭제 (drop=True)
# df_reset_drop = df_name_indexed.reset_index(drop=True)
# display_dataframe_info(df_reset_drop, "인덱스 리셋 (기존 인덱스 삭제)", show_describe=False)
"""
st.code(code_set_reset_index, language='python')

if st.checkbox("`.set_index()` 및 `.reset_index()` 예시 보기", key="set_reset_index_page"):
    st.write("원본 DataFrame (sample_df_transform):")
    st.dataframe(sample_df_transform.head(3))

    st.write("`sample_df_transform.set_index('Name', drop=False)` ('Name'을 인덱스로, 기존 'Name' 열 유지):")
    df_set_idx_ex = sample_df_transform.set_index('Name', drop=False)
    display_dataframe_info(df_set_idx_ex, "set_index('Name', drop=False) 결과", max_rows_to_display=5, show_describe=False)

    st.write("`df_set_idx_ex.reset_index()` (인덱스 리셋, 기존 인덱스 'index' 열로):")
    # Name이 인덱스였으므로, reset_index하면 Name이 열로 돌아감
    df_reset_idx_ex = df_set_idx_ex.reset_index(drop=True) # drop=True로 이전 인덱스(Name)를 열로 안 만듦. 기존 RangeIndex가 복원됨.
                                                          # 만약 drop=False 였다면, 'Name' 인덱스가 'Name' 컬럼으로 돌아옴.
    display_dataframe_info(df_reset_idx_ex, "reset_index(drop=True) 결과", max_rows_to_display=5, show_describe=False)


st.markdown("---")

# --- 7.4 행/열 삭제 (`.drop()`) ---
st.subheader("7.4 행/열 삭제 (`.drop()`)")
st.markdown("""
지정한 레이블의 행 또는 열을 삭제합니다.
- `labels`: 삭제할 인덱스/컬럼 레이블 (단일 또는 리스트).
- `axis`: 삭제할 축 (0은 행, 1은 열. 기본값 0).
- `index` 또는 `columns`: 각각 행 또는 열 레이블을 직접 지정하여 삭제 가능 (`labels`와 `axis` 대신 사용).
- `inplace`: 원본 DataFrame 직접 수정 여부.
""")
code_drop = """
import pandas as pd
# sample_df_transform DataFrame이 이미 있다고 가정 (인덱스: 0,1,2...)

# 특정 행 삭제 (인덱스 레이블 0, 2)
df_dropped_rows = sample_df_transform.drop(index=[0, 2])
# display_dataframe_info(df_dropped_rows, "0, 2번 인덱스 행 삭제 후", show_describe=False)

# 특정 열 삭제 ('Salary' 열)
df_dropped_col = sample_df_transform.drop(columns=['Salary']) # 또는 .drop('Salary', axis=1)
# display_dataframe_info(df_dropped_col, "'Salary' 열 삭제 후", show_describe=False)
"""
st.code(code_drop, language='python')

if st.checkbox("`.drop()` 예시 보기", key="drop_page"):
    st.write("원본 DataFrame (sample_df_transform):")
    st.dataframe(sample_df_transform)

    # 예제 DataFrame의 인덱스는 0,1,2,3,4
    st.write("`sample_df_transform.drop(index=[1, 3])` (1, 3번 인덱스 행 삭제):")
    display_dataframe_info(sample_df_transform.drop(index=[1, 3]), "행 삭제 결과", max_rows_to_display=3, show_describe=False)

    st.write("`sample_df_transform.drop(columns=['BirthDate', 'Department'])` ('BirthDate', 'Department' 열 삭제):")
    display_dataframe_info(sample_df_transform.drop(columns=['BirthDate', 'Department']), "열 삭제 결과", max_rows_to_display=5, show_describe=False)


st.markdown("---")

# --- 7.5 중복 처리 (`.duplicated()`, `.drop_duplicates()`) ---
st.subheader("7.5 중복 처리 (`.duplicated()`, `.drop_duplicates()`)")
st.markdown("""
- **`.duplicated(subset=None, keep='first')`**: 각 행이 중복인지 여부를 불리언 Series로 반환합니다.
  - `subset`: 중복을 확인할 특정 열(들)의 리스트. 기본값은 모든 열.
  - `keep`: 중복된 값들 중 어떤 것을 `False`(중복 아님)로 표시할지 결정.
    - `'first'`: 첫 번째 등장한 값은 `False`, 나머지는 `True` (기본값).
    - `'last'`: 마지막에 등장한 값은 `False`, 나머지는 `True`.
    - `False`: 모든 중복된 값을 `True`로 표시.
- **`.drop_duplicates(subset=None, keep='first', inplace=False)`**: 중복된 행을 제거합니다. 파라미터는 `.duplicated()`와 유사.
""")
code_duplicates = """
import pandas as pd
# 예제용 중복 데이터 포함 DataFrame
data_dup = {
    'A': [1, 1, 2, 3, 2, 1],
    'B': ['x', 'x', 'y', 'z', 'y', 'x'],
    'C': [10, 10, 20, 30, 20, 10]
}
df_dup = pd.DataFrame(data_dup)

# 중복 행 확인 (모든 열 기준)
duplicates_bool = df_dup.duplicated() # [False, True, False, False, True, True]
# print("중복 행 여부 (모든 열 기준):\\n", duplicates_bool)

# 'A' 열 기준 중복 확인
duplicates_A = df_dup.duplicated(subset=['A'])
# print("\\n'A' 열 기준 중복 여부:\\n", duplicates_A)


# 중복 행 제거 (첫 번째 등장 값 유지)
df_no_duplicates_first = df_dup.drop_duplicates()
# display_dataframe_info(df_no_duplicates_first, "중복 제거 (keep='first')")

# 중복 행 제거 (마지막 등장 값 유지)
df_no_duplicates_last = df_dup.drop_duplicates(keep='last')
# display_dataframe_info(df_no_duplicates_last, "중복 제거 (keep='last')")

# 'A', 'B' 열 기준으로 중복 제거
df_no_duplicates_subset = df_dup.drop_duplicates(subset=['A', 'B'])
# display_dataframe_info(df_no_duplicates_subset, "['A', 'B'] 기준 중복 제거")
"""
st.code(code_duplicates, language='python')

if st.checkbox("중복 처리 예시 보기", key="duplicates_page"):
    data_dup_ex = {
        'Col1': ['apple', 'banana', 'apple', 'orange', 'banana', 'apple'],
        'Col2': [100, 200, 100, 300, 200, 100],
        'Col3': ['US', 'FR', 'US', 'JP', 'FR', 'CA']
    }
    df_dup_ex = pd.DataFrame(data_dup_ex)
    st.write("중복 포함 예제 DataFrame (`df_dup_ex`):")
    st.dataframe(df_dup_ex)

    st.write("`df_dup_ex.duplicated()` (전체 행 기준, 첫 번째 발생은 False):")
    st.dataframe(pd.concat([df_dup_ex, df_dup_ex.duplicated().rename('Is_Duplicate')], axis=1))

    st.write("`df_dup_ex.drop_duplicates()` (기본, 첫 번째 중복 유지):")
    display_dataframe_info(df_dup_ex.drop_duplicates(), "drop_duplicates() 결과", max_rows_to_display=6, show_describe=False)

    st.write("`df_dup_ex.drop_duplicates(subset=['Col1'], keep='last')` ('Col1' 기준 중복 제거, 마지막 값 유지):")
    display_dataframe_info(df_dup_ex.drop_duplicates(subset=['Col1'], keep='last'), "drop_duplicates(subset=['Col1'], keep='last') 결과", max_rows_to_display=6, show_describe=False)


st.markdown("---")

# --- 7.6 문자열 처리 (`Series.str`) ---
st.subheader("7.6 문자열 처리 (`Series.str`)")
st.markdown("""
문자열 타입의 Series는 `.str` 접근자를 통해 다양한 문자열 처리 메소드를 벡터화된 방식으로 사용할 수 있습니다.
예: `lower()`, `upper()`, `len()`, `strip()`, `replace()`, `contains()`, `startswith()`, `endswith()`, `split()`, `get()`, `extract()` 등.
""")
code_string_methods = """
import pandas as pd
# sample_df_transform DataFrame의 'Name' 열 사용 가정
# sample_df_transform['Name']은 ['  alice  ', ' BOB  ', 'Charlie', ...] 형태

# 소문자로 변환하고 양쪽 공백 제거
clean_names = sample_df_transform['Name'].str.lower().str.strip()
# print("정리된 이름:\\n", clean_names)

# 'City' 열에서 'o'를 포함하는지 여부 확인
contains_o = sample_df_transform['City'].str.contains('o', case=False) # 대소문자 무시
# print("\\n'City'에 'o' 포함 여부:\\n", contains_o)

# 'Name'을 첫 글자 기준으로 분리 (예시)
# 이름이 항상 공백으로 구분된 두 단어라고 가정하지 않으므로 주의
# first_initial = sample_df_transform['Name'].str.strip().str.split(' ', expand=True)[0].str[0]
# print("\\n이름 첫 글자 (추정):\\n", first_initial)
"""
st.code(code_string_methods, language='python')

if st.checkbox("문자열 처리 예시 보기", key="string_methods_page"):
    st.write("원본 'Name' 열:", sample_df_transform['Name'].values)
    
    names_lower_strip = sample_df_transform['Name'].str.lower().str.strip()
    st.write("`sample_df_transform['Name'].str.lower().str.strip()` (소문자 변환 및 공백 제거):")
    st.write(names_lower_strip.values)

    st.write("`sample_df_transform['City'].str.len()` (도시 이름 길이):")
    st.dataframe(pd.concat([sample_df_transform['City'], sample_df_transform['City'].str.len().rename('Length')], axis=1))
    
    st.write("`sample_df_transform['City'].str.startswith('P')` ('P'로 시작하는 도시):")
    st.dataframe(sample_df_transform[sample_df_transform['City'].str.startswith('P')])


st.markdown("---")

# --- 7.7 날짜/시간 처리 (`Series.dt`) ---
st.subheader("7.7 날짜/시간 처리 (`Series.dt`)")
st.markdown("""
datetime64 타입의 Series는 `.dt` 접근자를 통해 날짜 및 시간 관련 속성과 메소드를 사용할 수 있습니다.
먼저 `pd.to_datetime()` 함수로 문자열 등을 datetime 타입으로 변환해야 할 수 있습니다.
예: `dt.year`, `dt.month`, `dt.day`, `dt.hour`, `dt.minute`, `dt.second`, `dt.dayofweek` (월요일=0, 일요일=6), `dt.day_name()`, `dt.strftime('%Y-%m-%d')` (포맷팅).
""")
code_datetime_methods = """
import pandas as pd
# sample_df_transform DataFrame의 'BirthDate' 열 사용 가정 (이미 datetime 타입으로 변환됨)

# 연도 추출
years = sample_df_transform['BirthDate'].dt.year
# print("출생 연도:\\n", years)

# 월 이름 추출
month_names = sample_df_transform['BirthDate'].dt.month_name()
# print("\\n출생 월 이름:\\n", month_names)

# 요일 추출 (월요일=0, 일요일=6)
day_of_week = sample_df_transform['BirthDate'].dt.dayofweek
# print("\\n출생 요일 (0=월, 6=일):\\n", day_of_week)

# 현재 날짜로부터 나이 계산 (근사치)
# from datetime import datetime
# current_year = datetime.now().year
# approx_age = current_year - sample_df_transform['BirthDate'].dt.year
# print("\\n현재 기준 대략적인 나이:\\n", approx_age)
"""
st.code(code_datetime_methods, language='python')

if st.checkbox("날짜/시간 처리 예시 보기", key="datetime_methods_page"):
    st.write("원본 'BirthDate' 열 (`datetime64` 타입):")
    st.write(sample_df_transform['BirthDate'])

    datetime_features_df = pd.DataFrame({
        'OriginalDate': sample_df_transform['BirthDate'],
        'Year': sample_df_transform['BirthDate'].dt.year,
        'Month': sample_df_transform['BirthDate'].dt.month,
        'Day': sample_df_transform['BirthDate'].dt.day,
        'DayOfWeek': sample_df_transform['BirthDate'].dt.day_name(), # 요일 이름
        'WeekOfYear': sample_df_transform['BirthDate'].dt.isocalendar().week.astype(int) # 연중 몇 번째 주
    })
    display_dataframe_info(datetime_features_df, "날짜/시간 특징 추출 결과", max_rows_to_display=5, show_describe=False, show_dtypes=False)
    
    # 날짜 간 차이 계산
    today = pd.to_datetime('today').normalize() # 오늘 날짜 (시간 부분 00:00:00)
    sample_df_transform['DaysSinceBirth'] = (today - sample_df_transform['BirthDate']).dt.days
    st.write(f"오늘 날짜 ({today.strftime('%Y-%m-%d')}) 기준, 태어난 후 경과일수:")
    display_dataframe_info(sample_df_transform[['Name', 'BirthDate', 'DaysSinceBirth']], "경과일수 계산", max_rows_to_display=5, show_describe=False, show_dtypes=False)


st.markdown("---")

# --- 7.8 데이터 구간화/범주화 (`pd.cut()`, `pd.qcut()`) ---
st.subheader("7.8 데이터 구간화/범주화 (`pd.cut()`, `pd.qcut()`)")
st.markdown("""
연속형 데이터를 여러 구간(bin)으로 나누어 범주형 데이터로 변환합니다.
- **`pd.cut(x, bins, labels=None, right=True, ...)`**: 사용자가 직접 구간 경계를 지정합니다.
  - `x`: 구간화할 1차원 배열 또는 Series.
  - `bins`: 구간 경계 값의 리스트 또는 정수(구간 개수). 정수이면 데이터의 최소/최대값을 기준으로 등간격으로 나눔.
  - `labels`: 각 구간에 부여할 이름의 리스트 (선택 사항). `False`이면 정수 인덱스 반환.
  - `right`: 구간의 오른쪽 경계를 포함할지 여부 (기본값 `True`).
- **`pd.qcut(x, q, labels=None, ...)`**: 데이터를 분위수(quantile) 기준으로 나눕니다. 각 구간이 거의 동일한 개수의 데이터를 갖도록 합니다.
  - `q`: 구간의 개수(정수) 또는 분위수 경계 값의 리스트 (예: `[0, 0.25, 0.5, 0.75, 1]`).
""")
code_binning = """
import pandas as pd
import numpy as np
# sample_df_transform DataFrame의 'Age' 또는 'Salary' 열 사용 가정

# 'Salary'를 기준으로 구간 나누기
salary_bins = [50000, 70000, 90000, np.inf] # 구간 경계
salary_labels = ['Low', 'Medium', 'High']    # 각 구간 레이블
sample_df_transform['Salary_Group_Cut'] = pd.cut(
    sample_df_transform['Salary'],
    bins=salary_bins,
    labels=salary_labels,
    right=False # 왼쪽 경계 포함, 오른쪽 경계 미포함 (예: [50000, 70000) )
)
# print("Salary 구간화 (pd.cut):\\n", sample_df_transform[['Salary', 'Salary_Group_Cut']])


# 'Score_Math'를 3개 분위수 구간으로 나누기
sample_df_transform['Math_Quantile'] = pd.qcut(
    sample_df_transform['Score_Math'],
    q=3, # 3개 분위수 (0-33.3%, 33.3-66.6%, 66.6-100%)
    labels=['Low_Perf', 'Mid_Perf', 'High_Perf']
)
# print("\\nScore_Math 분위수 구간화 (pd.qcut):\\n", sample_df_transform[['Score_Math', 'Math_Quantile']])
"""
st.code(code_binning, language='python')

if st.checkbox("데이터 구간화 예시 보기", key="binning_page"):
    st.write("원본 'Salary' 열:", sample_df_transform['Salary'].values)
    salary_bins_ex = [0, 60000, 80000, 100000, np.inf] # 0~6만, 6만~8만, 8만~10만, 10만 초과
    salary_labels_ex = ['Entry', 'Junior', 'Mid', 'Senior']
    
    binned_salary_df = sample_df_transform.copy()
    binned_salary_df['Salary_Level_Cut'] = pd.cut(
        binned_salary_df['Salary'],
        bins=salary_bins_ex,
        labels=salary_labels_ex,
        right=False, # [min, max) 구간. 즉, min <= x < max
        include_lowest=True # 첫 번째 구간에 최소값 포함
    )
    display_dataframe_info(binned_salary_df[['Name','Salary', 'Salary_Level_Cut']], "pd.cut() 결과 (Salary 구간화)", max_rows_to_display=5, show_describe=False)

    st.markdown("---")
    st.write("원본 'Score_English' 열:", sample_df_transform['Score_English'].values)
    binned_salary_df['English_Quantile_Group'] = pd.qcut(
        binned_salary_df['Score_English'],
        q=4, # 4개 분위수 (사분위수)
        labels=['Q1', 'Q2', 'Q3', 'Q4'] # 각 분위수 구간 레이블
    )
    display_dataframe_info(binned_salary_df[['Name','Score_English', 'English_Quantile_Group']].sort_values('Score_English'),
                           "pd.qcut() 결과 (Score_English 4분위수 구간화)", max_rows_to_display=5, show_describe=False)