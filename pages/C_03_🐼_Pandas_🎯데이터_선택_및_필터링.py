# pages/3_🎯_데이터_선택_및_필터링.py
import streamlit as st
import pandas as pd
import numpy as np
from utils.utils_pandas import display_dataframe_info

st.header("3. 데이터 선택 및 필터링")
st.markdown("""
Pandas DataFrame에서 원하는 데이터를 효과적으로 선택하고 필터링하는 것은 데이터 분석의 핵심 단계입니다.
다양한 방법을 통해 특정 행, 열, 또는 조건을 만족하는 데이터를 추출할 수 있습니다.
""")

# --- 예제 DataFrame 생성 ---
@st.cache_data
def create_sample_selection_df():
    data = {
        'Name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve', 'Frank', 'Grace'],
        'Age': [25, 30, 35, 28, 22, 45, 29],
        'City': ['New York', 'Paris', 'London', 'Tokyo', 'Seoul', 'Berlin', 'Paris'],
        'Score': [85.0, 92.5, 78.0, np.nan, 95.0, 88.2, 76.9],
        'Registered': [True, False, True, True, False, True, True]
    }
    return pd.DataFrame(data, index=[f'ID{i+1}' for i in range(len(data['Name']))])

sample_df_select = create_sample_selection_df()

st.subheader("선택/필터링 예제용 DataFrame 확인")
if st.checkbox("선택/필터링 예제 DataFrame 보기", key="show_select_base_df_page"):
    display_dataframe_info(sample_df_select, "예제 DataFrame (sample_df_select)", max_rows_to_display=7)

st.markdown("---")

# --- 3.1 열 선택 ---
st.subheader("3.1 열 선택 (Column Selection)")
st.markdown("""
- `df['column_name']`: 단일 열을 Series로 선택.
- `df.column_name`: 단일 열을 Series로 선택 (점 표기법, 컬럼 이름이 파이썬 변수명 규칙에 맞고 공백이 없을 때 가능).
- `df[['col1', 'col2']]`: 여러 열을 DataFrame으로 선택 (리스트 형태로 전달).
""")
code_select_columns = """
import pandas as pd
# sample_df_select DataFrame이 이미 있다고 가정

# 'Name' 열 선택 (Series 반환)
name_series = sample_df_select['Name']
# print("Name 열 (Series):\\n", name_series)

# 'Age' 열 선택 (점 표기법, Series 반환)
age_series = sample_df_select.Age
# print("\\nAge 열 (Series, 점 표기법):\\n", age_series)

# 'Name'과 'City' 열 선택 (DataFrame 반환)
name_city_df = sample_df_select[['Name', 'City']]
# print("\\nName, City 열 (DataFrame):\\n", name_city_df)
"""
st.code(code_select_columns, language='python')

if st.checkbox("열 선택 예시 보기", key="select_columns_page"):
    st.write("`sample_df_select['Name']` (Series):")
    st.write(sample_df_select['Name'])
    st.markdown("---")
    st.write("`sample_df_select.Score` (Series, 점 표기법):")
    st.write(sample_df_select.Score)
    st.markdown("---")
    st.write("`sample_df_select[['Age', 'City', 'Registered']]` (DataFrame):")
    display_dataframe_info(sample_df_select[['Age', 'City', 'Registered']], "Age, City, Registered 열", max_rows_to_display=7, show_describe=False, show_dtypes=False)

st.markdown("---")

# --- 3.2 행 선택 (`.loc[]`, `.iloc[]`) ---
st.subheader("3.2 행 선택 (`.loc[]`, `.iloc[]`)")
st.markdown("""
- **`.loc[]` (Label-based selection):** 레이블(인덱스 이름, 컬럼 이름)을 기반으로 데이터 선택.
  - `df.loc['index_label']`: 특정 레이블의 행 선택 (Series 반환).
  - `df.loc[['label1', 'label2']]`: 여러 레이블의 행 선택 (DataFrame 반환).
  - `df.loc['start_label':'end_label']`: 레이블 범위로 행 슬라이싱 (end_label 포함).
  - `df.loc[row_labels, column_labels]`: 특정 행과 열을 레이블로 선택.
- **`.iloc[]` (Integer position-based selection):** 정수 위치(0부터 시작하는 인덱스)를 기반으로 데이터 선택.
  - `df.iloc[0]`: 첫 번째 행 선택 (Series 반환).
  - `df.iloc[[0, 2, 4]]`: 특정 위치의 여러 행 선택 (DataFrame 반환).
  - `df.iloc[0:3]`: 위치 범위로 행 슬라이싱 (end_index 미포함, 파이썬 슬라이싱과 동일).
  - `df.iloc[row_positions, column_positions]`: 특정 위치의 행과 열을 정수로 선택.
""")
code_select_rows = """
import pandas as pd
# sample_df_select DataFrame이 이미 있다고 가정 (인덱스: 'ID1', 'ID2', ...)

# --- .loc[] 사용 예시 ---
# 'ID2' 행 선택
row_id2 = sample_df_select.loc['ID2']
# print("ID2 행 (Series):\\n", row_id2)

# 'ID1'부터 'ID3'까지 행 슬라이싱
rows_id1_to_id3 = sample_df_select.loc['ID1':'ID3']
# print("\\nID1~ID3 행 (DataFrame):\\n", rows_id1_to_id3)

# 'ID1' 행의 'Name'과 'Age' 열 선택
id1_name_age = sample_df_select.loc['ID1', ['Name', 'Age']]
# print("\\nID1 행의 Name, Age (Series):\\n", id1_name_age)


# --- .iloc[] 사용 예시 ---
# 첫 번째 행 (인덱스 0) 선택
first_row = sample_df_select.iloc[0]
# print("\\n첫 번째 행 (Series):\\n", first_row)

# 0, 2, 4 위치의 행 선택
selected_pos_rows = sample_df_select.iloc[[0, 2, 4]]
# print("\\n0, 2, 4 위치 행 (DataFrame):\\n", selected_pos_rows)

# 첫 3개 행 슬라이싱 (0, 1, 2 위치)
first_3_rows = sample_df_select.iloc[0:3]
# print("\\n첫 3개 행 (DataFrame):\\n", first_3_rows)

# 0행의 0, 2 위치 열 선택
row0_col02 = sample_df_select.iloc[0, [0, 2]]
# print("\\n0행, 0번/2번 열 (Series):\\n", row0_col02)
"""
st.code(code_select_rows, language='python')

if st.checkbox("`.loc[]` 및 `.iloc[]` 예시 보기", key="loc_iloc_page"):
    st.markdown("#### `.loc[]` (Label-based) 예시")
    st.write("`sample_df_select.loc['ID3']`:")
    st.write(sample_df_select.loc['ID3'])
    st.markdown("---")
    st.write("`sample_df_select.loc[['ID1', 'ID4', 'ID6']]`:")
    display_dataframe_info(sample_df_select.loc[['ID1', 'ID4', 'ID6']], "ID1, ID4, ID6 행", max_rows_to_display=3, show_describe=False)
    st.markdown("---")
    st.write("`sample_df_select.loc['ID2':'ID5', ['Name', 'Score']]`:") # 레이블 슬라이싱은 끝점 포함
    display_dataframe_info(sample_df_select.loc['ID2':'ID5', ['Name', 'Score']], "ID2-ID5 행의 Name, Score 열", max_rows_to_display=4, show_describe=False)

    st.markdown("#### `.iloc[]` (Integer position-based) 예시")
    st.write("`sample_df_select.iloc[1]` (두 번째 행):")
    st.write(sample_df_select.iloc[1])
    st.markdown("---")
    st.write("`sample_df_select.iloc[[1, 3, 5]]` (1,3,5번 위치 행):")
    display_dataframe_info(sample_df_select.iloc[[1, 3, 5]], "1,3,5 위치 행", max_rows_to_display=3, show_describe=False)
    st.markdown("---")
    st.write("`sample_df_select.iloc[1:4, 0:2]` (1-3번 위치 행, 0-1번 위치 열):") # 정수 슬라이싱은 끝점 미포함
    display_dataframe_info(sample_df_select.iloc[1:4, 0:2], "1-3번 행, 0-1번 열", max_rows_to_display=3, show_describe=False)

st.markdown("---")

# --- 3.3 불리언 인덱싱 (Boolean Indexing) ---
st.subheader("3.3 불리언 인덱싱 (Boolean Indexing / Conditional Selection)")
st.markdown("""
조건식을 사용하여 `True`/`False` 값으로 이루어진 불리언 Series/배열을 만들고, 이를 인덱스로 사용하여 `True`에 해당하는 행만 선택합니다.
- `df[boolean_condition]`: `boolean_condition`은 `df['col'] > value` 와 같은 형태.
- 여러 조건 조합: `&` (AND), `|` (OR), `~` (NOT) 연산자 사용. 각 조건은 `()`로 묶어야 함.
- `isin()`: 특정 값 목록에 포함되는지 여부.
- `str.contains()`: 문자열 컬럼에서 특정 부분 문자열 포함 여부. (자세한 내용은 '데이터 변환' 페이지에서)
""")
code_boolean_indexing_pd = """
import pandas as pd
# sample_df_select DataFrame이 이미 있다고 가정

# 'Age'가 30 이상인 행 선택
age_over_30 = sample_df_select[sample_df_select['Age'] >= 30]
# display_dataframe_info(age_over_30, "Age >= 30 인 행")

# 'City'가 'Paris'인 행 선택
city_paris = sample_df_select[sample_df_select['City'] == 'Paris']
# display_dataframe_info(city_paris, "City == 'Paris' 인 행")

# 여러 조건: 'Age'가 25 미만이거나 'Score'가 90 이상인 행
condition_or = sample_df_select[(sample_df_select['Age'] < 25) | (sample_df_select['Score'] >= 90)]
# display_dataframe_info(condition_or, "(Age < 25) OR (Score >= 90) 인 행")

# isin() 사용: 'City'가 'Tokyo' 또는 'Seoul'인 행
city_tokyo_seoul = sample_df_select[sample_df_select['City'].isin(['Tokyo', 'Seoul'])]
# display_dataframe_info(city_tokyo_seoul, "City가 'Tokyo' 또는 'Seoul'인 행")

# notnull() 사용: 'Score'가 결측치(NaN)가 아닌 행
score_not_null = sample_df_select[sample_df_select['Score'].notnull()]
# display_dataframe_info(score_not_null, "Score가 NaN이 아닌 행")
"""
st.code(code_boolean_indexing_pd, language='python')

if st.checkbox("불리언 인덱싱 예시 보기", key="boolean_indexing_page_pd"):
    st.write("`sample_df_select[sample_df_select['Score'] < 80]` (Score가 80 미만):")
    display_dataframe_info(sample_df_select[sample_df_select['Score'] < 80], "Score < 80", max_rows_to_display=7, show_describe=False)
    st.markdown("---")
    st.write("`sample_df_select[(sample_df_select['Age'] > 25) & (sample_df_select['Registered'] == True)]` (Age > 25 AND Registered):")
    display_dataframe_info(sample_df_select[(sample_df_select['Age'] > 25) & (sample_df_select['Registered'] == True)], "Age > 25 AND Registered", max_rows_to_display=7, show_describe=False)
    st.markdown("---")
    cities_to_check = ['London', 'Berlin']
    st.write(f"`sample_df_select[sample_df_select['City'].isin({cities_to_check})]` (City가 London 또는 Berlin):")
    display_dataframe_info(sample_df_select[sample_df_select['City'].isin(cities_to_check)], f"City in {cities_to_check}", max_rows_to_display=7, show_describe=False)
    st.markdown("---")
    st.write("`sample_df_select[sample_df_select['Score'].isnull()]` (Score가 결측치인 행):")
    display_dataframe_info(sample_df_select[sample_df_select['Score'].isnull()], "Score is NaN", max_rows_to_display=7, show_describe=False)

st.markdown("---")

# --- 3.4 .query() 메소드 ---
st.subheader("3.4 `.query()` 메소드")
st.markdown("""
문자열 형태로 조건을 표현하여 데이터를 필터링합니다. 불리언 인덱싱보다 가독성이 좋을 수 있습니다.
내부적으로 `pd.eval()`을 사용하므로, 대용량 데이터에서는 성능 저하가 있을 수 있습니다. 컬럼 이름에 공백이나 특수문자가 있으면 백틱(\`\`)으로 감싸야 합니다.
""")
code_query_method = """
import pandas as pd
# sample_df_select DataFrame이 이미 있다고 가정

# 'Age'가 30 이상인 행 선택
age_over_30_query = sample_df_select.query('Age >= 30')
# display_dataframe_info(age_over_30_query, "query('Age >= 30') 결과")

# 여러 조건: 'City'가 'Paris'이고 'Score'가 90 이상인 행
# (컬럼 이름에 공백이 없다면 백틱 불필요)
complex_query = sample_df_select.query("City == 'Paris' and Score >= 90")
# display_dataframe_info(complex_query, "query('City == \\'Paris\\' and Score >= 90') 결과")

# 외부 변수 참조 (변수명 앞에 @ 사용)
min_age = 25
max_score = 80
variable_query = sample_df_select.query('Age > @min_age and Score < @max_score')
# display_dataframe_info(variable_query, "query('Age > @min_age and Score < @max_score') 결과")
"""
st.code(code_query_method, language='python')

if st.checkbox("`.query()` 메소드 예시 보기", key="query_method_page"):
    st.write("`sample_df_select.query('Age < 28 and Registered == True')`:")
    display_dataframe_info(sample_df_select.query('Age < 28 and Registered == True'), "query 결과 1", max_rows_to_display=7, show_describe=False)
    st.markdown("---")
    target_city = 'New York'
    min_score_val = 80.0
    st.write(f"`sample_df_select.query('City == @target_city or Score > @min_score_val')` (외부 변수 target_city='{target_city}', min_score_val={min_score_val}):")
    display_dataframe_info(sample_df_select.query('City == @target_city or Score > @min_score_val'), "query 결과 2 (외부 변수 사용)", max_rows_to_display=7, show_describe=False)


st.markdown("---")
st.markdown("단일 값에 빠르게 접근하기 위한 `.at[]` (레이블 기반) 및 `.iat[]` (정수 위치 기반) 메소드도 있습니다. 이는 `loc`/`iloc`보다 스칼라 값 접근에 더 빠릅니다.")