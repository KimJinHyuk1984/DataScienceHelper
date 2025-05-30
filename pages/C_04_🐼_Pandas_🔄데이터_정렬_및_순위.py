# pages/4_🔄_데이터_정렬_및_순위.py
import streamlit as st
import pandas as pd
import numpy as np
from utils.utils_pandas import display_dataframe_info

st.header("4. 데이터 정렬 및 순위 매기기")
st.markdown("""
Pandas DataFrame의 데이터를 특정 기준에 따라 정렬하거나, 값에 대한 순위를 매길 수 있습니다.
""")

# --- 예제 DataFrame 생성 ---
@st.cache_data
def create_sample_sort_rank_df():
    data = {
        'Name': ['Eve', 'David', 'Alice', 'Charlie', 'Bob', 'Frank'],
        'Age': [22, 28, 25, 35, 30, 28], # David와 Frank 나이 동일
        'City': ['Seoul', 'Tokyo', 'New York', 'London', 'Paris', 'Tokyo'], # Tokyo 중복
        'Score': [95.0, 88.2, 78.0, 95.0, 92.5, 81.0], # Eve와 Charlie 점수 동일
        'Experience': [1, 3, 2, 5, 4, 3] # David와 Frank 경력 동일
    }
    # 인덱스를 일부러 순서 없이 만듦
    return pd.DataFrame(data, index=['id5', 'id4', 'id1', 'id3', 'id2', 'id6'])

sample_df_sort = create_sample_sort_rank_df()

st.subheader("정렬/순위 예제용 DataFrame 확인")
if st.checkbox("정렬/순위 예제 DataFrame 보기", key="show_sort_base_df_page"):
    display_dataframe_info(sample_df_sort, "예제 DataFrame (sample_df_sort)", max_rows_to_display=6)

st.markdown("---")

# --- 4.1 인덱스 기준 정렬 (`.sort_index()`) ---
st.subheader("4.1 인덱스 기준 정렬 (`.sort_index()`)")
st.markdown("""
DataFrame 또는 Series의 인덱스(행 또는 열 레이블)를 기준으로 데이터를 정렬합니다.
- `axis`: 정렬할 축 (0은 행 인덱스, 1은 열 이름. 기본값 0).
- `ascending`: 오름차순 정렬 여부 (기본값 `True`). `False`이면 내림차순.
- `inplace`: 원본 DataFrame을 직접 수정할지 여부 (기본값 `False`, 새 DataFrame 반환).
""")
code_sort_index = """
import pandas as pd
# sample_df_sort DataFrame이 이미 있다고 가정

# 행 인덱스 기준 오름차순 정렬 (기본값)
df_sorted_by_row_index = sample_df_sort.sort_index()
# display_dataframe_info(df_sorted_by_row_index, "행 인덱스 기준 오름차순 정렬")

# 행 인덱스 기준 내림차순 정렬
df_sorted_by_row_index_desc = sample_df_sort.sort_index(ascending=False)
# display_dataframe_info(df_sorted_by_row_index_desc, "행 인덱스 기준 내림차순 정렬")

# 열 이름(컬럼) 기준 오름차순 정렬
df_sorted_by_col_index = sample_df_sort.sort_index(axis=1)
# display_dataframe_info(df_sorted_by_col_index, "열 이름 기준 오름차순 정렬")
"""
st.code(code_sort_index, language='python')

if st.checkbox("`.sort_index()` 예시 보기", key="sort_index_page"):
    st.write("원본 DataFrame (sample_df_sort):")
    st.dataframe(sample_df_sort)

    st.write("`sample_df_sort.sort_index()` (행 인덱스 오름차순):")
    display_dataframe_info(sample_df_sort.sort_index(), "행 인덱스 오름차순", max_rows_to_display=6, show_describe=False)

    st.write("`sample_df_sort.sort_index(ascending=False)` (행 인덱스 내림차순):")
    display_dataframe_info(sample_df_sort.sort_index(ascending=False), "행 인덱스 내림차순", max_rows_to_display=6, show_describe=False)

    st.write("`sample_df_sort.sort_index(axis=1)` (열 이름 오름차순):")
    display_dataframe_info(sample_df_sort.sort_index(axis=1), "열 이름 오름차순", max_rows_to_display=6, show_describe=False)

st.markdown("---")

# --- 4.2 값 기준 정렬 (`.sort_values()`) ---
st.subheader("4.2 값 기준 정렬 (`.sort_values()`)")
st.markdown("""
DataFrame 또는 Series의 특정 열(들)의 값을 기준으로 데이터를 정렬합니다.
- `by`: 정렬 기준으로 사용할 열 이름 또는 열 이름의 리스트.
- `axis`: 정렬할 축 (0은 행 기준 정렬 - 특정 열의 값을 보고 행 순서 변경, 1은 열 기준 정렬 - 특정 행의 값을 보고 열 순서 변경. 기본값 0).
- `ascending`: 오름차순 정렬 여부. `by`가 리스트이면, 각 열에 대한 오름차순/내림차순을 리스트로 지정 가능 (예: `[True, False]`). 기본값 `True`.
- `inplace`: 원본 DataFrame을 직접 수정할지 여부 (기본값 `False`).
- `na_position`: NaN 값의 위치 ('first' 또는 'last'. 기본값 'last').
""")
code_sort_values = """
import pandas as pd
# sample_df_sort DataFrame이 이미 있다고 가정

# 'Age' 열 기준 오름차순 정렬
df_sorted_by_age = sample_df_sort.sort_values(by='Age')
# display_dataframe_info(df_sorted_by_age, "'Age' 열 기준 오름차순 정렬")

# 'Score' 열 기준 내림차순 정렬, NaN 값을 맨 앞에 표시
df_sorted_by_score_desc = sample_df_sort.sort_values(by='Score', ascending=False, na_position='first')
# display_dataframe_info(df_sorted_by_score_desc, "'Score' 열 기준 내림차순 (NaN 맨 앞)")

# 여러 열 기준 정렬: 'City' 오름차순 후, 'Age' 내림차순
df_sorted_by_city_age = sample_df_sort.sort_values(by=['City', 'Age'], ascending=[True, False])
# display_dataframe_info(df_sorted_by_city_age, "'City'(오름차순) 후 'Age'(내림차순) 정렬")
"""
st.code(code_sort_values, language='python')

if st.checkbox("`.sort_values()` 예시 보기", key="sort_values_page"):
    st.write("원본 DataFrame (sample_df_sort):")
    st.dataframe(sample_df_sort)

    st.write("`sample_df_sort.sort_values(by='Age')` ('Age' 오름차순):")
    display_dataframe_info(sample_df_sort.sort_values(by='Age'), "'Age' 오름차순", max_rows_to_display=6, show_describe=False)

    st.write("`sample_df_sort.sort_values(by='Score', ascending=False, na_position='first')` ('Score' 내림차순, NaN 맨 앞):")
    # 예제 데이터에 Score NaN이 없으므로, 하나 추가해서 시연
    temp_df_for_nan_sort = sample_df_sort.copy()
    temp_df_for_nan_sort.loc['id_nan_score'] = ['NaNTest', 30, 'TestCity', np.nan, 2]
    st.write("NaN 포함 임시 DataFrame:")
    st.dataframe(temp_df_for_nan_sort)
    display_dataframe_info(temp_df_for_nan_sort.sort_values(by='Score', ascending=False, na_position='first'),
                           "'Score' 내림차순 (NaN 맨 앞)", max_rows_to_display=7, show_describe=False)


    st.write("`sample_df_sort.sort_values(by=['City', 'Experience'], ascending=[True, False])` ('City' 오름차순, 그 안에서 'Experience' 내림차순):")
    display_dataframe_info(sample_df_sort.sort_values(by=['City', 'Experience'], ascending=[True, False]),
                           "'City' 오름차순, 'Experience' 내림차순", max_rows_to_display=6, show_describe=False)

st.markdown("---")

# --- 4.3 순위 매기기 (`.rank()`) ---
st.subheader("4.3 순위 매기기 (`.rank()`)")
st.markdown("""
Series의 각 값에 대해 순위를 계산합니다. DataFrame에 적용하려면 각 열 Series에 대해 개별적으로 적용해야 합니다.
- `axis`: 순위를 계산할 축 (Series는 항상 0).
- `method`: 동점자 처리 방법:
  - `'average'`: 동점자들의 평균 순위 (기본값).
  - `'min'`: 동점자 그룹 내 가장 낮은 순위.
  - `'max'`: 동점자 그룹 내 가장 높은 순위.
  - `'first'`: 데이터에 나타난 순서대로 순위 부여.
  - `'dense'`: `'min'`과 유사하지만, 그룹 간 순위가 1씩 증가 (즉, 순위가 건너뛰지 않음).
- `ascending`: 오름차순 순위 여부 (기본값 `True`, 작은 값이 높은 순위). `False`이면 큰 값이 높은 순위.
- `na_option`: NaN 값 처리 방법 (`'keep'`, `'top'`, `'bottom'`. 기본값 `'keep'`, NaN은 NaN 순위).
- `pct`: 순위를 백분위로 표시할지 여부 (기본값 `False`).
""")
code_rank = """
import pandas as pd
# sample_df_sort DataFrame이 이미 있다고 가정

# 'Score' 열에 대해 순위 매기기 (기본값: 오름차순, 동점자는 평균 순위)
score_rank_avg = sample_df_sort['Score'].rank()
# print("Score 순위 (average, ascending):\\n", score_rank_avg)

# 'Score' 열에 대해 내림차순 순위 (높은 점수가 높은 순위), 동점자는 min 사용
score_rank_desc_min = sample_df_sort['Score'].rank(method='min', ascending=False)
# print("\\nScore 순위 (min, descending):\\n", score_rank_desc_min)

# 'Age' 열에 대해 'dense' 방식으로 순위 (오름차순)
age_rank_dense = sample_df_sort['Age'].rank(method='dense')
# print("\\nAge 순위 (dense, ascending):\\n", age_rank_dense)

# DataFrame 전체에 rank 적용 (각 열별로 순위 계산)
df_ranked = sample_df_sort.rank(method='first') # 각 열을 독립적으로 순위 매김
# display_dataframe_info(df_ranked, "DataFrame 전체 순위 (각 열별, method='first')")
"""
st.code(code_rank, language='python')

if st.checkbox("`.rank()` 예시 보기", key="rank_page"):
    st.write("원본 DataFrame (sample_df_sort)의 'Score'와 'Age' 열:")
    st.dataframe(sample_df_sort[['Name', 'Score', 'Age']])

    st.write("`sample_df_sort['Score'].rank(method='average', ascending=False)` (Score 내림차순, 동점자 평균 순위):")
    score_rank_ex = sample_df_sort['Score'].rank(method='average', ascending=False)
    st.write(pd.concat([sample_df_sort['Score'], score_rank_ex.rename('Rank_Avg_Desc')], axis=1))
    # Eve (95.0) 와 Charlie (95.0)는 공동 1등이므로 (1+2)/2 = 1.5위

    st.write("`sample_df_sort['Score'].rank(method='min', ascending=False)` (Score 내림차순, 동점자 낮은 순위):")
    score_rank_min_ex = sample_df_sort['Score'].rank(method='min', ascending=False)
    st.write(pd.concat([sample_df_sort['Score'], score_rank_min_ex.rename('Rank_Min_Desc')], axis=1))
    # Eve (95.0) 와 Charlie (95.0)는 공동 1등이므로 둘 다 1위, 다음 순위는 3위

    st.write("`sample_df_sort['Age'].rank(method='dense', ascending=True)` (Age 오름차순, 동점자 dense 순위):")
    age_rank_dense_ex = sample_df_sort['Age'].rank(method='dense', ascending=True)
    st.write(pd.concat([sample_df_sort['Age'], age_rank_dense_ex.rename('Rank_Dense_Asc')], axis=1).sort_values(by='Age'))
    # David (28)와 Frank (28)는 Age가 같으므로 같은 순위, 다음 순위는 바로 이어짐

    st.write("`sample_df_sort[['Score', 'Age']].rank(method='first', pct=True)` (Score, Age 열 백분위 순위, 나타난 순서대로):")
    # pct=True는 순위를 백분율로 표시
    df_ranked_pct_ex = sample_df_sort[['Score', 'Age']].rank(method='first', pct=True)
    st.dataframe(pd.concat([sample_df_sort[['Name', 'Score', 'Age']],
                            df_ranked_pct_ex.add_suffix('_pct_rank')], axis=1))