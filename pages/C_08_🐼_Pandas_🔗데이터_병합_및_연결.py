# pages/8_🔗_데이터_병합_및_연결.py
import streamlit as st
import pandas as pd
import numpy as np
from utils.utils_pandas import display_dataframe_info

st.header("8. 데이터 병합 및 연결")
st.markdown("""
여러 개의 DataFrame을 하나로 합치는 다양한 방법을 제공합니다.
주요 방법으로는 `pd.concat()` (연결), `pd.merge()` (병합), `DataFrame.join()` (병합)이 있습니다.
""")

# --- 예제 DataFrame 생성 ---
@st.cache_data
def create_sample_merge_dfs():
    df1 = pd.DataFrame({
        'ID': ['A01', 'A02', 'A03', 'A04'],
        'Name': ['Alice', 'Bob', 'Charlie', 'David'],
        'DeptID': [10, 20, 10, 30]
    })
    df2 = pd.DataFrame({
        'ID': ['A03', 'A04', 'A05', 'A06'],
        'Salary': [70000, 85000, 60000, 92000],
        'City': ['London', 'Tokyo', 'Paris', 'Berlin']
    })
    df_dept = pd.DataFrame({
        'DeptID': [10, 20, 30, 40],
        'DeptName': ['HR', 'IT', 'Sales', 'Marketing'],
        'Location': ['NY', 'SF', 'LDN', 'BER']
    })
    # concat 예제용 df
    df_concat1 = pd.DataFrame({'A': ['A0', 'A1'], 'B': ['B0', 'B1']})
    df_concat2 = pd.DataFrame({'A': ['A2', 'A3'], 'B': ['B2', 'B3']})
    df_concat3 = pd.DataFrame({'C': ['C0', 'C1'], 'D': ['D0', 'D1']}, index=[0,1]) # 다른 컬럼, 같은 인덱스
    return df1, df2, df_dept, df_concat1, df_concat2, df_concat3

df1_merge, df2_merge, df_dept_merge, df_c1, df_c2, df_c3 = create_sample_merge_dfs()

st.subheader("병합/연결 예제용 DataFrame 확인")
if st.checkbox("병합/연결 예제 DataFrame 보기", key="show_merge_base_dfs_page"):
    display_dataframe_info(df1_merge, "DataFrame 1 (df1_merge)", max_rows_to_display=4)
    display_dataframe_info(df2_merge, "DataFrame 2 (df2_merge)", max_rows_to_display=4)
    display_dataframe_info(df_dept_merge, "Department DataFrame (df_dept_merge)", max_rows_to_display=4)
    st.markdown("---")
    display_dataframe_info(df_c1, "Concat DataFrame 1 (df_c1)", max_rows_to_display=2)
    display_dataframe_info(df_c2, "Concat DataFrame 2 (df_c2)", max_rows_to_display=2)
    display_dataframe_info(df_c3, "Concat DataFrame 3 (df_c3)", max_rows_to_display=2)

st.markdown("---")

# --- 8.1 데이터 연결 (`pd.concat()`) ---
st.subheader("8.1 데이터 연결 (`pd.concat()`)")
st.markdown("""
여러 DataFrame을 특정 축(행 또는 열)을 따라 단순 연결합니다.
- `objs`: 연결할 DataFrame 객체들의 리스트.
- `axis`: 연결할 축 (0은 행 방향으로 아래에 연결 - 기본값, 1은 열 방향으로 옆에 연결).
- `join`: 다른 축의 인덱스/컬럼 처리 방법.
  - `'outer'`: 합집합 (기본값, 없는 값은 NaN).
  - `'inner'`: 교집합.
- `ignore_index`: 연결 후 인덱스를 새로 만들지(0, 1, 2...) 여부 (기본값 `False`).
- `keys`: 다중 인덱스(hierarchical index)를 생성하여 각 DataFrame을 구분할 수 있도록 키를 부여.
""")
code_concat = """
import pandas as pd
# df_c1, df_c2, df_c3 DataFrame이 이미 있다고 가정

# 행 방향으로 연결 (기본값 axis=0)
df_row_concat = pd.concat([df_c1, df_c2])
# display_dataframe_info(df_row_concat, "행 방향 연결 (기본)")

# 행 방향 연결 + 인덱스 무시
df_row_concat_ignore = pd.concat([df_c1, df_c2], ignore_index=True)
# display_dataframe_info(df_row_concat_ignore, "행 방향 연결 (ignore_index=True)")

# 열 방향으로 연결 (axis=1), 인덱스 기준
# df_c1과 df_c3는 같은 인덱스(0,1)를 가짐
df_col_concat = pd.concat([df_c1, df_c3], axis=1)
# display_dataframe_info(df_col_concat, "열 방향 연결 (axis=1)")

# 열 방향 연결 (join='inner', 공통 인덱스만)
# df_c1의 인덱스를 [0,2]로 변경하여 df_c3와 공통 인덱스 줄이기
# df_c1_mod = df_c1.copy()
# df_c1_mod.index = [0,2]
# df_col_concat_inner = pd.concat([df_c1_mod, df_c3], axis=1, join='inner')
# display_dataframe_info(df_col_concat_inner, "열 방향 연결 (join='inner')")
"""
st.code(code_concat, language='python')

if st.checkbox("`pd.concat()` 예시 보기", key="concat_page"):
    st.write("`pd.concat([df_c1, df_c2])` (행 방향 연결, 기본):")
    display_dataframe_info(pd.concat([df_c1, df_c2]), "행 방향 연결", max_rows_to_display=4, show_describe=False)

    st.write("`pd.concat([df_c1, df_c2], ignore_index=True)` (행 방향 연결, 인덱스 리셋):")
    display_dataframe_info(pd.concat([df_c1, df_c2], ignore_index=True), "행 방향 연결 (인덱스 리셋)", max_rows_to_display=4, show_describe=False)

    st.write("`pd.concat([df_c1, df_c3], axis=1)` (열 방향 연결, outer join):")
    display_dataframe_info(pd.concat([df_c1, df_c3], axis=1), "열 방향 연결 (outer)", max_rows_to_display=2, show_describe=False)

    df_c1_mod_idx = df_c1.copy()
    df_c1_mod_idx.index = [0, 2] # df_c1의 인덱스를 0, 2로 변경 (df_c3는 0, 1)
    st.write("`pd.concat([df_c1_mod_idx, df_c3], axis=1, join='inner')` (열 방향, inner join, 공통 인덱스 0만):")
    display_dataframe_info(pd.concat([df_c1_mod_idx, df_c3], axis=1, join='inner'), "열 방향 연결 (inner)", max_rows_to_display=2, show_describe=False)


st.markdown("---")

# --- 8.2 데이터 병합 (`pd.merge()`) ---
st.subheader("8.2 데이터 병합 (`pd.merge()`) - SQL 스타일 조인")
st.markdown("""
하나 이상의 공통된 열(키)을 기준으로 두 DataFrame을 SQL의 JOIN처럼 병합합니다.
- `left`, `right`: 병합할 두 DataFrame.
- `how`: 병합 방법.
  - `'inner'`: 양쪽 DataFrame에 모두 키가 존재하는 경우만 (교집합, 기본값).
  - `'outer'`: 양쪽 DataFrame의 모든 키를 포함 (합집합, 없는 값은 NaN).
  - `'left'`: 왼쪽 DataFrame의 모든 키를 포함.
  - `'right'`: 오른쪽 DataFrame의 모든 키를 포함.
- `on`: 병합 기준으로 사용할 공통 열 이름 (양쪽 DataFrame에 같은 이름으로 존재).
- `left_on`, `right_on`: 왼쪽/오른쪽 DataFrame에서 각각 병합 기준으로 사용할 열 이름 (열 이름이 다를 경우).
- `left_index=True`, `right_index=True`: 인덱스를 병합 기준으로 사용할지 여부.
- `suffixes`: 양쪽 DataFrame에 같은 이름의 열(키 제외)이 있을 경우, 구분하기 위해 붙일 접미사 (예: `('_L', '_R')`).
""")
code_merge = """
import pandas as pd
# df1_merge, df2_merge, df_dept_merge DataFrame이 이미 있다고 가정

# 'ID' 열 기준 inner join (기본)
merged_inner = pd.merge(df1_merge, df2_merge, on='ID')
# display_dataframe_info(merged_inner, "'ID' 기준 Inner Join")

# 'ID' 열 기준 outer join
merged_outer = pd.merge(df1_merge, df2_merge, on='ID', how='outer')
# display_dataframe_info(merged_outer, "'ID' 기준 Outer Join")

# 'ID' 열 기준 left join
merged_left = pd.merge(df1_merge, df2_merge, on='ID', how='left')
# display_dataframe_info(merged_left, "'ID' 기준 Left Join")

# df1_merge와 df_dept_merge를 'DeptID' 기준으로 병합 (left join)
# df1_merge에는 DeptID, df_dept_merge에는 DeptID, DeptName, Location
merged_with_dept = pd.merge(df1_merge, df_dept_merge, on='DeptID', how='left')
# display_dataframe_info(merged_with_dept, "직원 정보와 부서 정보 Left Join")
"""
st.code(code_merge, language='python')

if st.checkbox("`pd.merge()` 예시 보기", key="merge_page"):
    st.write("`pd.merge(df1_merge, df2_merge, on='ID', how='inner')` (Inner Join):")
    display_dataframe_info(pd.merge(df1_merge, df2_merge, on='ID', how='inner'), "Inner Join 결과", max_rows_to_display=4, show_describe=False)

    st.write("`pd.merge(df1_merge, df2_merge, on='ID', how='outer', suffixes=('_emp', '_info'))` (Outer Join, 중복컬럼 접미사):")
    # df1, df2에 ID 외 중복 컬럼이 없지만, 예시를 위해 추가
    # 현재는 중복되는 non-key 컬럼이 없으므로 suffixes는 효과가 없음.
    # 만약 df1과 df2에 'Note'라는 컬럼이 둘 다 있었다면 suffixes가 작용.
    display_dataframe_info(pd.merge(df1_merge, df2_merge, on='ID', how='outer', suffixes=('_emp', '_info')),
                           "Outer Join 결과", max_rows_to_display=6, show_describe=False)

    st.write("`pd.merge(df1_merge, df2_merge, on='ID', how='left')` (Left Join):")
    display_dataframe_info(pd.merge(df1_merge, df2_merge, on='ID', how='left'), "Left Join 결과", max_rows_to_display=4, show_describe=False)

    st.write("`pd.merge(df1_merge, df_dept_merge, on='DeptID', how='left')` (직원 정보 + 부서 정보 Left Join):")
    display_dataframe_info(pd.merge(df1_merge, df_dept_merge, on='DeptID', how='left'), "직원-부서 Left Join 결과", max_rows_to_display=4, show_describe=False)


st.markdown("---")

# --- 8.3 DataFrame.join() ---
st.subheader("8.3 `DataFrame.join()`")
st.markdown("""
`DataFrame.join()` 메소드는 주로 인덱스를 기준으로 다른 DataFrame 또는 Series들을 병합할 때 사용됩니다. `pd.merge()`의 편리한 래퍼(wrapper)로 볼 수 있으며, `left.join(right)` 형태로 호출합니다.
- `other`: 조인할 단일 DataFrame 또는 DataFrame의 리스트.
- `on`: 호출하는 DataFrame(`left`)에서 조인 키로 사용할 열(들). 생략 시 인덱스 사용.
- `how`: `'left'`, `'right'`, `'outer'`, `'inner'` (기본값 `'left'`).
- `lsuffix`, `rsuffix`: 중복되는 열 이름에 붙일 접미사.
""")
code_join = """
import pandas as pd
# df1_merge (인덱스: RangeIndex, 'ID' 컬럼 있음)
# df2_merge (인덱스: RangeIndex, 'ID' 컬럼 있음)
# df_dept_merge (인덱스: RangeIndex, 'DeptID' 컬럼 있음)

# df1_merge의 'ID'를 인덱스로 설정, df2_merge의 'ID'를 인덱스로 설정 후 조인
df1_idx = df1_merge.set_index('ID')
df2_idx = df2_merge.set_index('ID')

joined_df = df1_idx.join(df2_idx, how='inner', lsuffix='_from_df1', rsuffix='_from_df2')
# display_dataframe_info(joined_df, "df1_idx.join(df2_idx, how='inner') 결과")


# df1_merge의 'DeptID'와 df_dept_merge의 인덱스(DeptID로 설정)를 기준으로 조인
df_dept_idx = df_dept_merge.set_index('DeptID')
# df1_merge의 'DeptID'를 키로 사용하고, df_dept_idx의 인덱스를 키로 사용
joined_on_col_to_index = df1_merge.join(df_dept_idx, on='DeptID', how='left', rsuffix='_dept')
# display_dataframe_info(joined_on_col_to_index, "df1.join(df_dept_idx, on='DeptID') 결과")
"""
st.code(code_join, language='python')

if st.checkbox("`DataFrame.join()` 예시 보기", key="join_page"):
    df1_idx_ex = df1_merge.set_index('ID')
    df2_idx_ex = df2_merge.set_index('ID') # df2도 ID를 인덱스로 설정
    
    st.write("`df1_idx_ex` (ID를 인덱스로):")
    st.dataframe(df1_idx_ex)
    st.write("`df2_idx_ex` (ID를 인덱스로):")
    st.dataframe(df2_idx_ex)

    st.write("`df1_idx_ex.join(df2_idx_ex, how='outer', lsuffix='_left', rsuffix='_right')` (Outer Join):")
    # ID가 인덱스이므로, 인덱스 기준으로 조인됨
    # 만약 df1, df2에 동일한 이름의 컬럼이 있다면 lsuffix, rsuffix가 사용됨.
    # 현재는 'DeptID' (df1_idx_ex) 와 'Salary', 'City' (df2_idx_ex)는 겹치지 않음.
    display_dataframe_info(df1_idx_ex.join(df2_idx_ex, how='outer', lsuffix='_left', rsuffix='_right'),
                           "join 결과 (인덱스 기준 Outer)", max_rows_to_display=6, show_describe=False)

    st.markdown("---")
    df_dept_idx_ex = df_dept_merge.set_index('DeptID') # DeptID를 인덱스로
    st.write("`df1_merge.join(df_dept_idx_ex, on='DeptID', how='left', rsuffix='_dept')` (df1의 'DeptID' 컬럼과 df_dept_idx_ex의 인덱스 기준 Left Join):")
    # df1_merge의 'DeptID' 컬럼 값과 df_dept_idx_ex의 인덱스(DeptID) 값을 기준으로 조인
    # df_dept_idx_ex의 컬럼 중 df1_merge와 겹치는 이름이 있다면 rsuffix_dept가 붙음 (여기서는 Name이 겹치지 않음)
    display_dataframe_info(df1_merge.join(df_dept_idx_ex, on='DeptID', how='left', rsuffix='_dept'),
                           "join 결과 (컬럼-인덱스 기준 Left)", max_rows_to_display=4, show_describe=False)


st.markdown("---")
st.markdown("""
데이터 연결 및 병합은 실제 데이터 분석에서 매우 빈번하게 사용되는 작업입니다.
각 함수의 특성과 파라미터를 잘 이해하고 상황에 맞게 사용하면 복잡한 데이터도 효과적으로 통합할 수 있습니다.
""")