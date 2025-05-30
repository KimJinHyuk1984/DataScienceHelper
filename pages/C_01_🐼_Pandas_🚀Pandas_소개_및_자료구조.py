# pages/1_🚀_Pandas_소개_및_자료구조.py
import streamlit as st
import pandas as pd
import numpy as np
from utils.utils_pandas import display_dataframe_info

st.header("1. Pandas 소개 및 핵심 자료구조")
st.markdown("""
Pandas는 파이썬에서 데이터를 빠르고 쉽게 다룰 수 있도록 강력한 데이터 구조와 분석 도구를 제공하는 라이브_label입니다.
핵심적인 자료구조는 **Series**와 **DataFrame**입니다.
""")

st.subheader("1.1 Pandas 임포트하기")
st.markdown("일반적으로 `pd`라는 별칭(alias)으로 Pandas를 임포트합니다.")
st.code("""
# Pandas 라이브러리를 pd라는 별칭으로 가져옵니다.
import pandas as pd
# NumPy도 함께 사용하는 경우가 많으므로 같이 임포트하는 것이 일반적입니다.
import numpy as np
""", language='python')

st.markdown("---")

# --- 1.2 Series ---
st.subheader("1.2 Series")
st.markdown("""
`Series`는 1차원 배열과 같은 자료구조로, 각 데이터 값(value)과 그에 해당하는 **인덱스(index)**를 가집니다.
모든 타입의 데이터를 담을 수 있지만, 보통 단일 데이터 타입을 가집니다.
""")
code_series = """
import pandas as pd
import numpy as np

# 파이썬 리스트로 Series 생성
s1 = pd.Series([10, 20, 30, 40, 50])
# st.write("s1:")
# st.write(s1)
# st.write("s1.values:", s1.values) # 값 확인
# st.write("s1.index:", s1.index)   # 인덱스 확인 (기본: RangeIndex)

# 인덱스를 직접 지정하여 Series 생성
s2 = pd.Series([95.5, 88.2, 76.9], index=['Alice', 'Bob', 'Charlie'], name='Scores')
# st.write("\\ns2:")
# st.write(s2)
# st.write("s2.name:", s2.name)       # Series 이름 확인
# st.write("s2['Bob']:", s2['Bob'])   # 인덱스로 값 접근

# 딕셔너리로 Series 생성 (딕셔너리 키가 인덱스가 됨)
data_dict = {'apple': 3, 'banana': 5, 'cherry': 2}
s3 = pd.Series(data_dict)
# st.write("\\ns3:")
# st.write(s3)

# NumPy 배열로 Series 생성
s4 = pd.Series(np.random.randn(4), index=['a', 'b', 'c', 'd'])
# st.write("\\ns4:")
# st.write(s4)
"""
st.code(code_series, language='python')

if st.checkbox("Series 예시 보기", key="series_creation_page"):
    st.markdown("##### 파이썬 리스트로 Series 생성 (기본 인덱스)")
    s1_ex = pd.Series([1.1, 2.2, 3.3, 4.4, 5.5])
    st.write(s1_ex)
    st.write(f"- 값 (Values): `{s1_ex.values}`")
    st.write(f"- 인덱스 (Index): `{s1_ex.index}`")
    st.write(f"- 데이터 타입 (dtype): `{s1_ex.dtype}`")
    st.markdown("---")

    st.markdown("##### 인덱스 지정 및 이름 부여")
    s2_ex = pd.Series(['Python', 'Java', 'C++'], index=['Lang1', 'Lang2', 'Lang3'], name='Programming Languages')
    st.write(s2_ex)
    st.write(f"- 이름 (Name): `{s2_ex.name}`")
    st.write(f"- `s2_ex['Lang2']`: {s2_ex['Lang2']}")
    st.markdown("---")

    st.markdown("##### 딕셔너리로 Series 생성")
    population_dict = {'Seoul': 970, 'Busan': 340, 'Incheon': 300} # 단위: 만 명
    s3_ex = pd.Series(population_dict, name="City Population (만 명)")
    st.write(s3_ex)
    st.write(f"- `s3_ex > 500`: \n{s3_ex[s3_ex > 500]}") # 불리언 인덱싱
    st.markdown("---")

st.markdown("---")

# --- 1.3 DataFrame ---
st.subheader("1.3 DataFrame")
st.markdown("""
`DataFrame`은 2차원 테이블 형태의 자료구조로, 여러 개의 `Series`가 모여서 구성된 것으로 생각할 수 있습니다.
각 열(column)은 서로 다른 데이터 타입을 가질 수 있으며, 행 인덱스(row index)와 열 이름(column name)을 가집니다.
""")

st.markdown("#### DataFrame 생성 방법")
code_dataframe = """
import pandas as pd
import numpy as np

# 1. 딕셔너리로부터 DataFrame 생성 (키가 열 이름, 값이 열 데이터)
data_dict_df = {
    'Name': ['Alice', 'Bob', 'Charlie', 'David'],
    'Age': [25, 30, 35, 28],
    'City': ['New York', 'Paris', 'London', 'Tokyo']
}
df1 = pd.DataFrame(data_dict_df)
# display_dataframe_info(df1, "딕셔너리로부터 생성된 DataFrame (df1)")

# 2. 리스트의 리스트 (또는 NumPy 배열)로부터 DataFrame 생성 (열 이름 지정 필요)
data_list_df = [
    ['Alice', 25, 'New York'],
    ['Bob', 30, 'Paris'],
    ['Charlie', 35, 'London']
]
df2 = pd.DataFrame(data_list_df, columns=['Name', 'Age', 'City'], index=['ID1', 'ID2', 'ID3'])
# display_dataframe_info(df2, "리스트의 리스트로부터 생성된 DataFrame (df2)")

# 3. NumPy 배열로부터 DataFrame 생성
np_array = np.random.randint(70, 101, size=(4, 3)) # 70~100 사이 정수, 4x3 배열
df3 = pd.DataFrame(np_array, columns=['Math', 'English', 'Science'], index=['Std1', 'Std2', 'Std3', 'Std4'])
# display_dataframe_info(df3, "NumPy 배열로부터 생성된 DataFrame (df3)")

# 4. Series들의 딕셔너리로부터 DataFrame 생성
series_dict_df = {
    'ColA': pd.Series(np.random.rand(3), index=['idx1', 'idx2', 'idx3']),
    'ColB': pd.Series(np.random.rand(3)+1, index=['idx1', 'idx2', 'idx3'])
}
df4 = pd.DataFrame(series_dict_df)
# display_dataframe_info(df4, "Series 딕셔너리로부터 생성된 DataFrame (df4)")
"""
st.code(code_dataframe, language='python')

if st.checkbox("DataFrame 생성 예시 보기", key="dataframe_creation_page"):
    st.markdown("##### 1. 딕셔너리로부터 DataFrame 생성")
    data_dict_df_ex = {
        'ProductID': [101, 102, 103, 104],
        'ProductName': ['Laptop', 'Mouse', 'Keyboard', 'Monitor'],
        'Price': [1200, 25, 75, 300],
        'Stock': [50, 150, 100, 30]
    }
    df1_ex = pd.DataFrame(data_dict_df_ex)
    display_dataframe_info(df1_ex, "제품 정보 DataFrame (df1_ex)", max_rows_to_display=4)

    st.markdown("##### 2. NumPy 배열로부터 DataFrame 생성 (인덱스 및 컬럼 지정)")
    np_array_ex = np.array([
        [202301, 100, 5.0],
        [202301, 101, 4.5],
        [202302, 100, 5.5],
        [202302, 102, 3.0]
    ])
    df2_ex = pd.DataFrame(np_array_ex,
                          columns=['DateCode', 'ItemID', 'Rating'],
                          index=pd.to_datetime(['2023-01-15', '2023-01-20', '2023-02-10', '2023-02-25']))
    df2_ex['DateCode'] = df2_ex['DateCode'].astype(int) # 타입 변환 예시
    df2_ex['ItemID'] = df2_ex['ItemID'].astype(int)
    display_dataframe_info(df2_ex, "평점 데이터 DataFrame (df2_ex)", max_rows_to_display=4)

st.markdown("#### DataFrame의 주요 속성")
code_df_attributes = """
import pandas as pd
# df = pd.DataFrame(...) # df가 이미 생성되었다고 가정

# df.index      # 행 인덱스
# df.columns    # 열 이름 (컬럼)
# df.values     # 값 (NumPy 배열 형태로 반환)
# df.dtypes     # 각 열의 데이터 타입
# df.shape      # (행의 수, 열의 수) 튜플
# df.ndim       # 차원 수 (DataFrame은 항상 2)
# df.size       # 전체 요소 수 (행 수 * 열 수)
# df.T          # 전치된 DataFrame (행과 열 바뀜)
"""
st.code(code_df_attributes, language="python")
if st.checkbox("DataFrame 속성 확인 예시 보기", key="df_attributes_page"):
    data_for_attrs = {'A': [1,2,3], 'B': [4.0, 5.5, 6.1], 'C': ['x', 'y', 'z']}
    df_attrs_ex = pd.DataFrame(data_for_attrs, index=['row1', 'row2', 'row3'])
    st.write("예제 DataFrame (`df_attrs_ex`):")
    st.dataframe(df_attrs_ex)
    st.write(f"- `df_attrs_ex.index`: `{df_attrs_ex.index}`")
    st.write(f"- `df_attrs_ex.columns`: `{list(df_attrs_ex.columns)}`")
    st.write(f"- `df_attrs_ex.values` (NumPy 배열):")
    st.text(df_attrs_ex.values)
    st.write(f"- `df_attrs_ex.dtypes`:")
    st.dataframe(df_attrs_ex.dtypes.rename("dtype"))
    st.write(f"- `df_attrs_ex.shape`: `{df_attrs_ex.shape}`")
    st.write(f"- `df_attrs_ex.ndim`: `{df_attrs_ex.ndim}`")
    st.write(f"- `df_attrs_ex.size`: `{df_attrs_ex.size}`")
    st.write(f"- `df_attrs_ex.T` (전치된 DataFrame):")
    st.dataframe(df_attrs_ex.T)

st.markdown("---")
st.markdown("DataFrame은 Pandas에서 가장 중심이 되는 데이터 구조로, 이어지는 페이지들에서 DataFrame을 다루는 다양한 방법을 살펴보겠습니다.")