# utils_pandas.py
import streamlit as st
import pandas as pd
import io # st.dataframe(df.dtypes) 대신 df.dtypes를 보기 좋게 출력하기 위함

def display_dataframe_info(df, title="Pandas DataFrame 정보", display_content=True, max_rows_to_display=5, show_dtypes=True, show_describe=True):
    """
    Streamlit에 Pandas DataFrame과 관련 정보를 표시합니다.

    Args:
        df (pd.DataFrame): 표시할 Pandas DataFrame.
        title (str): 정보 섹션의 부제목.
        display_content (bool): DataFrame 내용 일부를 표시할지 여부.
        max_rows_to_display (int): display_content가 True일 때 보여줄 최대 행 수.
        show_dtypes (bool): df.dtypes 정보를 표시할지 여부.
        show_describe (bool): df.describe() 정보를 표시할지 여부.
    """
    st.subheader(title)
    if display_content:
        st.write(f"DataFrame 내용 (상위 {max_rows_to_display} 행):")
        st.dataframe(df.head(max_rows_to_display))
        if len(df) > max_rows_to_display:
            st.caption(f"... (총 {len(df)} 행 중 {max_rows_to_display} 행 표시)")
    else:
        st.write(f"DataFrame 내용 (표시 생략, 총 {len(df)} 행)")

    st.write(f"- 형태 (Shape): `{df.shape}`")
    st.write(f"- 인덱스 (Index): `{df.index}`")
    st.write(f"- 컬럼 (Columns): `{list(df.columns)}`")

    if show_dtypes:
        st.write("- 데이터 타입 (Data Types):")
        # df.dtypes를 보기 좋게 출력하기 위한 방법
        buffer = io.StringIO()
        df.info(buf=buffer, verbose=False) # dtypes만 간략히
        st.text(buffer.getvalue().split("dtypes: ")[1].split("\nmemory")[0]) # dtypes 부분만 추출

    if show_describe:
        st.write("- 기본 통계 (`df.describe(include='all')`):")
        # describe() 결과가 길 수 있으므로 expander 사용
        with st.expander("`df.describe(include='all')` 결과 보기"):
            st.dataframe(df.describe(include='all'))
    st.markdown("---")