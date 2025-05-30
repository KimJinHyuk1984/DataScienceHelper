# utils_numpy.py
import streamlit as st
import numpy as np

def display_array_info(arr, title="NumPy 배열 정보", display_content=True):
    """
    Streamlit에 NumPy 배열과 관련 정보를 표시합니다.

    Args:
        arr (np.ndarray): 표시할 NumPy 배열.
        title (str): 정보 섹션의 부제목.
        display_content (bool): 배열 내용 전체를 표시할지 여부.
    """
    st.subheader(title)
    if display_content:
        st.write("배열 내용:")
        if arr.ndim > 2:
            # 3차원 이상 배열은 st.text()를 사용하여 일반 텍스트로 표시
            st.text(f"({arr.ndim}차원 배열, 아래는 문자열 표현입니다)\n{arr}")
        elif arr.ndim == 2 and arr.shape[0] > 10 and arr.shape[1] > 1: # 너무 큰 2D 배열도 st.text가 나을 수 있음
             st.text(f"({arr.ndim}차원 배열, 일부만 표시될 수 있음)\n{arr}")
        else:
            # 1차원 또는 작은 2차원 배열은 st.write()가 적절히 처리
            st.write(arr)
    else:
        st.write(f"배열 내용 (첫 5개 요소): `{arr.ravel()[:5]}...` (전체 표시는 생략)")

    st.write(f"- 형태 (Shape): `{arr.shape}`")
    st.write(f"- 차원 (Dimensions, ndim): `{arr.ndim}`")
    st.write(f"- 데이터 타입 (Data Type, dtype): `{arr.dtype}`")
    st.write(f"- 총 요소 수 (Size): `{arr.size}`")
    st.write(f"- 각 요소의 바이트 크기 (Item Size): `{arr.itemsize}` bytes")
    st.markdown("---")