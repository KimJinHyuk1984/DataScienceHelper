# pages/1_🚀_NumPy_소개_및_ndarray.py
import streamlit as st
import numpy as np
from utils.utils_numpy import display_array_info # 유틸리티 함수 사용

st.header("1. NumPy 소개 및 ndarray 객체")

st.markdown("""
NumPy는 "Numerical Python"의 약자로, 파이썬에서 과학 계산을 위한 핵심 라이브러리입니다.
NumPy의 가장 중요한 기능은 다차원 배열 객체인 `ndarray` (N-dimensional array)입니다.
이 배열은 일반 파이썬 리스트보다 훨씬 빠르고 메모리 효율적으로 대량의 숫자 데이터를 처리할 수 있게 해줍니다.
""")

st.subheader("1.1 NumPy 임포트하기")
st.markdown("일반적으로 `np`라는 별칭(alias)으로 NumPy를 임포트합니다.")
st.code("""
# NumPy 라이브러리를 np라는 별칭으로 가져옵니다.
import numpy as np
""", language='python')

st.subheader("1.2 `ndarray` 객체")
st.markdown("""
`ndarray`는 동일한 자료형(dtype)을 가지는 값들의 그리드(grid)입니다. 배열의 차원 수(rank)와 각 차원의 크기(shape)로 정의됩니다.
""")

code_ndarray_intro = """
# numpy를 np로 임포트합니다.
import numpy as np

# 파이썬 리스트로부터 1차원 NumPy 배열 생성
arr1d = np.array([1, 2, 3, 4, 5])

# 중첩된 파이썬 리스트로부터 2차원 NumPy 배열 생성
arr2d = np.array([[1, 2, 3], [4, 5, 6]])

# print("1차원 배열:")
# print(arr1d)
# print("형태:", arr1d.shape) # 배열의 각 차원의 크기
# print("차원 수:", arr1d.ndim) # 배열의 차원 수
# print("데이터 타입:", arr1d.dtype) # 배열 요소의 데이터 타입
# print("총 요소 수:", arr1d.size) # 배열의 전체 요소 수
# print("각 요소의 바이트 크기:", arr1d.itemsize) # 배열 요소 하나의 바이트 크기

# print("\\n2차원 배열:")
# print(arr2d)
# print("형태:", arr2d.shape)
# print("차원 수:", arr2d.ndim)
# print("데이터 타입:", arr2d.dtype)
"""
st.code(code_ndarray_intro, language='python')

if st.checkbox("`ndarray` 객체 예시 실행 및 정보 보기", key="ndarray_intro_page"):
    # 파이썬 리스트로부터 1차원 NumPy 배열 생성
    arr1d_ex = np.array([10, 20, 30, 40, 50])
    # 유틸리티 함수를 사용하여 배열 정보 표시
    display_array_info(arr1d_ex, title="1차원 배열 (arr1d_ex)")

    # 중첩된 파이썬 리스트로부터 2차원 NumPy 배열 생성
    arr2d_ex = np.array([[1.5, 2.5, 3.5], [4.5, 5.5, 6.5]])
    # 유틸리티 함수를 사용하여 배열 정보 표시
    display_array_info(arr2d_ex, title="2차원 배열 (arr2d_ex)")

    # 데이터 타입 지정하여 배열 생성
    arr_float32 = np.array([1, 2, 3], dtype=np.float32)
    display_array_info(arr_float32, title="float32 타입 배열 (arr_float32)")

    arr_complex = np.array([1+2j, 3+4j, 5+6j]) # 복소수 배열
    display_array_info(arr_complex, title="복소수 배열 (arr_complex)")


st.subheader("1.3 NumPy를 사용하는 이유")
st.markdown("""
-   **성능:** NumPy 배열은 C로 구현되어 있어, 파이썬 리스트보다 연산 속도가 훨씬 빠릅니다. 특히 대량의 데이터에 대한 반복문 없는 벡터화 연산에서 뛰어난 성능을 보입니다.
-   **메모리 효율:** 동일한 타입의 데이터를 연속된 메모리 블록에 저장하므로 파이썬 리스트보다 메모리를 적게 사용합니다.
-   **편리한 기능:** 선형대수, 푸리에 변환, 난수 생성 등 다양한 고급 수학 및 과학 함수를 제공합니다.
-   **생태계:** Pandas, SciPy, Matplotlib, Scikit-learn 등 수많은 과학 컴퓨팅 라이브러리들이 NumPy 배열을 기본 데이터 구조로 사용합니다.
""")