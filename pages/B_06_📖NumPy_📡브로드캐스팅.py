# pages/6_📡_브로드캐스팅.py
import streamlit as st
import numpy as np
from utils.utils_numpy import display_array_info

st.header("6. 브로드캐스팅 (Broadcasting)")
st.markdown("""
브로드캐스팅은 NumPy가 산술 연산 중에 서로 다른 모양의 배열을 처리하는 강력한 메커니즘입니다.
특정 제약 조건 하에서 작은 배열이 큰 배열의 모양에 맞게 "확장"되어 요소별 연산이 가능해집니다.
이를 통해 많은 경우 명시적인 반복문 없이 간결하고 효율적인 코드를 작성할 수 있습니다.
""")

st.subheader("6.1 브로드캐스팅 규칙")
st.markdown("""
두 배열 간의 연산에서 브로드캐스팅은 다음 규칙에 따라 차원별로 진행됩니다 (끝 차원부터 시작):

1.  **차원 수가 다를 경우:** 차원 수가 적은 배열의 모양(shape) 앞에 1을 추가하여 두 배열의 차원 수를 동일하게 만듭니다.
    -   예: `(3,4)` 배열과 `(4,)` 배열 -> `(3,4)` 와 `(1,4)` 로 간주.

2.  **차원 크기가 다를 경우:** 특정 차원에서 두 배열의 크기가 다르면, 다음 두 조건 중 하나를 만족해야 합니다:
    a.  한 배열의 해당 차원 크기가 1인 경우.
    b.  두 배열의 해당 차원 크기가 동일한 경우.

3.  **규칙 적용:**
    -   만약 어떤 차원에서 두 배열의 크기가 동일하면, 다음 차원으로 넘어갑니다.
    -   만약 어떤 차원에서 한 배열의 크기가 1이고 다른 배열의 크기가 1보다 크면, 크기가 1인 배열이 다른 배열의 크기에 맞게 "복제" 또는 "확장"된 것처럼 동작합니다. (실제 메모리 복사는 일어나지 않음)
    -   만약 어떤 차원에서 두 배열의 크기가 다르면서 어느 쪽도 1이 아니라면, `ValueError: operands could not be broadcast together` 에러가 발생합니다.

4.  **결과 배열:** 모든 차원에서 호환된다면 연산이 수행되며, 결과 배열의 각 차원 크기는 입력 배열들의 해당 차원 크기 중 더 큰 값으로 결정됩니다.
""")

st.markdown("---")
st.subheader("6.2 브로드캐스팅 예시")

# --- 예시 1: 스칼라와 배열 ---
st.markdown("#### 예시 1: 스칼라와 배열")
st.markdown("스칼라는 모든 배열과 브로드캐스팅될 수 있습니다. 스칼라는 배열의 각 요소에 대해 연산됩니다.")
code_scalar_array = """
import numpy as np

arr = np.array([[1, 2, 3], [4, 5, 6]])
scalar = 10

result = arr + scalar # arr의 각 요소에 10이 더해짐
# display_array_info(arr, "원본 배열 (arr)")
# st.write(f"스칼라 값: {scalar}")
# display_array_info(result, "arr + scalar 결과")
# 규칙:
# arr.shape = (2,3), scalar는 0차원.
# 1. scalar -> (1,1) 또는 해당 차원에 맞게 확장된 것으로 간주
# 2. (2,3)과 (1,1) 비교:
#    - 끝 차원: 3 vs 1 -> 1이 3으로 확장
#    - 다음 차원: 2 vs 1 -> 1이 2로 확장
# 결과 shape: (2,3)
"""
st.code(code_scalar_array, language='python')
if st.checkbox("스칼라와 배열 브로드캐스팅 예시 보기", key="bc_scalar_array_page"):
    arr_ex1 = np.array([[10, 20, 30], [40, 50, 60]])
    scalar_ex1 = 5
    display_array_info(arr_ex1, "원본 배열 `arr_ex1`")
    st.write(f"스칼라 값: `{scalar_ex1}`")
    result_ex1 = arr_ex1 * scalar_ex1
    display_array_info(result_ex1, "`arr_ex1` * `scalar_ex1` 결과")

st.markdown("---")

# --- 예시 2: 1D 배열과 2D 배열 ---
st.markdown("#### 예시 2: 1D 배열과 2D 배열")
st.markdown("1D 배열이 2D 배열의 각 행 또는 열에 대해 연산될 수 있습니다.")
code_1d_2d_array = """
import numpy as np

arr2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]) # shape (3,3)
arr1d_row = np.array([10, 20, 30]) # shape (3,) 또는 (1,3)으로 간주 가능

# arr1d_row가 각 행에 더해짐
result1 = arr2d + arr1d_row
# display_array_info(arr2d, "원본 2D 배열 (arr2d)")
# display_array_info(arr1d_row, "1D 배열 (arr1d_row)")
# display_array_info(result1, "arr2d + arr1d_row 결과")
# 규칙:
# arr2d.shape = (3,3), arr1d_row.shape = (3,)
# 1. arr1d_row -> (1,3)으로 확장 간주
# 2. (3,3)과 (1,3) 비교:
#    - 끝 차원: 3 vs 3 (동일)
#    - 다음 차원: 3 vs 1 -> 1이 3으로 확장 (arr1d_row가 세 번 복제되어 각 행에 더해짐)
# 결과 shape: (3,3)

arr1d_col = np.array([[100], [200], [300]]) # shape (3,1) 또는 arr1d_col = np.array([100,200,300]).reshape(3,1)

# arr1d_col이 각 열에 더해짐
result2 = arr2d + arr1d_col
# display_array_info(arr1d_col, "1D 열 벡터 (arr1d_col)")
# display_array_info(result2, "arr2d + arr1d_col 결과")
# 규칙:
# arr2d.shape = (3,3), arr1d_col.shape = (3,1)
# 2. (3,3)과 (3,1) 비교:
#    - 끝 차원: 3 vs 1 -> 1이 3으로 확장 (arr1d_col의 각 요소가 행 방향으로 복제)
#    - 다음 차원: 3 vs 3 (동일)
# 결과 shape: (3,3)
"""
st.code(code_1d_2d_array, language='python')
if st.checkbox("1D와 2D 배열 브로드캐스팅 예시 보기", key="bc_1d_2d_array_page"):
    arr2d_ex2 = np.arange(1, 10).reshape(3, 3)
    arr1d_row_ex2 = np.array([100, 0, -100]) # (3,)
    display_array_info(arr2d_ex2, "2D 배열 `arr2d_ex2`")
    display_array_info(arr1d_row_ex2, "1D 행방향 배열 `arr1d_row_ex2`")
    result1_ex2 = arr2d_ex2 + arr1d_row_ex2
    display_array_info(result1_ex2, "`arr2d_ex2` + `arr1d_row_ex2` 결과 (각 행에 더해짐)")

    st.markdown("---")
    arr1d_col_ex2 = np.array([[10], [20], [30]]) # (3,1)
    display_array_info(arr1d_col_ex2, "1D 열방향 배열 `arr1d_col_ex2`")
    result2_ex2 = arr2d_ex2 + arr1d_col_ex2
    display_array_info(result2_ex2, "`arr2d_ex2` + `arr1d_col_ex2` 결과 (각 열에 더해짐)")

st.markdown("---")

# --- 예시 3: 서로 다른 2D 배열 ---
st.markdown("#### 예시 3: 서로 다른 2D 배열")
st.markdown("두 2D 배열이 특정 조건을 만족하면 브로드캐스팅될 수 있습니다.")
code_2d_2d_array = """
import numpy as np

a = np.arange(3).reshape(3,1)   # shape (3,1) -> [[0],[1],[2]]
b = np.arange(3)                # shape (3,) -> [0,1,2] (브로드캐스팅 시 (1,3)으로 간주)

result = a + b
# display_array_info(a, "배열 a (3,1)")
# display_array_info(b, "배열 b (3,)")
# display_array_info(result, "a + b 결과")
# 규칙:
# a.shape = (3,1), b.shape = (3,)
# 1. b -> (1,3)으로 확장 간주
# 2. (3,1)과 (1,3) 비교:
#    - 끝 차원 (열): 1 vs 3 -> a의 열이 3으로 확장
#    - 다음 차원 (행): 3 vs 1 -> b의 행이 3으로 확장
# 결과 shape: (3,3)
# result[i,j] = a[i,0] + b[0,j] 와 유사하게 동작
# 예: result[0,0] = a[0,0](0) + b[0](0) = 0
#     result[1,1] = a[1,0](1) + b[1](1) = 2
"""
st.code(code_2d_2d_array, language='python')
if st.checkbox("서로 다른 2D 배열 브로드캐스팅 예시 보기", key="bc_2d_2d_array_page"):
    a_ex3 = np.array([[0], [10], [20], [30]]) # (4,1)
    b_ex3 = np.array([0, 1, 2])               # (3,) -> 브로드캐스팅 시 (1,3)
    display_array_info(a_ex3, "배열 `a_ex3` (4x1)")
    display_array_info(b_ex3, "배열 `b_ex3` (3,)")
    result_ex3 = a_ex3 + b_ex3
    display_array_info(result_ex3, "`a_ex3` + `b_ex3` 결과 (4x3)")
    st.markdown("""
    결과 `result_ex3[i, j] = a_ex3[i, 0] + b_ex3[j]` 형태로 계산됩니다.
    예: `result_ex3[1,1] = a_ex3[1,0] (10) + b_ex3[1] (1) = 11`
    """)

st.markdown("---")

# --- 예시 4: 브로드캐스팅 불가 사례 ---
st.markdown("#### 예시 4: 브로드캐스팅 불가 사례")
st.markdown("규칙을 만족하지 못하면 `ValueError`가 발생합니다.")
code_fail_broadcast = """
import numpy as np

a = np.array([[1,2,3],[4,5,6]]) # shape (2,3)
b = np.array([10,20])           # shape (2,)

try:
    result = a + b
except ValueError as e:
    # print(f"에러 발생: {e}")
    pass # Streamlit에서는 에러를 직접 print하기 보다 st.error 사용
# 규칙:
# a.shape = (2,3), b.shape = (2,)
# 1. b -> (1,2)로 확장 간주
# 2. (2,3)과 (1,2) 비교:
#    - 끝 차원: 3 vs 2 -> 크기가 다르고 어느 쪽도 1이 아님. 에러!
"""
st.code(code_fail_broadcast, language='python')
if st.checkbox("브로드캐스팅 불가 예시 보기", key="bc_fail_page"):
    a_fail = np.ones((3,4)) # (3,4)
    b_fail = np.ones((3,2)) # (3,2)
    display_array_info(a_fail, "배열 `a_fail` (3x4)")
    display_array_info(b_fail, "배열 `b_fail` (3x2)")
    st.write("`a_fail + b_fail` 시도 시:")
    try:
        result_fail = a_fail + b_fail
        st.write(result_fail) # 이 줄은 실행되지 않아야 함
    except ValueError as e:
        st.error(f"ValueError 발생: {e}")
        st.markdown("이유: `a_fail`의 shape은 `(3,4)`, `b_fail`의 shape은 `(3,2)`입니다. 끝 차원(열)의 크기가 4와 2로 다르며, 어느 쪽도 1이 아니므로 브로드캐스팅 규칙에 맞지 않아 에러가 발생합니다.")

st.markdown("---")
st.markdown("브로드캐스팅은 NumPy의 매우 중요한 기능으로, 코드를 간결하고 효율적으로 만들어줍니다. 규칙을 잘 이해하고 활용하면 복잡한 연산도 쉽게 처리할 수 있습니다.")