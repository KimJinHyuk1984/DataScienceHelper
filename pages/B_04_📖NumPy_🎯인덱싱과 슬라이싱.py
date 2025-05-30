# pages/4_🎯_인덱싱과_슬라이싱.py
import streamlit as st
import numpy as np
from utils.utils_numpy import display_array_info

st.header("4. 인덱싱과 슬라이싱")
st.markdown("""
NumPy 배열의 특정 요소나 부분 배열에 접근하는 방법입니다. 파이썬 리스트와 유사하지만, 다차원 배열과 고급 인덱싱 기능을 지원하여 훨씬 강력합니다.
""")

# 예제 배열 생성 (모든 예시에서 공통으로 사용)
base_arr_1d = np.arange(10, 20) # 10, 11, ..., 19
base_arr_2d = np.arange(1, 13).reshape(3, 4) # 1~12, 3x4 행렬
base_arr_3d = np.arange(24).reshape(2, 3, 4) # 0~23, 2x3x4 텐서

st.subheader("예제 배열 확인")
if st.checkbox("예제 배열 내용 보기", key="show_base_arrays_indexing_page"):
    display_array_info(base_arr_1d, "1차원 예제 배열 (base_arr_1d)")
    display_array_info(base_arr_2d, "2차원 예제 배열 (base_arr_2d)")
    display_array_info(base_arr_3d, "3차원 예제 배열 (base_arr_3d)")

st.markdown("---")

# --- 4.1 기본 인덱싱 및 슬라이싱 ---
st.subheader("4.1 기본 인덱싱 및 슬라이싱")
st.markdown("""
-   **기본 인덱싱:** `arr[i]`, `arr[i, j]`, `arr[i, j, k]` 와 같이 각 차원의 인덱스를 지정하여 단일 요소를 선택합니다.
-   **기본 슬라이싱:** `start:stop:step` 구문을 사용하여 각 차원의 부분 배열을 선택합니다. 슬라이싱된 배열은 원본 배열의 뷰(view)이므로, 슬라이스 수정 시 원본도 변경됩니다.
""")

code_basic_indexing = """
import numpy as np

# 1차원 배열
arr1d = np.arange(10, 20) # [10 11 12 13 14 15 16 17 18 19]
# 첫 번째 요소: arr1d[0] -> 10
# 마지막 요소: arr1d[-1] -> 19
# 슬라이싱: arr1d[2:5] -> [12 13 14] (인덱스 2,3,4)
# 슬라이싱 (step 사용): arr1d[::2] -> [10 12 14 16 18] (짝수 인덱스 요소)

# 2차원 배열
arr2d = np.arange(1, 13).reshape(3, 4)
# [[ 1  2  3  4]
#  [ 5  6  7  8]
#  [ 9 10 11 12]]
# 단일 요소: arr2d[0, 1] -> 2 (0행 1열)
# 특정 행 전체: arr2d[1] 또는 arr2d[1, :] -> [5 6 7 8] (1행 전체)
# 특정 열 전체: arr2d[:, 2] -> [3 7 11] (2열 전체)
# 부분 배열 슬라이싱: arr2d[0:2, 1:3]
# [[2 3]
#  [6 7]] (0~1행, 1~2열)

# 3차원 배열
arr3d = np.arange(24).reshape(2, 3, 4) # 2개의 (3x4) 행렬
# arr3d[0] -> 첫 번째 (3x4) 행렬
# arr3d[0, 1] -> 첫 번째 행렬의 1행 ([4 5 6 7])
# arr3d[0, 1, 2] -> 첫 번째 행렬의 1행 2열 요소 (6)
# arr3d[:, :, ::2] -> 모든 (2x3) 행렬에서 각 행의 짝수 열만 선택
"""
st.code(code_basic_indexing, language='python')

if st.checkbox("기본 인덱싱/슬라이싱 예시 보기", key="basic_indexing_page"):
    st.write("#### 1차원 배열 (`base_arr_1d`)")
    display_array_info(base_arr_1d, "`base_arr_1d` (10~19)", False)
    st.write(f"`base_arr_1d[0]`: {base_arr_1d[0]}")
    st.write(f"`base_arr_1d[-1]`: {base_arr_1d[-1]}")
    st.write(f"`base_arr_1d[2:5]`: {base_arr_1d[2:5]}")
    st.write(f"`base_arr_1d[::2]`: {base_arr_1d[::2]}")
    st.write(f"`base_arr_1d[5:]`: {base_arr_1d[5:]}")
    st.markdown("---")

    st.write("#### 2차원 배열 (`base_arr_2d`)")
    display_array_info(base_arr_2d, "`base_arr_2d` (1~12, 3x4)", True)
    st.write(f"`base_arr_2d[0, 1]`: {base_arr_2d[0, 1]}")
    st.write(f"`base_arr_2d[1]`: {base_arr_2d[1]}") # 또는 base_arr_2d[1,:]
    st.write(f"`base_arr_2d[:, 2]`: {base_arr_2d[:, 2]}")
    st.write(f"`base_arr_2d[0:2, 1:3]` (0~1행, 1~2열):")
    st.write(base_arr_2d[0:2, 1:3])
    st.markdown("---")

    st.write("#### 3차원 배열 (`base_arr_3d`)")
    display_array_info(base_arr_3d, "`base_arr_3d` (0~23, 2x3x4)", True)
    st.write(f"`base_arr_3d[0, 1, 2]`: {base_arr_3d[0, 1, 2]}")
    st.write(f"`base_arr_3d[1, :, ::2]` (두 번째 행렬, 모든 행, 짝수 열):")
    st.write(base_arr_3d[1, :, ::2])

    st.markdown("#### 슬라이스의 뷰(View) 특성")
    arr_slice_view = np.arange(5)
    st.write(f"원본 배열: `{arr_slice_view}`")
    my_slice = arr_slice_view[1:4] # [1,2,3]
    st.write(f"슬라이스: `{my_slice}`")
    my_slice[0] = 100 # 슬라이스 변경
    st.write(f"슬라이스 변경 후: `{my_slice}`")
    st.write(f"원본 배열 변경 확인: `{arr_slice_view}` (원본도 변경됨!)")
    st.write("복사본을 원하면 `.copy()`를 사용: `my_slice_copy = arr_slice_view[1:4].copy()`")


st.markdown("---")

# --- 4.2 불리언 배열 인덱싱 (Boolean Array Indexing) ---
st.subheader("4.2 불리언 배열 인덱싱 (Boolean Array Indexing)")
st.markdown("""
조건을 만족하는 요소들만 선택할 때 사용합니다. 조건 연산(예: `arr > 5`)의 결과로 얻어지는 불리언 배열을 인덱스로 사용합니다.
결과는 항상 원본 배열의 복사본입니다.
""")
code_boolean_indexing = """
import numpy as np

arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
# display_array_info(arr, "원본 배열")

# 5보다 큰 요소만 선택
bool_index = arr > 5
# display_array_info(bool_index, "불리언 인덱스 (arr > 5)")

selected_elements = arr[bool_index] # 또는 arr[arr > 5]
# display_array_info(selected_elements, "5보다 큰 요소들 (1차원 배열로 반환)")

# 특정 조건을 만족하는 요소에 값 할당
arr[arr % 2 == 0] = -1 # 짝수인 요소를 -1로 변경
# display_array_info(arr, "짝수 요소를 -1로 변경한 배열")
"""
st.code(code_boolean_indexing, language='python')
if st.checkbox("불리언 인덱싱 예시 보기", key="boolean_indexing_page"):
    arr_bool_ex = np.array([-2, -1, 0, 1, 2, 3, 4])
    display_array_info(arr_bool_ex, "원본 배열 `arr_bool_ex`")

    positive_values = arr_bool_ex[arr_bool_ex > 0]
    display_array_info(positive_values, "`arr_bool_ex > 0` 인 요소 (복사본)")

    # 여러 조건 조합 (&: AND, |: OR, ~: NOT)
    # 주의: ( ) 괄호 필수!
    multi_condition_values = arr_bool_ex[(arr_bool_ex >= 0) & (arr_bool_ex < 3)] # 0, 1, 2
    display_array_info(multi_condition_values, "`0 <= arr_bool_ex < 3` 인 요소")

    arr_assign_ex = base_arr_2d.copy() # 원본 유지를 위해 복사
    st.write("`base_arr_2d` 복사본 `arr_assign_ex`:")
    st.write(arr_assign_ex)
    arr_assign_ex[arr_assign_ex % 3 == 0] = 99 # 3의 배수인 요소를 99로 변경
    display_array_info(arr_assign_ex, "3의 배수를 99로 변경한 `arr_assign_ex`")


st.markdown("---")

# --- 4.3 정수 배열 인덱싱 (Fancy Indexing) ---
st.subheader("4.3 정수 배열 인덱싱 (Fancy Indexing)")
st.markdown("""
인덱스 배열(정수 리스트 또는 NumPy 배열)을 사용하여 배열의 특정 요소들을 선택하거나 순서를 바꿀 수 있습니다.
결과는 항상 원본 배열의 복사본입니다.
""")
code_fancy_indexing = """
import numpy as np

arr = np.array([10, 20, 30, 40, 50, 60])
# display_array_info(arr, "원본 1차원 배열")

# 인덱스 [1, 3, 4]에 해당하는 요소 선택
selected_by_indices = arr[[1, 3, 4]] # 결과: [20 40 50]
# display_array_info(selected_by_indices, "arr[[1, 3, 4]]")

# 인덱스 배열을 사용하여 순서 변경 및 반복 선택
reordered_arr = arr[[0, 2, 1, 0, 3, 3]] # 결과: [10 30 20 10 40 40]
# display_array_info(reordered_arr, "arr[[0, 2, 1, 0, 3, 3]]")

# 2차원 배열에서의 팬시 인덱싱
arr2d = np.array([[1,2,3],[4,5,6],[7,8,9]])
# display_array_info(arr2d, "원본 2차원 배열")

# (0,0), (1,1), (2,2) 요소 선택 (대각선 요소)
diag_elements = arr2d[[0, 1, 2], [0, 1, 2]] # 결과: [1 5 9]
# display_array_info(diag_elements, "arr2d[[0,1,2], [0,1,2]] - 대각선 요소")

# 특정 행들 선택 (예: 0행과 2행)
selected_rows = arr2d[[0, 2]]
# display_array_info(selected_rows, "arr2d[[0, 2]] - 0행과 2행 선택")

# 특정 행의 특정 열들 선택 (예: 0행의 0,2열 / 2행의 1,2열)
# 각 요소의 인덱스는 (row_indices[i], col_indices[i])
selected_fancy = arr2d[[0, 2], [0, 2]] # (0,0)과 (2,2) 요소. 즉, arr2d[0,0]과 arr2d[2,2]
# display_array_info(selected_fancy, "arr2d[[0,2], [0,2]]")

# 모든 행에 대해 특정 열들을 팬시 인덱싱 (조금 더 복잡)
# 예: 0열과 2열 선택
selected_cols_fancy = arr2d[:, [0, 2]]
# display_array_info(selected_cols_fancy, "arr2d[:, [0, 2]] - 0열과 2열 선택")
"""
st.code(code_fancy_indexing, language='python')
if st.checkbox("정수 배열 인덱싱 (팬시 인덱싱) 예시 보기", key="fancy_indexing_page"):
    arr_fancy_1d_ex = np.array(['A', 'B', 'C', 'D', 'E'])
    display_array_info(arr_fancy_1d_ex, "원본 1D 배열 (`arr_fancy_1d_ex`)")
    indices = np.array([0, 0, 3, 1, 3])
    selected_1d_ex = arr_fancy_1d_ex[indices]
    display_array_info(selected_1d_ex, f"`arr_fancy_1d_ex`[{indices}]")

    arr_fancy_2d_ex = base_arr_2d.copy() # (3x4) 배열
    display_array_info(arr_fancy_2d_ex, "원본 2D 배열 (`arr_fancy_2d_ex`)")
    # (0,1), (2,3), (1,0) 위치의 요소들 선택
    row_indices = np.array([0, 2, 1])
    col_indices = np.array([1, 3, 0])
    selected_2d_ex = arr_fancy_2d_ex[row_indices, col_indices]
    display_array_info(selected_2d_ex, f"`arr_fancy_2d_ex`[{row_indices}, {col_indices}]")

    st.write("특정 행들만 선택 (예: 0행과 2행):")
    selected_rows_ex = arr_fancy_2d_ex[[0, 2]] # 또는 arr_fancy_2d_ex[np.array([0,2]), :]
    display_array_info(selected_rows_ex, "`arr_fancy_2d_ex`[[0, 2]]")

    st.write("특정 열들만 선택 (예: 1열과 3열):")
    selected_cols_ex = arr_fancy_2d_ex[:, [1, 3]]
    display_array_info(selected_cols_ex, "`arr_fancy_2d_ex`[:, [1, 3]]")

st.markdown("---")
st.subheader("4.4 `np.newaxis` 와 `...` (Ellipsis)")
st.markdown("""
- `np.newaxis`: 기존 배열에 새로운 축을 추가하여 차원을 늘립니다. 슬라이싱과 함께 사용되어 특정 축의 크기가 1인 형태로 만듭니다.
- `...` (Ellipsis): 다차원 배열에서 여러 개의 `:` (콜론)을 대체하여 사용할 수 있는 간결한 표기법입니다. 필요한 만큼의 차원을 모두 선택합니다.
""")
code_newaxis_ellipsis = """
import numpy as np

arr = np.arange(1, 5) # [1 2 3 4], shape (4,)
# display_array_info(arr, "원본 1차원 배열")

# 열 벡터로 변환 (4x1)
col_vec1 = arr[:, np.newaxis]
# display_array_info(col_vec1, "arr[:, np.newaxis] - 열 벡터")

# 행 벡터로 변환 (1x4) - 이미 1차원이지만, 명시적으로 2D 행 벡터로
row_vec1 = arr[np.newaxis, :]
# display_array_info(row_vec1, "arr[np.newaxis, :] - 행 벡터")

# Ellipsis 예제
arr3d = np.arange(24).reshape(2, 3, 4)
# display_array_info(arr3d, "원본 3차원 배열 (2x3x4)")

# 첫 번째 축의 모든 요소, 두 번째 축의 0번 인덱스, 세 번째 축의 모든 요소
# arr3d[:, 0, :] 와 동일
ellipsis_ex1 = arr3d[..., 0, :] # 또는 arr3d[:,0,:]
# display_array_info(ellipsis_ex1, "arr3d[..., 0, :]")

# 첫 번째 축의 0번 인덱스, 나머지 모든 축의 모든 요소
# arr3d[0, :, :] 와 동일
ellipsis_ex2 = arr3d[0, ...] # 또는 arr3d[0]
# display_array_info(ellipsis_ex2, "arr3d[0, ...]")

# 마지막 축의 1번 인덱스, 앞의 모든 축의 모든 요소
# arr3d[:, :, 1] 와 동일
ellipsis_ex3 = arr3d[..., 1]
# display_array_info(ellipsis_ex3, "arr3d[..., 1]")
"""
st.code(code_newaxis_ellipsis, language='python')
if st.checkbox("`np.newaxis` 및 `Ellipsis (...)` 예시 보기", key="newaxis_ellipsis_page"):
    arr_na_ex = np.array([10, 20, 30])
    display_array_info(arr_na_ex, "원본 1D 배열 `arr_na_ex`")
    st.write("`np.newaxis`를 사용하여 차원 추가:")
    display_array_info(arr_na_ex[:, np.newaxis], "`arr_na_ex[:, np.newaxis]` (열 벡터)")
    display_array_info(arr_na_ex[np.newaxis, :, np.newaxis], "`arr_na_ex[np.newaxis, :, np.newaxis]` (1x3x1 형태)")

    arr_el_ex = np.arange(60).reshape(3,4,5) # 3x4x5 배열
    display_array_info(arr_el_ex, "원본 3D 배열 `arr_el_ex` (3x4x5)")
    st.write("`Ellipsis (...)` 사용 예:")
    st.write("`arr_el_ex[0, ..., 2]` (0번 '판', 모든 '행', 2번 '열'):")
    st.write(arr_el_ex[0, ..., 2]) # arr_el_ex[0, :, 2]와 동일
    st.write("`arr_el_ex[..., 0]` (모든 '판', 모든 '행', 0번 '열'):")
    st.write(arr_el_ex[..., 0]) # arr_el_ex[:, :, 0]와 동일