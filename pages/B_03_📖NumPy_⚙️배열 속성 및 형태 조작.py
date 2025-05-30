# pages/3_⚙️_배열_속성_및_형태_조작.py
import streamlit as st
import numpy as np
from utils.utils_numpy import display_array_info

st.header("3. 배열 속성 및 형태 조작")
st.markdown("NumPy 배열(`ndarray`)은 다양한 속성을 가지며, 그 형태(shape)를 유연하게 변경할 수 있습니다.")

# --- 3.1 배열의 주요 속성 ---
st.subheader("3.1 배열의 주요 속성")
st.markdown("""
- `ndarray.ndim`: 배열의 차원 수 (정수).
- `ndarray.shape`: 각 차원의 크기를 나타내는 튜플.
- `ndarray.size`: 배열의 전체 요소 수 (정수). `shape` 튜플의 모든 요소를 곱한 값과 같습니다.
- `ndarray.dtype`: 배열 요소의 데이터 타입 객체.
- `ndarray.itemsize`: 배열의 각 요소가 차지하는 메모리 크기 (바이트 단위, 정수).
- `ndarray.data`: 배열의 실제 요소들이 저장된 메모리 버퍼 객체. (일반적으로 직접 접근할 일은 적음)
""")
code_attributes = """
import numpy as np

# 예제 배열 생성
arr = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int16)

# print(f"배열 내용:\\n{arr}")
# print(f"arr.ndim: {arr.ndim}")     # 차원 수
# print(f"arr.shape: {arr.shape}")   # 형태 (행, 열)
# print(f"arr.size: {arr.size}")     # 전체 요소 수
# print(f"arr.dtype: {arr.dtype}")   # 데이터 타입
# print(f"arr.itemsize: {arr.itemsize}") # 각 요소의 바이트 크기 (int16은 2바이트)
"""
st.code(code_attributes, language='python')
if st.checkbox("배열 속성 확인 예시 보기", key="attributes_page"):
    arr_ex = np.arange(1, 13, dtype=np.float64).reshape(3, 4) # 1~12 범위, float64 타입, 3x4 형태
    display_array_info(arr_ex, title="예제 배열 `arr_ex`의 속성", display_content=True)

st.markdown("---")

# --- 3.2 데이터 타입 변경 (`astype()`) ---
st.subheader("3.2 데이터 타입 변경 (`astype()`)")
st.markdown("""
`astype()` 메소드를 사용하면 배열의 데이터 타입을 변경할 수 있습니다. 이 작업은 항상 새로운 배열을 생성합니다(원본 배열은 변경되지 않음).
""")
code_astype = """
import numpy as np

arr_int = np.array([1, 2, 3, 4, 5])
# display_array_info(arr_int, "원본 정수 배열 (arr_int)")

# 정수형 배열을 부동소수점형으로 변경
arr_float = arr_int.astype(np.float64)
# display_array_info(arr_float, "float64로 변경된 배열 (arr_float)")

arr_str = np.array(['1.1', '2.2', '3.3'])
# display_array_info(arr_str, "원본 문자열 배열 (arr_str)")

# 문자열 배열을 부동소수점형으로 변경
# 주의: 변환 불가능한 문자열이 있으면 에러 발생
arr_from_str = arr_str.astype(float)
# display_array_info(arr_from_str, "float으로 변경된 배열 (arr_from_str)")

# 부동소수점형 배열을 정수형으로 변경 (소수점 이하 버림)
arr_back_to_int = arr_float.astype(int) # 또는 np.int32, np.int64 등
# display_array_info(arr_back_to_int, "int로 변경된 배열 (arr_back_to_int)")
"""
st.code(code_astype, language='python')
if st.checkbox("`astype()` 예시 보기", key="astype_page"):
    arr_orig_ex = np.array([0, 1, 2, 3], dtype=np.uint8) # 부호 없는 8비트 정수
    display_array_info(arr_orig_ex, "원본 배열 (uint8)")

    arr_to_float_ex = arr_orig_ex.astype(np.float32)
    display_array_info(arr_to_float_ex, "float32로 변경")

    arr_bool_ex = np.array([0, 1, -2, 0.5, 0.0])
    arr_to_bool_ex = arr_bool_ex.astype(bool) # 0 또는 0.0은 False, 그 외는 True
    display_array_info(arr_to_bool_ex, "bool로 변경 (0은 False, 나머지는 True)")

st.markdown("---")

# --- 3.3 배열 형태 변경 (`reshape()`, `ravel()`, `flatten()`) ---
st.subheader("3.3 배열 형태 변경 (`reshape()`, `ravel()`, `flatten()`)")
st.markdown("""
- `ndarray.reshape(new_shape)` 또는 `np.reshape(array, new_shape)`: 배열의 전체 요소 수는 유지하면서 형태를 변경합니다. 변경된 배열은 원본 배열과 데이터를 공유하는 뷰(view)일 수도 있고, 아닐 수도 있습니다 (메모리 레이아웃에 따라 다름). `-1`을 `new_shape`의 한 차원 값으로 사용하면 해당 차원의 크기가 자동으로 계산됩니다.
- `ndarray.ravel(order='C')`: 다차원 배열을 1차원 배열로 펼칩니다. 가능한 경우 원본 배열의 뷰를 반환합니다. `order='F'`는 Fortran 스타일(열 우선)로 펼칩니다.
- `ndarray.flatten(order='C')`: 다차원 배열을 1차원 배열로 펼칩니다. 항상 원본 배열의 복사본을 반환합니다.
""")
code_reshape = """
import numpy as np

arr = np.arange(12) # 0부터 11까지의 1차원 배열
# display_array_info(arr, "원본 1차원 배열 (arr)")

# 3x4 형태로 변경
reshaped_arr1 = arr.reshape(3, 4)
# display_array_info(reshaped_arr1, "arr.reshape(3, 4)")

# 2x2x3 형태로 변경 (-1 사용)
reshaped_arr2 = arr.reshape(2, 2, -1) # 마지막 차원 크기 자동 계산 (12 / (2*2) = 3)
# display_array_info(reshaped_arr2, "arr.reshape(2, 2, -1)")

# 1차원으로 펼치기 (ravel - 뷰 가능성 있음)
raveled_arr = reshaped_arr1.ravel()
# display_array_info(raveled_arr, "reshaped_arr1.ravel()")

# 1차원으로 펼치기 (flatten - 항상 복사본)
flattened_arr = reshaped_arr1.flatten()
# display_array_info(flattened_arr, "reshaped_arr1.flatten()")

# ravel() 예시: 원본 배열의 뷰가 될 수 있음
# raveled_arr[0] = 100
# print("ravel 후 원본 reshaped_arr1[0,0]:", reshaped_arr1[0,0]) # 100으로 변경됨

# flatten() 예시: 항상 복사본
# flattened_arr[1] = 200
# print("flatten 후 원본 reshaped_arr1[0,1]:", reshaped_arr1[0,1]) # 변경되지 않음
"""
st.code(code_reshape, language='python')
if st.checkbox("형태 변경 예시 (`reshape`, `ravel`, `flatten`) 보기", key="reshape_page"):
    orig_arr_ex = np.arange(1, 10) # 1~9
    display_array_info(orig_arr_ex, "원본 배열 (1~9)")

    reshaped_ex = orig_arr_ex.reshape(3, 3)
    display_array_info(reshaped_ex, "3x3으로 reshape")

    st.write("`reshape(-1, 1)`: 열 벡터로 변환 (Nx1 형태)")
    col_vector_ex = orig_arr_ex.reshape(-1, 1)
    display_array_info(col_vector_ex, "열 벡터로 reshape")

    st.write("`reshape(1, -1)`: 행 벡터로 변환 (1xN 형태)")
    row_vector_ex = orig_arr_ex.reshape(1, -1)
    display_array_info(row_vector_ex, "행 벡터로 reshape")

    multi_dim_arr = np.arange(24).reshape(2,3,4)
    display_array_info(multi_dim_arr, "다차원 배열 (2x3x4)")
    raveled_ex = multi_dim_arr.ravel()
    display_array_info(raveled_ex, "`ravel()` 결과 (1D)")
    flattened_ex = multi_dim_arr.flatten(order='F') # Fortran 스타일 (열 우선)
    display_array_info(flattened_ex, "`flatten(order='F')` 결과 (1D, 열 우선)")

st.markdown("---")

# --- 3.4 배열 전치 (`transpose()`, `.T`) ---
st.subheader("3.4 배열 전치 (`transpose()`, `.T`)")
st.markdown("""
배열의 축을 바꾸는 연산입니다. 2차원 배열의 경우 행과 열을 바꿉니다.
- `ndarray.transpose(*axes)` 또는 `np.transpose(array, axes=None)`: 축의 순서를 지정하여 전치합니다. `axes`가 없으면 축의 순서를 반대로 합니다 (예: (0,1,2) -> (2,1,0)). 2D의 경우 행렬 전치.
- `ndarray.T`: `ndarray.transpose()`의 축 지정 없는 간단한 형태.
전치는 항상 원본 배열과 데이터를 공유하는 뷰를 반환합니다.
""")
code_transpose = """
import numpy as np

arr2d = np.array([[1, 2, 3], [4, 5, 6]])
# display_array_info(arr2d, "원본 2D 배열 (arr2d)")

# .T 속성을 사용한 전치
transposed_t = arr2d.T
# display_array_info(transposed_t, "arr2d.T")

# transpose() 메소드를 사용한 전치 (2D에서는 .T와 동일)
transposed_method = arr2d.transpose()
# display_array_info(transposed_method, "arr2d.transpose()")

# 3차원 배열 예시
arr3d = np.arange(24).reshape(2, 3, 4) # 2개의 (3x4) 행렬
# display_array_info(arr3d, "원본 3D 배열 (arr3d) - shape (2,3,4)")

# 축 순서 변경: (2,3,4) -> (4,3,2) (기본 전치)
transposed_3d_default = arr3d.T # arr3d.transpose()와 동일
# display_array_info(transposed_3d_default, "arr3d.T - shape (4,3,2)")

# 축 순서 명시적 지정: (2,3,4) -> (0,2,1) 즉 (2,4,3)
transposed_3d_custom = arr3d.transpose(0, 2, 1)
# display_array_info(transposed_3d_custom, "arr3d.transpose(0, 2, 1) - shape (2,4,3)")
"""
st.code(code_transpose, language='python')
if st.checkbox("배열 전치 예시 보기", key="transpose_page"):
    arr2d_ex = np.arange(1, 7).reshape(2, 3)
    display_array_info(arr2d_ex, "원본 2D 배열 `arr2d_ex`")
    display_array_info(arr2d_ex.T, "`arr2d_ex.T` (전치된 배열)")

    arr3d_ex = np.arange(1, 25).reshape(2,3,4) # 2개의 (3x4) 행렬
    display_array_info(arr3d_ex, "원본 3D 배열 `arr3d_ex` (2,3,4)")
    # (0,1,2) -> (1,2,0) : (3,4,2) 형태로 변경
    transposed_3d_ex = arr3d_ex.transpose(1,2,0)
    display_array_info(transposed_3d_ex, "`arr3d_ex.transpose(1,2,0)` (3,4,2)")


# (이후 페이지들은 이어서 작성)