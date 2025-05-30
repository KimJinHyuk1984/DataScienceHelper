# pages/2_➕_배열_생성.py
import streamlit as st
import numpy as np
from utils.utils_numpy import display_array_info

st.header("2. NumPy 배열 생성하기")
st.markdown("NumPy는 다양한 방법으로 배열을 생성할 수 있는 함수들을 제공합니다.")

# --- 2.1 파이썬 리스트나 튜플로부터 생성 ---
st.subheader("2.1 `np.array()`: 파이썬 리스트나 튜플로부터 생성")
st.markdown("가장 기본적인 배열 생성 방법입니다. `dtype` 인자를 사용하여 데이터 타입을 명시적으로 지정할 수 있습니다.")
code_from_list = """
import numpy as np

# 1차원 배열 생성
list1 = [1, 2, 3, 4, 5]
arr1 = np.array(list1) # list1을 NumPy 배열로 변환
# display_array_info(arr1, "리스트로부터 생성된 1차원 배열")

# 2차원 배열 생성
list2 = [[1, 2, 3], [4, 5, 6]]
arr2 = np.array(list2) # 중첩 리스트를 2차원 NumPy 배열로 변환
# display_array_info(arr2, "중첩 리스트로부터 생성된 2차원 배열")

# 데이터 타입 지정
arr_float = np.array([1, 2, 3], dtype=float) # 요소를 float 타입으로 지정
# display_array_info(arr_float, "float 타입으로 생성된 배열")

arr_complex = np.array([1, 2, 3], dtype=complex) # 요소를 complex 타입으로 지정
# display_array_info(arr_complex, "complex 타입으로 생성된 배열")
"""
st.code(code_from_list, language='python')
if st.checkbox("`np.array()` 예시 보기", key="array_from_list_page"):
    list1_ex = [10.0, 20.5, 30.1]
    arr1_ex = np.array(list1_ex)
    display_array_info(arr1_ex, "리스트 `list1_ex`로부터 생성")

    list2_ex = ((1, 0, 0), (0, 1, 0), (0, 0, 1)) # 튜플의 튜플도 가능
    arr2_ex = np.array(list2_ex, dtype=np.int8)
    display_array_info(arr2_ex, "튜플 `list2_ex`로부터 생성 (int8 타입)")

st.markdown("---")

# --- 2.2 특정 값으로 채워진 배열 생성 ---
st.subheader("2.2 특정 값으로 채워진 배열 생성")
st.markdown("""
- `np.zeros(shape, dtype=float)`: 모든 요소가 0으로 채워진 배열을 생성합니다.
- `np.ones(shape, dtype=float)`: 모든 요소가 1로 채워진 배열을 생성합니다.
- `np.full(shape, fill_value, dtype=None)`: 지정된 값(`fill_value`)으로 채워진 배열을 생성합니다.
- `np.empty(shape, dtype=float)`: 초기화되지 않은 (임의의 값으로 채워진) 배열을 생성합니다. 빠르지만, 사용 전 요소 값을 직접 할당해야 합니다.
""")
code_filled_arrays = """
import numpy as np

# 2x3 형태의 0으로 채워진 배열
zeros_arr = np.zeros((2, 3))
# display_array_info(zeros_arr, "np.zeros((2, 3))")

# 3x2 형태의 1로 채워진 정수 배열
ones_arr = np.ones((3, 2), dtype=int)
# display_array_info(ones_arr, "np.ones((3, 2), dtype=int)")

# 2x2 형태의 7로 채워진 배열
full_arr = np.full((2, 2), 7.0)
# display_array_info(full_arr, "np.full((2, 2), 7.0)")

# 2x4 형태의 초기화되지 않은 배열 (내용은 예측 불가)
# 주의: empty_arr의 내용은 실행 시마다 다를 수 있으며, 의미 없는 값일 수 있습니다.
empty_arr = np.empty((2, 4))
# display_array_info(empty_arr, "np.empty((2, 4)) - 주의: 값은 임의적")
"""
st.code(code_filled_arrays, language='python')
if st.checkbox("특정 값으로 채워진 배열 예시 보기", key="filled_arrays_page"):
    zeros_arr_ex = np.zeros((3, 4), dtype=np.bool_) # bool 타입의 0 (False) 배열
    display_array_info(zeros_arr_ex, "np.zeros((3, 4), dtype=np.bool_)")

    ones_arr_ex = np.ones(5) # 1차원 배열, 기본 dtype은 float64
    display_array_info(ones_arr_ex, "np.ones(5)")

    fill_value_ex = 3.14
    full_arr_ex = np.full((2, 3, 2), fill_value_ex, dtype=np.float32) # 3차원 배열
    display_array_info(full_arr_ex, f"np.full((2, 3, 2), {fill_value_ex}, dtype=np.float32)")

    st.write("`np.empty()`는 배열을 위한 메모리 공간만 할당하고 초기화하지 않아 매우 빠릅니다. 하지만 요소 값은 예측할 수 없으므로 사용 전 반드시 값을 채워야 합니다.")
    empty_arr_ex = np.empty((2,2))
    display_array_info(empty_arr_ex, "np.empty((2,2)) - 값은 임의적")


st.markdown("---")

# --- 2.3 연속된 값 또는 특정 간격의 값으로 배열 생성 ---
st.subheader("2.3 연속된 값 또는 특정 간격의 값으로 배열 생성")
st.markdown("""
- `np.arange([start,] stop[, step,], dtype=None)`: 파이썬의 `range()`와 유사하지만, 실수 `step`도 가능하며 NumPy 배열을 반환합니다.
- `np.linspace(start, stop, num=50, endpoint=True, retstep=False, dtype=None)`: `start`와 `stop` 사이를 `num`개의 일정한 간격으로 나눈 값들로 배열을 생성합니다. `endpoint=False`이면 `stop`을 포함하지 않습니다.
- `np.logspace(start, stop, num=50, endpoint=True, base=10.0, dtype=None)`: 로그 스케일에서 `base**start` 부터 `base**stop` 까지 `num`개의 일정한 간격으로 나눈 값들로 배열을 생성합니다.
""")
code_sequence_arrays = """
import numpy as np

# 0부터 9까지 1씩 증가하는 배열 (10은 미포함)
arange_arr1 = np.arange(10)
# display_array_info(arange_arr1, "np.arange(10)")

# 2부터 10까지 2씩 증가하는 배열 (10은 미포함)
arange_arr2 = np.arange(2, 10, 2)
# display_array_info(arange_arr2, "np.arange(2, 10, 2)")

# 0부터 1까지 5개의 일정한 간격으로 나눈 배열 (1 포함)
linspace_arr = np.linspace(0, 1, 5)
# display_array_info(linspace_arr, "np.linspace(0, 1, 5)")

# 10^0 (1)부터 10^2 (100)까지 로그 스케일에서 3개의 일정한 간격으로 나눈 배열
logspace_arr = np.logspace(0, 2, 3) # 결과: [1., 10., 100.]
# display_array_info(logspace_arr, "np.logspace(0, 2, 3)")
"""
st.code(code_sequence_arrays, language='python')
if st.checkbox("연속/간격 값 배열 예시 보기", key="sequence_arrays_page"):
    arange_arr_ex = np.arange(0.5, 5.5, 0.5, dtype=float) # 실수 step 사용
    display_array_info(arange_arr_ex, "np.arange(0.5, 5.5, 0.5)")

    linspace_arr_ex, step_ex = np.linspace(0, 10, 11, endpoint=True, retstep=True) # endpoint 포함, step 값 반환
    display_array_info(linspace_arr_ex, "np.linspace(0, 10, 11, endpoint=True, retstep=True)")
    st.write(f"Linspace step: `{step_ex}`")

    logspace_arr_ex = np.logspace(1, 4, 4, base=2) # 2^1, 2^2, 2^3, 2^4
    display_array_info(logspace_arr_ex, "np.logspace(1, 4, 4, base=2)")

st.markdown("---")

# --- 2.4 기타 특수 배열 생성 ---
st.subheader("2.4 기타 특수 배열 생성")
st.markdown("""
- `np.eye(N, M=None, k=0, dtype=float)`: 주 대각선(diagonal)이 1이고 나머지는 0인 (N x M) 배열을 생성합니다. `M`이 None이면 N x N. `k`는 대각선의 위치 (0은 주 대각선, 양수는 위쪽, 음수는 아래쪽).
- `np.identity(n, dtype=float)`: N x N 단위 행렬(주 대각선이 1, 나머지는 0)을 생성합니다. `np.eye(N)`와 동일.
- `np.diag(v, k=0)`: `v`가 1차원 배열이면, `v`를 주 대각선으로 하는 2차원 배열을 생성. `v`가 2차원 배열이면, k번째 대각선 요소를 추출하여 1차원 배열로 반환.
""")
code_special_arrays = """
import numpy as np

# 3x3 단위 행렬
eye_arr = np.eye(3)
# display_array_info(eye_arr, "np.eye(3)")

# 4x4 단위 행렬 (identity)
identity_arr = np.identity(4)
# display_array_info(identity_arr, "np.identity(4)")

# 1차원 배열을 대각선으로 하는 2차원 배열 생성
diag_from_1d = np.diag([1, 2, 3, 4])
# display_array_info(diag_from_1d, "np.diag([1, 2, 3, 4])")

# 2차원 배열에서 주 대각선 요소 추출
matrix = np.array([[1,2,3],[4,5,6],[7,8,9]])
diag_from_2d = np.diag(matrix) # 결과: [1, 5, 9]
# display_array_info(diag_from_2d, "np.diag(matrix) from 2D matrix")
"""
st.code(code_special_arrays, language='python')
if st.checkbox("기타 특수 배열 예시 보기", key="special_arrays_page"):
    eye_arr_ex = np.eye(4, k=1) # 주 대각선보다 한 칸 위
    display_array_info(eye_arr_ex, "np.eye(4, k=1)")

    diag_arr_ex = np.diag([10, 20, 30], k=-1) # 주 대각선보다 한 칸 아래
    display_array_info(diag_arr_ex, "np.diag([10, 20, 30], k=-1)")

    existing_matrix = np.arange(1,10).reshape(3,3)
    st.write("기존 행렬:")
    st.write(existing_matrix)
    diag_extract_ex = np.diag(existing_matrix, k=1)
    display_array_info(diag_extract_ex, "np.diag(existing_matrix, k=1) - 대각선 추출")

st.markdown("---")

# --- 2.5 난수 배열 생성 (np.random) ---
st.subheader("2.5 난수 배열 생성 (`np.random`)")
st.markdown("""
NumPy의 `random` 모듈은 다양한 종류의 난수 배열을 생성하는 함수를 제공합니다.
- `np.random.rand(d0, d1, ..., dn)`: 0과 1 사이의 균일 분포에서 난수를 생성하여 주어진 형태의 배열을 만듭니다.
- `np.random.randn(d0, d1, ..., dn)`: 평균 0, 표준편차 1의 표준 정규 분포(가우시안 분포)에서 난수를 생성합니다.
- `np.random.randint(low, high=None, size=None, dtype=int)`: `low` (포함)와 `high` (미포함) 사이의 정수 난수를 생성합니다. `high`가 None이면 0부터 `low` (미포함) 사이.
- `np.random.seed(seed)`: 난수 생성기의 시드를 설정하여 재현 가능한 난수를 생성할 수 있게 합니다.
- `np.random.choice(a, size=None, replace=True, p=None)`: 주어진 1차원 배열 `a`에서 무작위로 샘플을 추출합니다.
- `np.random.permutation(x)`: 배열 `x`의 복사본을 무작위로 섞거나, 정수 `x`가 주어지면 `np.arange(x)`를 섞어서 반환합니다.
""")
code_random_arrays = """
import numpy as np

# 시드 설정 (결과 재현을 위함)
np.random.seed(42)

# 0과 1 사이의 균일 분포에서 2x3 배열 생성
rand_arr = np.random.rand(2, 3)
# display_array_info(rand_arr, "np.random.rand(2, 3)")

# 표준 정규 분포에서 3x2 배열 생성
randn_arr = np.random.randn(3, 2)
# display_array_info(randn_arr, "np.random.randn(3, 2)")

# 0부터 9 사이의 정수 난수 5개로 이루어진 1차원 배열 생성
randint_arr = np.random.randint(0, 10, size=5)
# display_array_info(randint_arr, "np.random.randint(0, 10, size=5)")

# 주어진 배열에서 3개의 샘플을 비복원 추출 (replace=False)
elements = [10, 20, 30, 40, 50]
choice_arr = np.random.choice(elements, size=3, replace=False)
# display_array_info(choice_arr, "np.random.choice(elements, size=3, replace=False)")

# 배열 섞기
arr_to_shuffle = np.arange(5)
shuffled_arr = np.random.permutation(arr_to_shuffle)
# display_array_info(shuffled_arr, "np.random.permutation(arr_to_shuffle)")
# print(f"원본 배열은 변경되지 않음: {arr_to_shuffle}")
"""
st.code(code_random_arrays, language='python')
if st.checkbox("난수 배열 예시 보기", key="random_arrays_page"):
    st.write("`np.random.seed()`를 사용하면 동일한 난수 시퀀스를 재현할 수 있습니다.")
    np.random.seed(123) # 시드 설정

    rand_arr_ex = np.random.rand(3,2,2) # 3x2x2 형태
    display_array_info(rand_arr_ex, "np.random.rand(3,2,2) (0~1 균일분포)")

    randn_arr_ex = np.random.randn(4) # 1차원
    display_array_info(randn_arr_ex, "np.random.randn(4) (표준정규분포)")

    randint_arr_ex = np.random.randint(100, 200, size=(2,3)) # 100~199 사이 정수
    display_array_info(randint_arr_ex, "np.random.randint(100, 200, size=(2,3))")

    elements_ex = np.array(['A', 'B', 'C', 'D', 'E'])
    # 확률을 지정하여 복원 추출
    choice_arr_ex = np.random.choice(elements_ex, size=10, replace=True, p=[0.1, 0.1, 0.5, 0.2, 0.1])
    display_array_info(choice_arr_ex, "np.random.choice() (확률 지정, 복원 추출)")

    perm_arr_ex = np.random.permutation(10) # 0~9까지의 숫자를 섞음
    display_array_info(perm_arr_ex, "np.random.permutation(10)")