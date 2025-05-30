# pages/5_🧮_유니버설_함수_UFuncs.py
import streamlit as st
import numpy as np
from utils.utils_numpy import display_array_info

st.header("5. 유니버설 함수 (Universal Functions - UFuncs)")
st.markdown("""
유니버설 함수(UFuncs)는 NumPy 배열의 각 요소별(element-wise) 연산을 수행하는 함수입니다.
이 함수들은 내부적으로 C로 구현되어 있어 매우 빠르며, 반복문 없이 배열 전체에 대한 연산을 간결하게 표현할 수 있게 해줍니다.
UFuncs는 단항(unary) UFuncs (입력 배열 하나)와 이항(binary) UFuncs (입력 배열 두 개)로 나눌 수 있습니다.
""")

# 예제 배열
arr_a = np.array([1, 2, 3, 4])
arr_b = np.array([10, 20, 30, 40])
arr_c = np.array([-2.5, -1.0, 0.0, 1.7, 2.3, 3.9])
arr_2d = np.array([[1, 2, 3], [4, 5, 6]])
angles = np.array([0, np.pi/2, np.pi]) # 라디안 값

st.subheader("예제 배열 확인")
if st.checkbox("UFuncs 예제용 배열 보기", key="show_ufunc_base_arrays_page_5"): # 키 중복 방지
    display_array_info(arr_a, "`arr_a`")
    display_array_info(arr_b, "`arr_b`")
    display_array_info(arr_c, "`arr_c` (실수 및 음수 포함)")
    display_array_info(arr_2d, "`arr_2d`")
    display_array_info(angles, "`angles` (라디안 값)")

st.markdown("---")

# --- 5.1 산술 UFuncs ---
st.subheader("5.1 산술 UFuncs")
st.markdown("""
NumPy 배열 간의 기본적인 산술 연산을 수행합니다. 파이썬의 기본 산술 연산자(`+`, `-`, `*`, `/`, `**` 등)를 사용하거나, NumPy 함수(예: `np.add`, `np.subtract`)를 직접 호출할 수 있습니다.
""")
code_arithmetic_ufuncs = """
import numpy as np

a = np.array([1, 2, 3, 4])
b = np.array([10, 10, 10, 10])

# 덧셈
add_result1 = a + b         # 연산자 사용
add_result2 = np.add(a, b)  # np.add 함수 사용
# print(f"a + b = {add_result1}")

# 뺄셈
sub_result = a - 5         # 스칼라 값과의 연산 (브로드캐스팅)
# print(f"a - 5 = {sub_result}")

# 곱셈, 나눗셈, 거듭제곱, 나머지 등도 유사하게 사용 가능
# np.multiply(a,b), np.divide(b,a), np.power(a,2), np.mod(b,a)
# np.sqrt(a), np.exp(a), np.log(a) 등
"""
st.code(code_arithmetic_ufuncs, language='python')
if st.checkbox("산술 UFuncs 예시 보기", key="arithmetic_ufuncs_page_5"):
    display_array_info(arr_a, "배열 `arr_a`")
    display_array_info(arr_b, "배열 `arr_b`")

    st.write(f"`arr_a` + `arr_b` (또는 `np.add(arr_a, arr_b)`): `{arr_a + arr_b}`")
    st.write(f"`arr_a` - 1 (스칼라 연산): `{arr_a - 1}`")
    st.write(f"`arr_a` * `arr_b` (또는 `np.multiply(arr_a, arr_b)`): `{arr_a * arr_b}`")
    st.write(f"`arr_b` / `arr_a` (또는 `np.divide(arr_b, arr_a)`): `{arr_b / arr_a}`")
    st.write(f"`arr_a` ** 3 (또는 `np.power(arr_a, 3)`): `{arr_a ** 3}`")
    st.write(f"`np.negative(arr_a)` (부호 변경): `{np.negative(arr_a)}`")

st.markdown("---")

# --- 5.2 비교 및 논리 UFuncs ---
st.subheader("5.2 비교 및 논리 UFuncs")
st.markdown("""
배열 요소 간 비교 (`>`, `<`, `==` 등) 또는 논리 연산 (`&`, `|`, `~` 또는 `np.logical_and`, `np.logical_or`, `np.logical_not`)을 수행합니다. 결과는 불리언 배열입니다.
""")
code_compare_logical_ufuncs = """
import numpy as np

a = np.array([1, 5, 3, 8, 2])
b = np.array([0, 6, 3, 7, 9])

# 비교 연산
greater_than_3 = a > 3      # 결과: [False  True False  True False]
# print(f"a > 3: {greater_than_3}")
equal_to_b = a == b         # 결과: [False False  True False False]
# print(f"a == b: {equal_to_b}")

# 논리 연산
arr_bool1 = np.array([True, True, False, False])
arr_bool2 = np.array([True, False, True, False])
logical_and_result = np.logical_and(arr_bool1, arr_bool2) # 결과: [ True False False False]
# print(f"logical_and: {logical_and_result}")
"""
st.code(code_compare_logical_ufuncs, language='python')
if st.checkbox("비교 및 논리 UFuncs 예시 보기", key="compare_logical_ufuncs_page_5"):
    display_array_info(arr_a, "배열 `arr_a`")
    comp_arr_b = np.array([0,2,6,4]) # arr_a와 비교할 배열
    display_array_info(comp_arr_b, "배열 `comp_arr_b` (arr_a와 비교용)")


    st.write(f"`arr_a` > 2: `{arr_a > 2}`")
    st.write(f"`arr_a` == `comp_arr_b`: `{arr_a == comp_arr_b}`")
    st.write(f"`np.greater_equal(arr_a, 3)` (arr_a >= 3): `{np.greater_equal(arr_a, 3)}`")


    bool1_ex = np.array([True, False, True, False])
    bool2_ex = np.array([False, False, True, True])
    display_array_info(bool1_ex, "불리언 배열 `bool1_ex`")
    display_array_info(bool2_ex, "불리언 배열 `bool2_ex`")

    st.write(f"`np.logical_and(bool1_ex, bool2_ex)`: `{np.logical_and(bool1_ex, bool2_ex)}`")
    st.write(f"`np.logical_or(bool1_ex, bool2_ex)`: `{np.logical_or(bool1_ex, bool2_ex)}`")
    st.write(f"`np.logical_not(bool1_ex)`: `{np.logical_not(bool1_ex)}`")
    st.write(f"`np.logical_xor(bool1_ex, bool2_ex)`: `{np.logical_xor(bool1_ex, bool2_ex)}`")


st.markdown("---")

# --- 5.3 집계 함수 (Aggregation Functions) ---
st.subheader("5.3 집계 함수 (Aggregation Functions)")
st.markdown("""
배열 전체 또는 특정 축(axis)을 따라 통계량을 계산합니다.
- `arr.sum()`, `arr.min()`, `arr.max()`, `arr.mean()`, `arr.std()`, `arr.var()`, `arr.cumsum()`, `arr.cumprod()`
- `np.any(arr)`, `np.all(arr)`: 불리언 배열에 대한 연산.
`axis` 인자를 사용하여 연산을 수행할 축을 지정할 수 있습니다. (2D 배열 기준: `axis=0`은 열 방향, `axis=1`은 행 방향)
""")
code_aggregation_ufuncs = """
import numpy as np

arr = np.array([[1, 2, 3], [4, 5, 6]])
# display_array_info(arr, "원본 2D 배열")

# 전체 합계, 평균, 최소값, 최대값
# total_sum = arr.sum() # 21
# mean_val = arr.mean() # 3.5
# min_val = arr.min()   # 1
# max_val = arr.max()   # 6

# 각 열의 합계 (axis=0), 각 행의 합계 (axis=1)
# col_sum = arr.sum(axis=0) # [5 7 9]
# row_sum = arr.sum(axis=1) # [ 6 15]

# 누적 합계 (기본적으로 1차원으로 펼쳐서 계산)
# cumulative_sum_flat = arr.cumsum() # [ 1  3  6 10 15 21]
# 열 방향 누적 합계
# cumulative_sum_axis0 = arr.cumsum(axis=0)
# [[1 2 3]
#  [5 7 9]]
"""
st.code(code_aggregation_ufuncs, language='python')
if st.checkbox("집계 함수 예시 보기", key="aggregation_ufuncs_page_5"):
    display_array_info(arr_2d, "2D 배열 `arr_2d`")
    st.write(f"`arr_2d.sum()` (전체 합): {arr_2d.sum()}")
    st.write(f"`arr_2d.sum(axis=0)` (열별 합): `{arr_2d.sum(axis=0)}`")
    st.write(f"`arr_2d.sum(axis=1)` (행별 합): `{arr_2d.sum(axis=1)}`")
    st.write(f"`arr_2d.mean()` (전체 평균): {arr_2d.mean():.2f}") # 소수점 둘째자리까지
    st.write(f"`arr_2d.std(axis=0)` (열별 표준편차): `{arr_2d.std(axis=0)}`")
    st.write(f"`arr_2d.min(axis=1)` (행별 최소값): `{arr_2d.min(axis=1)}`")
    st.write(f"`arr_2d.argmax()` (전체 최대값의 1차원 인덱스): {arr_2d.argmax()}")
    st.write(f"`arr_2d.argmax(axis=0)` (열별 최대값의 행 인덱스): `{arr_2d.argmax(axis=0)}`")

    arr_cs_ex = np.array([1,2,3,4])
    display_array_info(arr_cs_ex, "1D 배열 `arr_cs_ex` (for cumsum/cumprod)")
    st.write(f"`arr_cs_ex.cumsum()` (누적 합): `{arr_cs_ex.cumsum()}`")
    st.write(f"`arr_cs_ex.cumprod()` (누적 곱): `{arr_cs_ex.cumprod()}`")

    bool_arr_ex = np.array([[True, False], [True, True]])
    display_array_info(bool_arr_ex, "불리언 배열 `bool_arr_ex`")
    st.write(f"`np.any(bool_arr_ex)` (하나라도 True인가?): {np.any(bool_arr_ex)}")
    st.write(f"`np.all(bool_arr_ex)` (모두 True인가?): {np.all(bool_arr_ex)}")
    st.write(f"`np.all(bool_arr_ex, axis=1)` (행별로 모두 True인가?): `{np.all(bool_arr_ex, axis=1)}`")

st.markdown("---")

# --- 5.4 삼각 함수 (Trigonometric Functions) ---
st.subheader("5.4 삼각 함수 (Trigonometric Functions)")
st.markdown("입력 배열의 각 요소에 대해 삼각 함수를 계산합니다. 입력값은 라디안(radian) 단위여야 합니다.")
code_trigonometric_ufuncs = """
import numpy as np

angles_rad = np.array([0, np.pi/6, np.pi/4, np.pi/3, np.pi/2, np.pi]) # 0, 30, 45, 60, 90, 180도

# 사인(sine) 함수
sin_values = np.sin(angles_rad)
# print(f"sin 값: {np.round(sin_values, 3)}") # 소수점 3자리 반올림

# 코사인(cosine) 함수
cos_values = np.cos(angles_rad)
# print(f"cos 값: {np.round(cos_values, 3)}")

# 탄젠트(tangent) 함수
tan_values = np.tan(angles_rad)
# print(f"tan 값: {np.round(tan_values, 3)}") # np.pi/2 에서 매우 큰 값(무한대에 가까움)

# 역삼각 함수: np.arcsin, np.arccos, np.arctan 등
# np.degrees(angles_rad) # 라디안을 각도로 변환
"""
st.code(code_trigonometric_ufuncs, language='python')
if st.checkbox("삼각 함수 예시 보기", key="trigonometric_ufuncs_page_5"):
    display_array_info(angles, "`angles` (라디안 값)")
    st.write(f"`np.sin(angles)`: `{np.round(np.sin(angles), decimals=3)}`")
    st.write(f"`np.cos(angles)`: `{np.round(np.cos(angles), decimals=3)}`")
    st.write(f"`np.tan(angles)` (pi/2 근처에서 매우 큼): `{np.round(np.tan(angles), decimals=3)}`")
    st.write(f"`np.degrees(angles)` (각도 변환): `{np.degrees(angles)}`")
    
    arcsin_input = np.array([-1, 0, 1])
    display_array_info(arcsin_input, "`arcsin_input`")
    st.write(f"`np.arcsin(arcsin_input)` (라디안): `{np.arcsin(arcsin_input)}`")

st.markdown("---")

# --- 5.5 지수 및 로그 함수 (Exponential and Logarithmic Functions) ---
st.subheader("5.5 지수 및 로그 함수")
st.markdown("지수, 로그, 제곱근 등의 연산을 수행합니다.")
code_exp_log_ufuncs = """
import numpy as np

arr = np.array([1, 2, 3, 4, 10])

# 지수 함수 (e^x)
exp_values = np.exp(arr)
# print(f"np.exp(arr): {np.round(exp_values, 2)}")

# 자연 로그 (밑이 e인 로그)
log_values = np.log(arr) # 0 또는 음수 입력 시 경고/에러 발생
# print(f"np.log(arr): {np.round(log_values, 2)}")

# 밑이 10인 상용 로그
log10_values = np.log10(arr)
# print(f"np.log10(arr): {np.round(log10_values, 2)}")

# 제곱근
sqrt_values = np.sqrt(arr)
# print(f"np.sqrt(arr): {np.round(sqrt_values, 2)}")

# 밑이 2인 로그
log2_values = np.log2(arr)
# print(f"np.log2(arr): {np.round(log2_values, 2)}")
"""
st.code(code_exp_log_ufuncs, language='python')
if st.checkbox("지수 및 로그 함수 예시 보기", key="exp_log_ufuncs_page_5"):
    display_array_info(arr_a, "배열 `arr_a` (1,2,3,4)")
    st.write(f"`np.exp(arr_a)` (e^x): `{np.round(np.exp(arr_a), 2)}`")
    st.write(f"`np.log(arr_a)` (자연 로그): `{np.round(np.log(arr_a), 2)}`")
    st.write(f"`np.log10(arr_a)` (상용 로그): `{np.round(np.log10(arr_a), 2)}`")
    st.write(f"`np.sqrt(arr_a)` (제곱근): `{np.round(np.sqrt(arr_a), 2)}`")
    st.write(f"`np.square(arr_a)` (제곱, a**2와 동일): `{np.square(arr_a)}`")


st.markdown("---")

# --- 5.6 반올림, 올림, 내림, 절대값 함수 ---
st.subheader("5.6 반올림, 올림, 내림, 절대값 함수")
st.markdown("""
- `np.round(arr, decimals=0)`: 지정된 소수점 자리에서 반올림합니다.
- `np.floor(arr)`: 각 요소보다 작거나 같은 가장 큰 정수 (내림).
- `np.ceil(arr)`: 각 요소보다 크거나 같은 가장 작은 정수 (올림).
- `np.abs(arr)` 또는 `np.absolute(arr)`: 각 요소의 절대값을 계산합니다. 복소수의 경우 크기(magnitude)를 반환.
- `np.fabs(arr)`: `np.abs`와 유사하나, 복소수를 처리하지 못하고 항상 float을 반환합니다.
""")
code_round_abs_ufuncs = """
import numpy as np

arr_float = np.array([-2.7, -1.5, 0.0, 1.5, 2.7, 3.14159])

# 반올림 (소수점 첫째 자리까지)
rounded_values = np.round(arr_float, decimals=1)
# print(f"np.round(arr_float, 1): {rounded_values}")

# 내림
floor_values = np.floor(arr_float)
# print(f"np.floor(arr_float): {floor_values}")

# 올림
ceil_values = np.ceil(arr_float)
# print(f"np.ceil(arr_float): {ceil_values}")

# 절대값
abs_values = np.abs(arr_float)
# print(f"np.abs(arr_float): {abs_values}")

# 복소수 절대값 (크기)
arr_complex = np.array([3 + 4j, -5 - 12j])
abs_complex = np.abs(arr_complex) # 결과: [5. 13.] (sqrt(3^2+4^2), sqrt((-5)^2+(-12)^2))
# print(f"np.abs(arr_complex): {abs_complex}")
"""
st.code(code_round_abs_ufuncs, language='python')
if st.checkbox("반올림, 올림, 내림, 절대값 함수 예시 보기", key="round_abs_ufuncs_page_5"):
    display_array_info(arr_c, "`arr_c`")
    st.write(f"`np.round(arr_c)` (정수 반올림): `{np.round(arr_c)}`")
    st.write(f"`np.round(arr_c, decimals=1)` (소수 첫째자리 반올림): `{np.round(arr_c, decimals=1)}`")
    st.write(f"`np.floor(arr_c)` (내림): `{np.floor(arr_c)}`")
    st.write(f"`np.ceil(arr_c)` (올림): `{np.ceil(arr_c)}`")
    st.write(f"`np.abs(arr_c)` (절대값): `{np.abs(arr_c)}`")
    st.write(f"`np.trunc(arr_c)` (소수점 이하 버림, 0을 향해 자름): `{np.trunc(arr_c)}`")

    complex_arr_ex = np.array([1+1j, -2-2j, 3-4j])
    display_array_info(complex_arr_ex, "`complex_arr_ex`")
    st.write(f"`np.abs(complex_arr_ex)` (복소수의 크기): `{np.abs(complex_arr_ex)}`")


st.markdown("---")
st.markdown("이 외에도 `np.where(condition, x, y)` (조건에 따라 x 또는 y에서 요소 선택), `np.isnan(arr)` (NaN 여부 확인), `np.isfinite(arr)` (유한수 여부 확인) 등 다양한 UFuncs가 있습니다.")