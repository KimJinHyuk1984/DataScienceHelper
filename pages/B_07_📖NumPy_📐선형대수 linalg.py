# pages/7_📐_선형대수_linalg.py
import streamlit as st
import numpy as np
from utils.utils_numpy import display_array_info

st.header("7. 선형 대수 (`numpy.linalg`)")
st.markdown("""
NumPy의 `linalg` 모듈은 선형 대수의 핵심 기능들을 제공합니다.
행렬 곱, 역행렬, 행렬식, 고유값/고유벡터 계산, 선형 방정식 풀이 등 다양한 연산을 수행할 수 있습니다.
""")

# 예제 행렬
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])
C_nonsquare = np.array([[1,2,3],[4,5,6]]) # 정방행렬이 아닌 경우
vec1 = np.array([1,2])
vec2 = np.array([3,4])

st.subheader("예제 행렬 및 벡터 확인")
if st.checkbox("선형대수 예제용 배열 보기", key="show_linalg_base_arrays_page"):
    display_array_info(A, "행렬 `A` (2x2)")
    display_array_info(B, "행렬 `B` (2x2)")
    display_array_info(C_nonsquare, "행렬 `C_nonsquare` (2x3)")
    display_array_info(vec1, "벡터 `vec1`")
    display_array_info(vec2, "벡터 `vec2`")

st.markdown("---")

# --- 7.1 점곱 및 행렬 곱 (Dot Product and Matrix Multiplication) ---
st.subheader("7.1 점곱 및 행렬 곱")
st.markdown("""
- `np.dot(a, b)`: 두 배열의 점곱(dot product).
  - 1D 배열(벡터) 간: 내적 (스칼라 반환).
  - 2D 배열(행렬) 간: 행렬 곱.
  - N-D 배열과 M-D 배열: 복잡한 규칙 따름 (일반적으로 마지막 차원의 내적).
- `@` 연산자: Python 3.5+ 에서 행렬 곱을 위한 중위 연산자로 도입됨. `np.matmul(a, b)`와 동일. 2D 배열에 주로 사용.
""")
code_dot_matmul = """
import numpy as np

v1 = np.array([1, 2, 3])
v2 = np.array([4, 5, 6])

# 벡터 내적
dot_product_vectors = np.dot(v1, v2) # 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
# print(f"벡터 내적 (np.dot(v1, v2)): {dot_product_vectors}")

matrix_A = np.array([[1, 2], [3, 4]])
matrix_B = np.array([[5, 0], [0, 5]])

# 행렬 곱 (np.dot)
matrix_product_dot = np.dot(matrix_A, matrix_B)
# print(f"행렬 곱 (np.dot(A, B)):\\n{matrix_product_dot}")

# 행렬 곱 (@ 연산자)
matrix_product_at = matrix_A @ matrix_B
# print(f"행렬 곱 (A @ B):\\n{matrix_product_at}")

# np.matmul도 사용 가능
matrix_product_matmul = np.matmul(matrix_A, matrix_B)
# print(f"행렬 곱 (np.matmul(A, B)):\\n{matrix_product_matmul}")
"""
st.code(code_dot_matmul, language='python')
if st.checkbox("점곱 및 행렬 곱 예시 보기", key="dot_matmul_page"):
    st.write("#### 벡터 내적")
    display_array_info(vec1, "벡터 `vec1`")
    display_array_info(vec2, "벡터 `vec2`")
    dot_v = np.dot(vec1, vec2)
    st.write(f"`np.dot(vec1, vec2)`: {dot_v}")

    st.write("#### 행렬 곱")
    display_array_info(A, "행렬 `A`")
    display_array_info(B, "행렬 `B`")
    prod_dot = np.dot(A, B)
    display_array_info(prod_dot, "`np.dot(A, B)` 결과")
    prod_at = A @ B
    display_array_info(prod_at, "`A @ B` 결과")

st.markdown("---")

# --- 7.2 행렬식 (Determinant) ---
st.subheader("7.2 행렬식 (`np.linalg.det`)")
st.markdown("정방 행렬(square matrix)의 행렬식을 계산합니다.")
code_determinant = """
import numpy as np

matrix = np.array([[1, 2], [3, 4]]) # (1*4) - (2*3) = 4 - 6 = -2

det_value = np.linalg.det(matrix)
# print(f"행렬:\\n{matrix}")
# print(f"행렬식 (np.linalg.det(matrix)): {det_value:.2f}") # 소수점 둘째자리
"""
st.code(code_determinant, language='python')
if st.checkbox("행렬식 계산 예시 보기", key="determinant_page"):
    display_array_info(A, "행렬 `A`")
    det_A = np.linalg.det(A)
    st.write(f"`np.linalg.det(A)`: {det_A:.2f}") # -2.00

    matrix_singular = np.array([[1,2],[2,4]]) # 행렬식이 0인 특이 행렬
    display_array_info(matrix_singular, "특이 행렬 (행렬식 0)")
    det_singular = np.linalg.det(matrix_singular)
    st.write(f"`np.linalg.det(matrix_singular)`: {det_singular:.2f}") # 0.00

st.markdown("---")

# --- 7.3 역행렬 (Inverse of a Matrix) ---
st.subheader("7.3 역행렬 (`np.linalg.inv`)")
st.markdown("정방 행렬의 역행렬을 계산합니다. 행렬식이 0이 아닌 경우(비특이 행렬, non-singular)에만 존재합니다.")
code_inverse = """
import numpy as np

matrix = np.array([[1, 2], [3, 4]])

# 역행렬 계산
inv_matrix = np.linalg.inv(matrix)
# print(f"원본 행렬:\\n{matrix}")
# print(f"역행렬 (np.linalg.inv(matrix)):\\n{np.round(inv_matrix, 2)}")

# 원본 행렬과 역행렬의 곱은 단위 행렬에 가까워야 함
identity_check = np.dot(matrix, inv_matrix)
# print(f"원본 @ 역행렬 (단위 행렬 근사값):\\n{np.round(identity_check, 2)}")
"""
st.code(code_inverse, language='python')
if st.checkbox("역행렬 계산 예시 보기", key="inverse_page"):
    display_array_info(A, "행렬 `A`") # A는 페이지 상단에서 정의된 예제 행렬
    try:
        inv_A = np.linalg.inv(A)
        display_array_info(inv_A, "`np.linalg.inv(A)` 결과 (A의 역행렬)")
        identity_A = A @ inv_A
        # 부동소수점 오차를 고려하여 np.round 또는 np.allclose로 비교
        display_array_info(np.round(identity_A, decimals=10), "`A @ inv_A` (단위행렬 근사)")
    except np.linalg.LinAlgError as e:
        st.error(f"역행렬 계산 오류: {e}")

    matrix_singular_inv = np.array([[1,2],[2,4]]) # 행렬식이 0인 특이 행렬
    display_array_info(matrix_singular_inv, "특이 행렬 `matrix_singular_inv` (역행렬 없음)")
    st.write("특이 행렬의 역행렬 계산 시도:")
    try:
        inv_singular = np.linalg.inv(matrix_singular_inv)
        display_array_info(inv_singular, "역행렬 (이 부분은 실행되지 않아야 함)")
    except np.linalg.LinAlgError as e:
        # "Singular matrix" 문자열이 에러 메시지에 포함되어 있는지 확인
        if "Singular matrix" in str(e).lower(): # .lower()로 대소문자 구분 없이 확인
            st.error("역행렬 계산 오류 (예상된 결과): 해당 행렬은 특이 행렬(singular matrix)이므로 역행렬이 존재하지 않습니다.")
        else:
            # 다른 종류의 선형대수 에러일 경우 원래 에러 메시지 표시
            st.error(f"선형대수 계산 오류: {e}")


st.markdown("---")

# --- 7.4 선형 방정식 풀이 (Solving Linear Systems) ---
st.subheader("7.4 선형 방정식 풀이 (`np.linalg.solve`)")
st.markdown("""
$Ax = b$ 형태의 선형 방정식을 풉니다. 여기서 $A$는 계수 행렬 (정방 행렬), $x$는 미지수 벡터, $b$는 결과 벡터입니다.
$A$는 비특이 행렬이어야 합니다.
""")
code_solve_linear_eq = """
import numpy as np

# 방정식 시스템:
# 2x + y = 5
# x + 3y = 5

# 계수 행렬 A
A_coeffs = np.array([[2, 1], [1, 3]])
# 결과 벡터 b
b_results = np.array([5, 5])

# 해 x 계산 (x = A^-1 * b 와 동일)
x_solution = np.linalg.solve(A_coeffs, b_results)
# print(f"계수 행렬 A:\\n{A_coeffs}")
# print(f"결과 벡터 b: {b_results}")
# print(f"해 x (np.linalg.solve(A, b)): {x_solution}") # 예상 결과: [2. 1.] (즉, x=2, y=1)

# 검증: A @ x_solution 이 b_results와 같은지 확인
# print(f"검증 (A @ x): {np.dot(A_coeffs, x_solution)}")
"""
st.code(code_solve_linear_eq, language='python')
if st.checkbox("선형 방정식 풀이 예시 보기", key="solve_linear_eq_page"):
    # 3x + 2y = 8
    # x  + 4y = 9
    A_eq_ex = np.array([[3, 2], [1, 4]])
    b_eq_ex = np.array([8, 9])
    display_array_info(A_eq_ex, "계수 행렬 `A_eq_ex`")
    display_array_info(b_eq_ex, "결과 벡터 `b_eq_ex`")
    try:
        x_sol_ex = np.linalg.solve(A_eq_ex, b_eq_ex)
        display_array_info(x_sol_ex, "해 `x_sol_ex` ([x, y])") # 예상: [1.4, 1.9]
        st.write(f"검증 (A @ x): `{np.round(A_eq_ex @ x_sol_ex, decimals=10)}` (b와 일치해야 함)")
    except np.linalg.LinAlgError as e:
        st.error(f"방정식 풀이 오류: {e}")

st.markdown("---")

# --- 7.5 고유값 및 고유벡터 (Eigenvalues and Eigenvectors) ---
st.subheader("7.5 고유값 및 고유벡터 (`np.linalg.eig`)")
st.markdown("""
정방 행렬 $A$에 대해 $Ax = \lambda x$ 를 만족하는 스칼라 $\lambda$ (고유값)와 벡터 $x$ (고유벡터)를 계산합니다.
`np.linalg.eig(A)`는 고유값들의 배열과 고유벡터들로 이루어진 행렬(각 열이 고유벡터)을 튜플로 반환합니다.
""")
code_eigen = """
import numpy as np

matrix = np.array([[1, -1], [1, 1]])
# 이 행렬의 고유값은 1+i, 1-i 입니다.

# 고유값과 고유벡터 계산
eigenvalues, eigenvectors = np.linalg.eig(matrix)
# print(f"행렬:\\n{matrix}")
# print(f"고유값: {eigenvalues}")
# print(f"고유벡터 (각 열이 하나의 고유벡터):\\n{eigenvectors}")

# 검증: A @ v = lambda * v (여기서 v는 eigenvectors의 열)
# for i in range(len(eigenvalues)):
#     Av = np.dot(matrix, eigenvectors[:, i])
#     lambda_v = eigenvalues[i] * eigenvectors[:, i]
#     # print(f"A @ v{i+1}: {np.round(Av, 3)}")
#     # print(f"lambda{i+1} * v{i+1}: {np.round(lambda_v, 3)}")
#     # print(f"일치 여부 (근사): {np.allclose(Av, lambda_v)}") # 부동소수점 비교
"""
st.code(code_eigen, language='python')
if st.checkbox("고유값/고유벡터 계산 예시 보기", key="eigen_page"):
    # 대칭 행렬은 실수 고유값을 가짐
    A_sym = np.array([[4, 1], [1, 3]])
    display_array_info(A_sym, "대칭 행렬 `A_sym`")
    
    eigenvalues_sym, eigenvectors_sym = np.linalg.eig(A_sym)
    display_array_info(eigenvalues_sym, "고유값 (`eigenvalues_sym`)")
    display_array_info(eigenvectors_sym, "고유벡터 (`eigenvectors_sym` - 각 열이 고유벡터)")

    st.write("검증 (A @ v = lambda * v):")
    for i in range(len(eigenvalues_sym)):
        lambda_val = eigenvalues_sym[i]
        eigen_vec = eigenvectors_sym[:, i] # i번째 열이 i번째 고유값에 해당하는 고유벡터
        
        Av = A_sym @ eigen_vec
        lambda_v = lambda_val * eigen_vec
        
        st.write(f"고유값 $\lambda_{i+1}$ = {lambda_val:.3f}")
        display_array_info(eigen_vec, f"  대응 고유벡터 $v_{i+1}$")
        display_array_info(np.round(Av,3), f"  $A \cdot v_{i+1}$")
        display_array_info(np.round(lambda_v,3), f"  $\lambda_{i+1} \cdot v_{i+1}$")
        st.write(f"  일치 여부 (np.allclose): `{np.allclose(Av, lambda_v)}`")
        st.markdown("---")


st.markdown("`numpy.linalg` 모듈에는 이 외에도 특이값 분해(SVD), 행렬 랭크, QR 분해 등 다양한 고급 선형대수 함수들이 포함되어 있습니다.")