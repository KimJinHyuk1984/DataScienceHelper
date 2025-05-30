# pages/8_💾_파일_입출력.py
import streamlit as st
import numpy as np
import os # 파일 존재 여부 확인 및 삭제용
from utils.utils_numpy import display_array_info

st.header("8. NumPy 배열 파일 입출력")
st.markdown("""
NumPy 배열을 파일로 저장하고 불러오는 여러 가지 방법을 제공합니다.
이를 통해 데이터를 영구적으로 저장하거나 다른 프로그램과 데이터를 교환할 수 있습니다.
""")

# --- 8.1 바이너리 `.npy` 파일 (단일 배열) ---
st.subheader("8.1 바이너리 `.npy` 파일 (`np.save`, `np.load`)")
st.markdown("""
- `np.save(filename, arr)`: NumPy 배열 하나를 바이너리 `.npy` 파일로 저장합니다. 확장자는 자동으로 `.npy`가 붙습니다.
- `np.load(filename)`: `.npy` 파일에서 배열을 불러옵니다.
이 형식은 빠르고 효율적이며, 배열의 형태와 데이터 타입을 그대로 유지합니다.
""")
code_npy_files = """
import numpy as np
import os # 파일 작업용

# 예제 배열
arr_to_save = np.arange(10, 20).reshape(2, 5)
filename_npy = 'my_array.npy' # 저장할 파일 이름

# 배열 저장
np.save(filename_npy, arr_to_save)
# print(f"'{filename_npy}' 파일로 배열 저장 완료.")

# 배열 불러오기
loaded_arr_npy = np.load(filename_npy)
# print(f"'{filename_npy}' 파일에서 배열 불러오기 완료:")
# print(loaded_arr_npy)

# 파일 삭제 (예제 실행 후 정리)
# if os.path.exists(filename_npy):
#     os.remove(filename_npy)
#     print(f"'{filename_npy}' 파일 삭제 완료.")
"""
st.code(code_npy_files, language='python')
if st.checkbox("`.npy` 파일 저장/불러오기 예시 실행", key="npy_files_page"):
    arr_to_save_ex = np.array([[1.5, 2.0, 3.1], [4.6, 5.9, 6.2]], dtype=np.float32)
    filename_npy_ex = 'example_array.npy'

    display_array_info(arr_to_save_ex, "저장할 배열 `arr_to_save_ex`")

    # 배열 저장
    np.save(filename_npy_ex, arr_to_save_ex)
    st.success(f"배열이 `{filename_npy_ex}` 파일로 저장되었습니다.")

    # 배열 불러오기
    if os.path.exists(filename_npy_ex):
        loaded_arr_ex = np.load(filename_npy_ex)
        st.success(f"`{filename_npy_ex}` 파일에서 배열을 성공적으로 불러왔습니다.")
        display_array_info(loaded_arr_ex, "불러온 배열 `loaded_arr_ex`")

        # 원본과 동일한지 확인
        if np.array_equal(arr_to_save_ex, loaded_arr_ex):
            st.info("저장된 배열과 불러온 배열이 동일합니다.")
        else:
            st.warning("저장된 배열과 불러온 배열이 다릅니다.")
        
        # 예제 실행 후 생성된 파일 삭제
        try:
            os.remove(filename_npy_ex)
            st.caption(f"`{filename_npy_ex}` 파일이 정리되었습니다.")
        except Exception as e:
            st.caption(f"`{filename_npy_ex}` 파일 삭제 중 오류: {e}")
    else:
        st.error(f"`{filename_npy_ex}` 파일이 생성되지 않았습니다.")


st.markdown("---")

# --- 8.2 압축된 `.npz` 파일 (여러 배열) ---
st.subheader("8.2 압축된 `.npz` 파일 (`np.savez`, `np.savez_compressed`, `np.load`)")
st.markdown("""
- `np.savez(filename, name1=arr1, name2=arr2, ...)`: 여러 개의 NumPy 배열을 하나의 압축되지 않은 `.npz` 파일로 저장합니다. 각 배열은 지정된 이름(키)으로 저장됩니다.
- `np.savez_compressed(filename, name1=arr1, ...)`: 여러 배열을 압축된 `.npz` 파일로 저장합니다. 용량이 줄어들지만, 저장/불러오기 시간이 약간 더 걸릴 수 있습니다.
- `np.load(filename)`: `.npz` 파일에서 배열들을 불러옵니다. 반환되는 객체는 딕셔너리 유사 객체로, 저장 시 사용한 이름(키)으로 각 배열에 접근할 수 있습니다.
""")
code_npz_files = """
import numpy as np
import os

# 예제 배열들
arr_x = np.arange(10)
arr_y = np.linspace(0, 1, 5)
filename_npz = 'my_arrays.npz' # 저장할 파일 이름 (압축 안됨)
filename_npzc = 'my_arrays_compressed.npz' # 저장할 파일 이름 (압축됨)


# 여러 배열 저장 (압축 안 함)
# 키워드 인자로 배열 이름 지정: x_data=arr_x, y_data=arr_y
np.savez(filename_npz, x_data=arr_x, y_data=arr_y)
# print(f"'{filename_npz}' 파일로 여러 배열 저장 완료.")

# 여러 배열 저장 (압축 함)
np.savez_compressed(filename_npzc, x_val=arr_x, y_val=arr_y, z_val=np.array([[1,2],[3,4]]))
# print(f"'{filename_npzc}' 파일로 여러 배열 압축 저장 완료.")


# .npz 파일에서 배열 불러오기
loaded_data_npz = np.load(filename_npz)
# print(f"'{filename_npz}' 파일 내용:")
# print(f"저장된 배열 이름(키): {list(loaded_data_npz.keys())}")
# loaded_x = loaded_data_npz['x_data']
# loaded_y = loaded_data_npz['y_data']
# print(f"불러온 x_data:\\n{loaded_x}")
# print(f"불러온 y_data:\\n{loaded_y}")
# loaded_data_npz.close() # 파일 닫기 (필수는 아님, with문 사용 권장)

# 파일 삭제
# if os.path.exists(filename_npz): os.remove(filename_npz)
# if os.path.exists(filename_npzc): os.remove(filename_npzc)
"""
st.code(code_npz_files, language='python')
if st.checkbox("`.npz` 파일 저장/불러오기 예시 실행", key="npz_files_page"):
    arr_x_ex = np.random.rand(5,2)
    arr_y_ex = np.random.randint(0,100, size=(3,3))
    filename_npz_ex = 'example_multi_array.npz'

    display_array_info(arr_x_ex, "저장할 배열 `arr_x_ex`")
    display_array_info(arr_y_ex, "저장할 배열 `arr_y_ex`")

    # 여러 배열 저장 (압축 안 함)
    np.savez(filename_npz_ex, first_array=arr_x_ex, second_array=arr_y_ex)
    st.success(f"여러 배열이 `{filename_npz_ex}` 파일로 저장되었습니다.")

    # .npz 파일에서 배열 불러오기
    if os.path.exists(filename_npz_ex):
        loaded_data = np.load(filename_npz_ex)
        st.success(f"`{filename_npz_ex}` 파일에서 데이터를 성공적으로 불러왔습니다.")
        st.write(f"저장된 배열의 이름(키): `{list(loaded_data.keys())}`")

        loaded_x = loaded_data['first_array']
        display_array_info(loaded_x, "불러온 배열 `first_array`")
        loaded_y = loaded_data['second_array']
        display_array_info(loaded_y, "불러온 배열 `second_array`")
        
        loaded_data.close() # 파일 핸들러 닫기

        # 예제 실행 후 생성된 파일 삭제
        try:
            os.remove(filename_npz_ex)
            st.caption(f"`{filename_npz_ex}` 파일이 정리되었습니다.")
        except Exception as e:
            st.caption(f"`{filename_npz_ex}` 파일 삭제 중 오류: {e}")
    else:
        st.error(f"`{filename_npz_ex}` 파일이 생성되지 않았습니다.")

st.markdown("---")

# --- 8.3 텍스트 파일 (`.txt`, `.csv`) ---
st.subheader("8.3 텍스트 파일 (`.txt`, `.csv`) (`np.loadtxt`, `np.savetxt`)")
st.markdown("""
- `np.savetxt(filename, arr, fmt='%.18e', delimiter=' ', newline='\\n', header='', footer='', comments='# ')`: 1차원 또는 2차원 배열을 텍스트 파일로 저장합니다. 복잡한 데이터 타입이나 3차원 이상 배열은 저장할 수 없습니다.
  - `fmt`: 출력 형식 (예: `'%d'` 정수, `'%.2f'` 소수점 둘째 자리).
  - `delimiter`: 구분자 (예: `,` for CSV, `\\t` for TSV).
- `np.loadtxt(filename, dtype=float, comments='#', delimiter=None, skiprows=0, usecols=None, unpack=False, max_rows=None)`: 텍스트 파일에서 데이터를 불러와 배열을 생성합니다.
이 방법은 사람이 읽을 수 있는 형태로 저장되지만, `.npy`나 `.npz`에 비해 느리고 용량이 클 수 있으며, 데이터 타입 정보가 손실될 수 있습니다.
""")
code_text_files = """
import numpy as np
import os

# 예제 배열 (2차원)
arr_to_txt = np.arange(1, 10).reshape(3, 3)
filename_txt = 'my_array.txt'
filename_csv = 'my_array.csv'

# 텍스트 파일로 저장 (공백 구분)
np.savetxt(filename_txt, arr_to_txt, fmt='%d', header='Col1 Col2 Col3', comments='File created by NumPy\\n')
# print(f"'{filename_txt}' 파일로 배열 저장 완료 (정수형, 공백 구분).")

# CSV 파일로 저장 (쉼표 구분, 소수점 2자리)
np.savetxt(filename_csv, arr_to_txt.astype(float)/2.0, fmt='%.2f', delimiter=',', header='Val1,Val2,Val3')
# print(f"'{filename_csv}' 파일로 배열 저장 완료 (실수형, 쉼표 구분).")


# 텍스트 파일 불러오기 (dtype 자동 추론 시도, 주석은 '#'으로 시작하는 줄)
loaded_arr_txt = np.loadtxt(filename_txt, dtype=int, comments='#', skiprows=2) # 헤더 2줄 건너뛰기
# print(f"'{filename_txt}' 파일에서 배열 불러오기 완료:")
# print(loaded_arr_txt)

# CSV 파일 불러오기 (구분자 명시)
loaded_arr_csv = np.loadtxt(filename_csv, dtype=float, delimiter=',', skiprows=1) # 헤더 1줄 건너뛰기
# print(f"'{filename_csv}' 파일에서 배열 불러오기 완료:")
# print(loaded_arr_csv)

# 파일 삭제
# if os.path.exists(filename_txt): os.remove(filename_txt)
# if os.path.exists(filename_csv): os.remove(filename_csv)
"""
st.code(code_text_files, language='python')
if st.checkbox("텍스트 파일 저장/불러오기 예시 실행", key="text_files_page"):
    arr_to_txt_ex = np.array([[10,20,30],[40,50,60],[70,80,90]])
    filename_txt_ex = 'example.txt'
    filename_csv_ex = 'example.csv'

    display_array_info(arr_to_txt_ex, "저장할 배열 `arr_to_txt_ex`")

    # .txt 파일로 저장 (공백 구분)
    header_txt = "X_Val Y_Val Z_Val"
    np.savetxt(filename_txt_ex, arr_to_txt_ex, fmt='%i', delimiter=' ', header=header_txt, comments='') # comments='' 로 # 제거
    st.success(f"배열이 `{filename_txt_ex}` 파일로 저장되었습니다.")
    with open(filename_txt_ex, 'r') as f:
        st.text_area(f"`{filename_txt_ex}` 파일 내용:", f.read(), height=150)

    # .csv 파일로 저장 (쉼표 구분, 실수형)
    header_csv = "ColA,ColB,ColC"
    np.savetxt(filename_csv_ex, arr_to_txt_ex / 10.0, fmt='%.1f', delimiter=',', header=header_csv, comments='')
    st.success(f"배열이 `{filename_csv_ex}` 파일로 저장되었습니다.")
    with open(filename_csv_ex, 'r') as f:
        st.text_area(f"`{filename_csv_ex}` 파일 내용:", f.read(), height=150)
    
    # 텍스트 파일 불러오기
    if os.path.exists(filename_txt_ex):
        # skiprows=1 로 헤더 한 줄만 건너뛰기
        loaded_txt = np.loadtxt(filename_txt_ex, dtype=int, delimiter=' ', skiprows=1)
        display_array_info(loaded_txt, f"`{filename_txt_ex}` 에서 불러온 배열 (skiprows=1)")
    
    if os.path.exists(filename_csv_ex):
        loaded_csv = np.loadtxt(filename_csv_ex, dtype=float, delimiter=',', skiprows=1)
        display_array_info(loaded_csv, f"`{filename_csv_ex}` 에서 불러온 배열 (skiprows=1)")

    # 예제 실행 후 생성된 파일 삭제
    for fname in [filename_txt_ex, filename_csv_ex]:
        try:
            if os.path.exists(fname):
                os.remove(fname)
                st.caption(f"`{fname}` 파일이 정리되었습니다.")
        except Exception as e:
            st.caption(f"`{fname}` 파일 삭제 중 오류: {e}")

st.markdown("---")
st.markdown("일반적으로 NumPy 배열을 저장할 때는 바이너리 형식(`.npy`, `.npz`)이 더 효율적입니다. 텍스트 파일은 다른 프로그램과의 호환성이나 사람이 직접 읽어야 할 필요가 있을 때 유용합니다.")