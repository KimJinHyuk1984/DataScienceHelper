# pages/2_💾_데이터_입출력.py
import streamlit as st
import pandas as pd
import numpy as np
import io # StringIO를 사용하여 문자열을 파일처럼 다루기 위함
from utils.utils_pandas import display_dataframe_info

st.header("2. 데이터 입출력 (I/O)")
st.markdown("""
Pandas는 다양한 파일 형식으로부터 데이터를 읽어오거나 파일로 저장하는 강력한 기능을 제공합니다.
가장 흔하게 사용되는 CSV와 Excel 파일 입출력을 중심으로 살펴보겠습니다.
""")

# --- 예제 DataFrame 생성 (모든 예시에서 사용) ---
@st.cache_data # 데이터프레임 생성을 캐싱하여 반복 실행 방지
def create_sample_io_df():
    data = {
        'Name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
        'Age': [25, 30, np.nan, 28, 22], # 결측치 포함
        'City': ['New York', 'Paris', 'London', 'Tokyo', 'Seoul'],
        'JoinDate': pd.to_datetime(['2021-01-10', '2020-05-15', '2022-03-01', '2021-08-20', '2023-01-05']),
        'Salary': [70000, 80000, 65000, 90000, np.nan] # 결측치 포함
    }
    return pd.DataFrame(data)

sample_df_io = create_sample_io_df()

st.subheader("입출력 예제용 DataFrame 확인")
if st.checkbox("입출력 예제 DataFrame 보기", key="show_io_base_df_page"):
    display_dataframe_info(sample_df_io, "예제 DataFrame (sample_df_io)", max_rows_to_display=5)

st.markdown("---")

# --- 2.1 CSV 파일 읽기 (`pd.read_csv()`) ---
st.subheader("2.1 CSV 파일 읽기 (`pd.read_csv()`)")
st.markdown("""
쉼표로 구분된 값(Comma-Separated Values) 형식의 파일을 읽어 DataFrame으로 변환합니다.
주요 파라미터:
- `filepath_or_buffer`: 파일 경로, URL, 또는 파일 객체.
- `sep` (또는 `delimiter`): 구분자 (기본값: `,`).
- `header`: 헤더로 사용할 행 번호 (기본값: `0`, 첫 번째 행을 헤더로 사용). 헤더가 없으면 `None`.
- `index_col`: 인덱스로 사용할 열 번호 또는 이름.
- `usecols`: 읽어올 열의 리스트 또는 이름.
- `dtype`: 각 열의 데이터 타입을 지정하는 딕셔너리.
- `parse_dates`: 날짜/시간으로 변환할 열의 리스트 또는 이름.
- `na_values`: 특정 값을 NaN으로 처리할 문자열 또는 리스트.
""")

# Streamlit 앱 내에서 실행 가능한 CSV 읽기 예제를 위해 StringIO 사용
csv_data_string = """Name,Age,City,JoinDate,Salary
Alice,25,New York,2021-01-10,70000
Bob,30,Paris,2020-05-15,80000
Charlie,,London,2022-03-01,65000
David,28,Tokyo,2021-08-20,90000
Eve,22,Seoul,2023-01-05,
""" # Eve의 Salary는 빈 문자열로 결측치 표현 가능

code_read_csv = f"""
import pandas as pd
import io # 문자열을 파일처럼 다루기 위함

# 가상의 CSV 데이터 (실제로는 파일 경로 사용)
csv_string = \"\"\"
{csv_data_string.strip()}
\"\"\"

# 문자열 데이터를 파일처럼 읽기 위해 StringIO 사용
csv_file_like = io.StringIO(csv_string)

# CSV 데이터 읽기
df_from_csv = pd.read_csv(
    csv_file_like,             # 파일 객체 (또는 파일 경로)
    sep=',',                   # 구분자
    header=0,                  # 첫 번째 줄을 헤더로 사용
    # index_col='Name',        # 'Name' 컬럼을 인덱스로 사용 (선택 사항)
    parse_dates=['JoinDate'],  # 'JoinDate' 컬럼을 datetime으로 파싱
    na_values=['', 'NA', 'N/A'] # 빈 문자열, 'NA', 'N/A'를 NaN으로 처리
)

# display_dataframe_info(df_from_csv, "CSV 파일로부터 읽어온 DataFrame")
"""
st.code(code_read_csv, language='python')

if st.checkbox("`pd.read_csv()` 예시 보기", key="read_csv_page"):
    csv_file_like_ex = io.StringIO(csv_data_string) # 예제 실행 시마다 StringIO 재생성
    df_from_csv_ex = pd.read_csv(
        csv_file_like_ex,
        sep=',',
        header=0,
        # index_col=0, # 첫번째 열을 인덱스로 (이 예제에서는 Name이 첫번째)
        parse_dates=['JoinDate'],
        na_values=['', 'NA', 'N/A'] # 빈 문자열과 'NA'를 NaN으로 인식
    )
    display_dataframe_info(df_from_csv_ex, "CSV 데이터로부터 읽어온 DataFrame", max_rows_to_display=5)
    st.write("읽어온 후 `Age`와 `Salary` 컬럼의 NaN 값 확인:")
    st.dataframe(df_from_csv_ex.isnull().sum().rename("결측치 수"))


st.markdown("---")

# --- 2.2 CSV 파일 저장 (`df.to_csv()`) ---
st.subheader("2.2 CSV 파일 저장 (`df.to_csv()`)")
st.markdown("""
DataFrame을 CSV 파일로 저장합니다.
주요 파라미터:
- `path_or_buf`: 저장할 파일 경로 또는 파일 객체.
- `sep`: 구분자 (기본값: `,`).
- `index`: 인덱스를 파일에 포함할지 여부 (기본값: `True`).
- `header`: 헤더(컬럼 이름)를 파일에 포함할지 여부 (기본값: `True`).
- `na_rep`: NaN 값을 대체할 문자열 (기본값: 빈 문자열 `''`).
- `encoding`: 파일 인코딩 (예: `'utf-8'`, `'cp949'`).
""")

code_to_csv = """
import pandas as pd
import io

# 예제 DataFrame (sample_df_io 사용 가정)
# sample_df_io = pd.DataFrame(...)

# DataFrame을 CSV 문자열로 저장 (실제로는 파일 경로 지정)
csv_output_buffer = io.StringIO()
sample_df_io.to_csv(
    csv_output_buffer,
    sep='|',          # 구분자를 파이프(|)로 변경
    index=False,      # 인덱스는 저장하지 않음
    header=True,      # 헤더는 저장함
    na_rep='NULL',    # NaN 값은 'NULL' 문자열로 대체
    encoding='utf-8'
)
csv_string_output = csv_output_buffer.getvalue()
# print("저장된 CSV 문자열 내용:\\n", csv_string_output)

# 파일로 저장 시:
# sample_df_io.to_csv('output_data.csv', index=False, encoding='utf-8-sig') # utf-8-sig는 Excel에서 한글 깨짐 방지
"""
st.code(code_to_csv, language='python')

if st.checkbox("`df.to_csv()` 예시 보기 (문자열로 출력)", key="to_csv_page"):
    csv_output_buffer_ex = io.StringIO()
    sample_df_io.to_csv(
        csv_output_buffer_ex,
        sep=';',
        index=True, # 인덱스 포함
        header=True,
        na_rep='결측',
        encoding='utf-8',
        date_format='%Y-%m-%d %H:%M' # 날짜 형식 지정
    )
    saved_csv_string = csv_output_buffer_ex.getvalue()
    st.text_area("저장된 CSV 내용 (문자열):", saved_csv_string, height=200)
    st.download_button(
        label="생성된 CSV 다운로드",
        data=saved_csv_string,
        file_name='streamlit_generated.csv',
        mime='text/csv',
    )

st.markdown("---")

# --- 2.3 Excel 파일 읽기/저장 (`pd.read_excel()`, `df.to_excel()`) ---
st.subheader("2.3 Excel 파일 읽기/저장")
st.markdown("""
Excel 파일을 읽고 쓰는 기능을 제공합니다. (`openpyxl` 또는 `xlrd` 라이브러리 필요)
- `pd.read_excel(io, sheet_name=0, header=0, index_col=None, ...)`: Excel 파일에서 데이터를 읽습니다. `sheet_name`으로 특정 시트를 지정할 수 있습니다 (기본값은 첫 번째 시트).
- `df.to_excel(excel_writer, sheet_name='Sheet1', index=True, header=True, ...)`: DataFrame을 Excel 파일로 저장합니다.
""")
st.warning("💡 Excel 파일 입출력 기능을 사용하려면 추가 라이브러리 설치가 필요할 수 있습니다:\n`pip install openpyxl` (xlsx 읽기/쓰기용) 또는 `pip install xlrd` (구형 xls 읽기용)")

code_excel_io = """
import pandas as pd
import io # BytesIO를 사용하여 바이트 스트림을 파일처럼 다루기 위함

# 예제 DataFrame (sample_df_io 사용 가정)
# sample_df_io = pd.DataFrame(...)

# --- Excel 파일로 저장 (메모리 내 BytesIO 객체 사용) ---
excel_output_buffer = io.BytesIO()
sample_df_io.to_excel(
    excel_output_buffer,
    sheet_name='MyDataSheet', # 시트 이름 지정
    index=False,            # 인덱스 저장 안 함
    header=True,
    engine='openpyxl'       # 사용할 엔진 지정 (xlsx)
)
excel_output_buffer.seek(0) # 버퍼의 처음으로 포인터 이동 (읽기 위해)

# --- 저장된 Excel 데이터 읽기 (메모리 내 BytesIO 객체로부터) ---
# 실제 사용 시에는 파일 경로를 io 파라미터에 전달: pd.read_excel('filename.xlsx')
df_from_excel = pd.read_excel(
    excel_output_buffer, # 파일 객체 (또는 파일 경로)
    sheet_name='MyDataSheet', # 읽을 시트 이름
    engine='openpyxl'
)
# display_dataframe_info(df_from_excel, "Excel 파일로부터 읽어온 DataFrame")
"""
st.code(code_excel_io, language='python')

if st.checkbox("Excel 파일 저장/읽기 예시 보기 (메모리 객체 사용)", key="excel_io_page"):
    try:
        # Excel 파일로 저장 (메모리 내 BytesIO 객체 사용)
        excel_output_buffer_ex = io.BytesIO()
        sample_df_io.to_excel(
            excel_output_buffer_ex,
            sheet_name='SampleData',
            index=False,
            engine='openpyxl'
        )
        st.success("DataFrame이 메모리 내 Excel 형식으로 변환되었습니다.")
        
        # 다운로드 버튼 제공
        st.download_button(
            label="생성된 Excel 파일 다운로드",
            data=excel_output_buffer_ex.getvalue(), # getvalue()로 바이트 데이터 가져오기
            file_name="streamlit_generated.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

        # 저장된 Excel 데이터 읽기 (메모리 객체에서)
        excel_output_buffer_ex.seek(0) # 버퍼 포인터를 처음으로 되돌림
        df_from_excel_ex = pd.read_excel(
            excel_output_buffer_ex,
            sheet_name='SampleData', # 저장 시 사용한 시트 이름
            engine='openpyxl'
        )
        display_dataframe_info(df_from_excel_ex, "메모리에서 Excel 형식으로 읽어온 DataFrame", max_rows_to_display=5)

    except ImportError:
        st.error("Excel 기능을 사용하려면 `openpyxl` 라이브러리가 필요합니다. `pip install openpyxl` 명령어로 설치해주세요.")
    except Exception as e:
        st.error(f"Excel 처리 중 오류 발생: {e}")


st.markdown("---")
st.markdown("""
Pandas는 이 외에도 JSON (`pd.read_json`, `df.to_json`), SQL 데이터베이스 (`pd.read_sql`, `df.to_sql`), HTML (`pd.read_html`, `df.to_html`), Pickle (`pd.read_pickle`, `df.to_pickle`) 등 다양한 파일 형식과의 입출력을 지원합니다.
각 함수의 문서를 참고하여 필요한 파라미터를 적절히 사용하세요.
""")