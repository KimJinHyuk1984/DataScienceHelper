# pages/2_📄_파일에서_데이터_읽기.py
import streamlit as st
import pandas as pd
import json # JSON 파일 처리를 위해
import io   # 문자열을 파일처럼 다루기 위해
# from utils_pandas import display_dataframe_info # 이전 Pandas 도우미의 유틸리티 함수, 필요시 사용 가능

st.header("2. 파일에서 데이터 읽기")
st.markdown("""
가장 일반적인 데이터 수집 방법 중 하나는 이미 존재하는 파일에서 데이터를 읽어오는 것입니다.
Pandas는 CSV, Excel, JSON 등 다양한 파일 형식을 손쉽게 읽고 DataFrame으로 변환하는 강력한 기능을 제공합니다.
""")

# --- 예제 데이터 문자열 (파일 대신 사용) ---
csv_sample_data = """ID,Name,Age,City
1,Alice,25,New York
2,Bob,30,Paris
3,Charlie,22,London
4,David,35,Tokyo"""

excel_sample_data_sheet1 = {
    'Product_ID': [101, 102, 103],
    'Product_Name': ['Laptop', 'Mouse', 'Keyboard'],
    'Price': [1200, 25, 75]
}
excel_sample_data_sheet2 = {
    'Order_ID': ['ORD001', 'ORD002'],
    'Product_ID': [101, 103],
    'Quantity': [1, 2]
}


json_sample_data_list_of_dicts = """
[
    {"id": 1, "name": "Apple", "color": "Red", "price": 1.2},
    {"id": 2, "name": "Banana", "color": "Yellow", "price": 0.5},
    {"id": 3, "name": "Orange", "color": "Orange", "price": 0.8}
]
"""

json_sample_data_records_oriented = """
{"A01": {"name": "Eve", "age": 28, "city": "Seoul"},
 "A02": {"name": "Frank", "age": 45, "city": "Berlin"}}
"""


st.markdown("---")

# --- 2.1 CSV 파일 읽기 (`pd.read_csv()`) ---
st.subheader("2.1 CSV 파일 읽기 (`pd.read_csv()`)")
st.markdown("""
쉼표로 구분된 값(Comma-Separated Values) 형식의 파일을 읽어 DataFrame으로 변환합니다.
주요 파라미터는 이전 Pandas 도우미의 '데이터 입출력' 페이지를 참고하세요.
""")
code_read_csv_coll = """
import pandas as pd
import io # 문자열을 파일처럼 다루기 위함

csv_data_string = \"\"\"ID,Name,Age,City
1,Alice,25,New York
2,Bob,30,Paris
3,Charlie,22,London
4,David,35,Tokyo\"\"\"

# 문자열 데이터를 파일처럼 읽기 위해 StringIO 사용
csv_file_like = io.StringIO(csv_data_string)

df_from_csv = pd.read_csv(csv_file_like)
# 수집한 데이터에 한글이 포함되어 있는 경우 인코딩을 지정해야 할 수도 있습니다. 
# 인코딩은 cp949, utf-8, euc-kr 등으로 다음과 같이 지정할 수 있습니다.
# df_from_csv = pd.read_csv(csv_file_like, encoding = 'euc-kr')

# print("CSV에서 읽은 DataFrame:\\n", df_from_csv)
"""
st.code(code_read_csv_coll, language='python')

if st.checkbox("CSV 파일 읽기 예시 실행", key="read_csv_coll_page"):
    st.markdown("#### 예제 CSV 데이터 내용:")
    st.text(csv_sample_data)
    csv_file_like_ex = io.StringIO(csv_sample_data)
    df_csv_ex = pd.read_csv(csv_file_like_ex)
    st.markdown("#### `pd.read_csv()` 실행 결과 DataFrame:")
    st.dataframe(df_csv_ex)
    st.write("DataFrame 정보:")
    buffer = io.StringIO()
    df_csv_ex.info(buf=buffer)
    st.text(buffer.getvalue())


st.markdown("---")

# --- 2.2 Excel 파일 읽기 (`pd.read_excel()`) ---
st.subheader("2.2 Excel 파일 읽기 (`pd.read_excel()`)")
st.markdown("""
Excel 파일(`.xls`, `.xlsx`)을 읽어 DataFrame으로 변환합니다. `openpyxl` (xlsx용) 또는 `xlrd` (구형 xls용) 라이브러리가 필요할 수 있습니다.
- `io`: 파일 경로 또는 파일 객체.
- `sheet_name`: 읽어올 시트 이름(문자열), 시트 번호(0부터 시작), 또는 여러 시트를 읽을 경우 리스트나 `None`(모든 시트). 기본값은 첫 번째 시트.
""")
st.warning("💡 Excel 파일 기능을 사용하려면 `pip install openpyxl` 또는 `pip install xlrd` 설치가 필요할 수 있습니다.")

code_read_excel_coll = """
import pandas as pd
import io # BytesIO를 사용하여 바이트 스트림을 파일처럼 다루기 위함

# 예제 DataFrame을 Excel 형식의 바이트 스트림으로 변환 (파일 대신 사용)
excel_writer_buffer = io.BytesIO()
with pd.ExcelWriter(excel_writer_buffer, engine='openpyxl') as writer:
    pd.DataFrame({'Product_ID': [101,102], 'Name': ['Laptop','Mouse']}).to_excel(writer, sheet_name='Products', index=False)
    pd.DataFrame({'Order_ID': ['O1'], 'Product_ID': [101]}).to_excel(writer, sheet_name='Orders', index=False)
excel_writer_buffer.seek(0) # 스트림의 처음으로 포인터 이동

# 첫 번째 시트 읽기
df_excel_sheet1 = pd.read_excel(excel_writer_buffer, sheet_name='Products')
# print("Excel 첫 번째 시트 ('Products'):\\n", df_excel_sheet1)

# 모든 시트 읽기 (결과는 딕셔너리 형태: {시트이름: DataFrame})
excel_writer_buffer.seek(0) # 다시 처음으로
all_sheets_dict = pd.read_excel(excel_writer_buffer, sheet_name=None)
# print("\\nExcel 모든 시트:")
# for sheet_name, df_sheet in all_sheets_dict.items():
#     print(f"--- 시트: {sheet_name} ---")
#     print(df_sheet)
"""
st.code(code_read_excel_coll, language='python')

if st.checkbox("Excel 파일 읽기 예시 실행", key="read_excel_coll_page"):
    try:
        # Streamlit 앱 내에서 실제 파일 대신 메모리 상의 BytesIO 객체 사용
        excel_buffer_ex = io.BytesIO()
        with pd.ExcelWriter(excel_buffer_ex, engine='openpyxl') as writer:
            pd.DataFrame(excel_sample_data_sheet1).to_excel(writer, sheet_name='ProductInfo', index=False)
            pd.DataFrame(excel_sample_data_sheet2).to_excel(writer, sheet_name='OrderData', index=False)
        
        excel_buffer_ex.seek(0) # 읽기 위해 포인터 초기화
        
        st.markdown("#### 'ProductInfo' 시트 읽기 결과:")
        df_excel_p_info = pd.read_excel(excel_buffer_ex, sheet_name='ProductInfo')
        st.dataframe(df_excel_p_info)

        excel_buffer_ex.seek(0) # 다시 초기화
        st.markdown("#### 모든 시트 읽기 결과 (딕셔너리):")
        all_sheets_data = pd.read_excel(excel_buffer_ex, sheet_name=None) # None은 모든 시트
        for s_name, s_df in all_sheets_data.items():
            st.write(f"**시트 이름: {s_name}**")
            st.dataframe(s_df)

    except ImportError:
        st.error("Excel 기능을 사용하려면 `openpyxl` 라이브러리가 필요합니다. `pip install openpyxl`로 설치해주세요.")
    except Exception as e:
        st.error(f"Excel 처리 중 오류: {e}")


st.markdown("---")

# --- 2.3 JSON 파일 읽기 (`pd.read_json()` 또는 `json` 모듈) ---
st.subheader("2.3 JSON 파일 읽기")
st.markdown("""
JavaScript Object Notation (JSON) 형식의 파일을 읽습니다.
- **`pd.read_json(path_or_buf, orient='columns', ...)`**: JSON 문자열이나 파일을 DataFrame으로 변환. `orient` 파라미터로 JSON 구조를 어떻게 해석할지 지정합니다 (예: `'records'`, `'columns'`, `'index'`, `'split'`, `'values'`).
- **`json` 모듈 (`json.load()`, `json.loads()`):** 표준 파이썬 라이브러리로, JSON 데이터를 파이썬 딕셔너리나 리스트로 변환합니다. 이후 Pandas DataFrame으로 직접 변환할 수 있습니다.
""")
code_read_json_coll = """
import pandas as pd
import json
import io

# 예제 1: 리스트 형태의 JSON (orient='records' 와 유사)
json_str_list_of_dicts = \"\"\"
[
    {"id": 1, "name": "Apple", "color": "Red"},
    {"id": 2, "name": "Banana", "color": "Yellow"}
]
\"\"\"
df_from_json1 = pd.read_json(io.StringIO(json_str_list_of_dicts), orient='records')
# print("JSON (list of dicts) 읽기 결과:\\n", df_from_json1)


# 예제 2: 레코드 지향 JSON (키가 인덱스, 값이 레코드 딕셔너리)
# pandas.read_json의 orient='index'와 유사
json_str_records_oriented = \"\"\"
{"A01": {"name": "Eve", "age": 28},
 "A02": {"name": "Frank", "age": 45}}
\"\"\"
df_from_json2 = pd.read_json(io.StringIO(json_str_records_oriented), orient='index')
# print("\\nJSON (records oriented, index) 읽기 결과:\\n", df_from_json2)


# json 모듈 사용 예시
# parsed_json_data = json.loads(json_str_list_of_dicts) # 문자열을 파이썬 객체로
# df_from_json_module = pd.DataFrame(parsed_json_data)
# print("\\njson 모듈 사용 후 DataFrame 변환 결과:\\n", df_from_json_module)
"""
st.code(code_read_json_coll, language='python')

if st.checkbox("JSON 파일 읽기 예시 실행", key="read_json_coll_page"):
    st.markdown("#### 예제 1: 리스트 of 딕셔너리 형태 JSON")
    st.text(json_sample_data_list_of_dicts)
    df_json1_ex = pd.read_json(io.StringIO(json_sample_data_list_of_dicts), orient='records')
    st.markdown("`pd.read_json(..., orient='records')` 결과:")
    st.dataframe(df_json1_ex)

    st.markdown("#### 예제 2: 레코드 지향 JSON (인덱스로 사용)")
    st.text(json_sample_data_records_oriented)
    df_json2_ex = pd.read_json(io.StringIO(json_sample_data_records_oriented), orient='index')
    st.markdown("`pd.read_json(..., orient='index')` 결과:")
    st.dataframe(df_json2_ex)
    
    st.markdown("#### `json` 모듈 사용 예시")
    parsed_data_ex = json.loads(json_sample_data_list_of_dicts) # Python 리스트로 변환
    st.write("`json.loads()` 결과 (Python 객체):")
    st.write(parsed_data_ex)
    df_from_parsed_ex = pd.DataFrame(parsed_data_ex)
    st.write("이후 `pd.DataFrame()`으로 변환 결과:")
    st.dataframe(df_from_parsed_ex)


st.markdown("---")
st.markdown("이 외에도 `pd.read_xml()` (XML), `pd.read_html()` (HTML 테이블), `pd.read_sql()` (SQL 데이터베이스), `pd.read_pickle()` (파이썬 피클 객체) 등 다양한 파일 및 데이터 소스를 읽는 함수들이 Pandas에 준비되어 있습니다.")
