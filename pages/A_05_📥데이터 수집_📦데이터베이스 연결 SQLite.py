# pages/5_📦_데이터베이스_연결_SQLite.py
import streamlit as st
import pandas as pd
import sqlite3 # SQLite 데이터베이스를 사용하기 위한 파이썬 내장 모듈
import os      # 파일 존재 여부 확인 및 삭제용 (예제용 DB 파일 관리)
from utils.utils_pandas import display_dataframe_info # Pandas DataFrame 정보 표시용 (필요시 사용)

st.header("5. 데이터베이스 연결 및 사용 (SQLite 기초)")
st.markdown("""
데이터베이스는 구조화된 데이터를 효율적으로 저장, 관리, 검색할 수 있는 시스템입니다.
대량의 데이터를 영구적으로 보관하거나 복잡한 쿼리를 통해 원하는 정보를 추출할 때 유용합니다.

이 페이지에서는 파이썬에 기본적으로 내장되어 있어 별도의 설치 없이 사용할 수 있는 **SQLite** 데이터베이스를 중심으로 기본적인 사용법을 알아봅니다.
SQLite는 서버 설정 없이 파일 기반으로 동작하는 가벼운 관계형 데이터베이스 관리 시스템(RDBMS)입니다.
""")

DB_FILENAME = "example_sqlite.db" # 예제용 데이터베이스 파일 이름

st.info(f"""
💡 **SQLite 정보:**
-   별도의 서버 프로세스가 필요 없는 파일 기반 데이터베이스입니다.
-   데이터베이스 전체가 단일 파일(`{DB_FILENAME}`과 같은 형태)에 저장됩니다.
-   파이썬 표준 라이브러리 `sqlite3` 모듈을 통해 쉽게 사용할 수 있습니다.
-   간단한 애플리케이션, 로컬 데이터 저장, 프로토타이핑 등에 적합합니다.
""")
st.markdown("---")

# --- 5.1 SQLite 데이터베이스 연결 및 기본 작업 ---
st.subheader("5.1 SQLite 데이터베이스 연결 및 기본 작업 (`sqlite3` 모듈)")
st.markdown("""
`sqlite3` 모듈을 사용하여 데이터베이스에 연결하고 SQL 쿼리를 실행할 수 있습니다.

**주요 단계:**
1.  **연결 (`sqlite3.connect()`):** 데이터베이스 파일에 연결합니다. 파일이 없으면 새로 생성됩니다. 인메모리 DB의 경우 `':memory:'` 사용.
2.  **커서 생성 (`conn.cursor()`):** SQL 명령을 실행하고 결과를 가져오기 위한 커서 객체를 생성합니다.
3.  **SQL 실행 (`cursor.execute()`, `cursor.executemany()`):** SQL 쿼리 문자열을 실행합니다.
    -   테이블 생성: `CREATE TABLE`
    -   데이터 삽입: `INSERT INTO`
    -   데이터 조회: `SELECT`
    -   데이터 수정: `UPDATE`
    -   데이터 삭제: `DELETE`
4.  **결과 가져오기 (SELECT의 경우):**
    -   `cursor.fetchone()`: 결과 중 한 행을 가져옵니다.
    -   `cursor.fetchall()`: 결과 중 모든 행을 리스트로 가져옵니다.
    -   `cursor.fetchmany(size)`: 지정된 개수만큼의 행을 가져옵니다.
5.  **변경사항 확정 (`conn.commit()`):** `INSERT`, `UPDATE`, `DELETE` 등 데이터베이스 내용을 변경하는 작업 후에는 반드시 `commit()`을 호출해야 변경사항이 영구적으로 저장됩니다.
6.  **연결 종료 (`conn.close()`):** 데이터베이스 사용이 끝나면 연결을 닫습니다. `with` 문을 사용하면 자동으로 처리됩니다.
""")

code_sqlite_basic = """
import sqlite3
import os

DB_FILE = 'my_sqlite_app.db' # 데이터베이스 파일명

# 1. 데이터베이스 연결 (파일이 없으면 자동 생성)
conn = sqlite3.connect(DB_FILE)
# print(f"'{DB_FILE}' 데이터베이스에 연결되었습니다.")

# 2. 커서 생성
cursor = conn.cursor()

# 3. 테이블 생성 (이미 존재하면 에러 방지: IF NOT EXISTS)
cursor.execute('''
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    email TEXT UNIQUE NOT NULL,
    age INTEGER
)
''')
# print(" 'users' 테이블 생성 또는 이미 존재함.")

# 4. 데이터 삽입 (SQL 인젝션 방지를 위해 ? 플레이스홀더 사용)
try:
    cursor.execute("INSERT INTO users (name, email, age) VALUES (?, ?, ?)", 
                   ('Alice Wonderland', 'alice@example.com', 30))
    cursor.execute("INSERT INTO users (name, email, age) VALUES (?, ?, ?)",
                   ('Bob The Builder', 'bob@example.com', 45))
    # print("데이터 삽입 시도.")
except sqlite3.IntegrityError as e:
    # print(f"데이터 삽입 중 오류 (아마도 email UNIQUE 제약조건 위반): {e}")
    pass # 이미 데이터가 있다면 넘어감

# 여러 데이터 한 번에 삽입 (executemany)
# new_users = [
#     ('Charlie Chaplin', 'charlie@example.com', 25),
#     ('David Copperfield', 'david@example.com', 50)
# ]
# try:
#     cursor.executemany("INSERT INTO users (name, email, age) VALUES (?, ?, ?)", new_users)
# except sqlite3.IntegrityError: pass


# 5. 변경사항 확정 (데이터 삽입/수정/삭제 후 필수)
conn.commit()
# print("데이터 변경사항이 확정되었습니다 (commit).")

# 6. 데이터 조회 (SELECT)
cursor.execute("SELECT id, name, email, age FROM users WHERE age > ?", (30,)) # 30세 초과 사용자
# print("\\n30세 초과 사용자 조회 결과:")
# for row in cursor.fetchall(): # 모든 결과 가져오기
#     print(row) # 각 행은 튜플 형태로 반환됨 (id, name, email, age)

# 7. 연결 종료
conn.close()
# print(f"'{DB_FILE}' 데이터베이스 연결이 종료되었습니다.")

# 예제 실행 후 생성된 DB 파일 삭제 (선택 사항)
# if os.path.exists(DB_FILE):
#     os.remove(DB_FILE)
#     print(f"'{DB_FILE}' 파일이 삭제되었습니다.")
"""
st.code(code_sqlite_basic, language='python')

if st.checkbox(f"`sqlite3` 기본 작업 예시 실행 (로컬에 `{DB_FILENAME}` 파일 생성/사용)", key="sqlite_basic_page_5"):
    conn_ex = None # 연결 객체 초기화
    try:
        # 1. 연결 (파일 DB 사용)
        conn_ex = sqlite3.connect(DB_FILENAME)
        cursor_ex = conn_ex.cursor()
        st.success(f"`{DB_FILENAME}` 데이터베이스에 연결되었습니다.")

        # 2. 테이블 생성
        cursor_ex.execute("""
        CREATE TABLE IF NOT EXISTS employees (
            id INTEGER PRIMARY KEY,
            name TEXT,
            department TEXT,
            salary INTEGER
        )""")
        st.write("'employees' 테이블이 준비되었습니다.")

        # 3. 데이터 삽입 (간단한 예시, 중복 실행 시 에러 방지 로직은 생략)
        # 실제 앱에서는 이미 데이터가 있는지 확인하거나, UNIQUE 제약조건 활용
        employees_data = [
            (1, 'Alice Smith', 'HR', 60000),
            (2, 'Bob Johnson', 'IT', 80000),
            (3, 'Charlie Brown', 'Sales', 75000),
            (4, 'Diana Prince', 'IT', 90000)
        ]
        # 기존 데이터가 없을 경우에만 삽입 (간단히 ID로 확인)
        cursor_ex.execute("SELECT COUNT(*) FROM employees WHERE id IN (1,2,3,4)")
        if cursor_ex.fetchone()[0] == 0:
            cursor_ex.executemany("INSERT INTO employees VALUES (?,?,?,?)", employees_data)
            conn_ex.commit()
            st.write("샘플 데이터가 'employees' 테이블에 삽입되었습니다.")
        else:
            st.write("샘플 데이터가 이미 'employees' 테이블에 존재합니다.")


        # 4. 데이터 조회
        st.write("--- 'IT' 부서 직원 조회 결과 ---")
        cursor_ex.execute("SELECT name, salary FROM employees WHERE department = ?", ('IT',))
        it_employees = cursor_ex.fetchall()
        if it_employees:
            df_it = pd.DataFrame(it_employees, columns=['Name', 'Salary'])
            st.dataframe(df_it)
        else:
            st.write("'IT' 부서 직원이 없습니다.")
        
        st.caption(f"예제가 끝나면 생성된 `{DB_FILENAME}` 파일은 수동으로 삭제하시거나, 아래 버튼으로 삭제할 수 있습니다.")
        if st.button(f"`{DB_FILENAME}` 파일 삭제하기", key="delete_db_file_basic"):
            if conn_ex: conn_ex.close() # 연결 먼저 닫기
            if os.path.exists(DB_FILENAME):
                os.remove(DB_FILENAME)
                st.success(f"`{DB_FILENAME}` 파일이 삭제되었습니다.")
            else:
                st.warning(f"`{DB_FILENAME}` 파일이 이미 존재하지 않습니다.")


    except sqlite3.Error as e:
        st.error(f"SQLite 작업 중 오류 발생: {e}")
    finally:
        if conn_ex: # 연결 객체가 생성되었다면
            conn_ex.close()
            # st.write(f"`{DB_FILENAME}` 데이터베이스 연결이 안전하게 종료되었습니다.")


st.markdown("---")

# --- 5.2 Pandas와 SQLite 연동 ---
st.subheader("5.2 Pandas와 SQLite 연동")
st.markdown("""
Pandas는 SQLite 데이터베이스와 손쉽게 연동할 수 있는 기능을 제공하여, DataFrame과 데이터베이스 테이블 간의 데이터 이동을 편리하게 합니다.

- **`pd.read_sql_query(sql_query, connection_object)`**: SQL `SELECT` 쿼리를 실행하고 그 결과를 Pandas DataFrame으로 직접 불러옵니다.
- **`pd.read_sql_table(table_name, connection_object, index_col=None, ...)`**: 데이터베이스 테이블 전체를 DataFrame으로 불러옵니다 (SQLAlchemy 엔진 필요할 수 있음, SQLite는 보통 괜찮음).
- **`DataFrame.to_sql(table_name, connection_object, if_exists='fail', index=True, ...)`**: DataFrame의 내용을 데이터베이스 테이블로 저장합니다.
    - `if_exists`: 테이블이 이미 존재할 경우 동작 방식.
        - `'fail'`: 에러 발생 (기본값).
        - `'replace'`: 기존 테이블 삭제 후 새로 생성하여 저장.
        - `'append'`: 기존 테이블에 데이터 추가.
    - `index`: DataFrame의 인덱스를 테이블의 컬럼으로 저장할지 여부 (기본값 `True`).
""")

code_pandas_sqlite = """
import pandas as pd
import sqlite3
import os

DB_FILE_PANDAS = 'pandas_sqlite.db'

# 예제 DataFrame
data_for_db = {
    'Ticker': ['AAPL', 'MSFT', 'GOOG'],
    'Price': [170.34, 305.22, 2520.50],
    'Sector': ['Technology', 'Technology', 'Technology']
}
df_stocks = pd.DataFrame(data_for_db)

# SQLite 연결 (with 문 사용 시 자동 commit/close)
with sqlite3.connect(DB_FILE_PANDAS) as conn_pd:
    # DataFrame을 'stocks' 테이블로 저장 (이미 있으면 대체)
    df_stocks.to_sql('stocks', conn_pd, if_exists='replace', index=False)
    # print("'stocks' 테이블에 DataFrame 저장 완료.")

    # 'stocks' 테이블에서 모든 데이터 조회하여 DataFrame으로 불러오기
    df_from_db = pd.read_sql_query("SELECT Ticker, Price FROM stocks WHERE Sector = 'Technology'", conn_pd)
    # print("\\n'Technology' 섹터 주식 정보 조회 결과 (DataFrame):")
    # print(df_from_db)

    # (선택 사항) 'stocks' 테이블에서 Price가 200 이상인 데이터만 조회
    # high_price_stocks = pd.read_sql_query("SELECT * FROM stocks WHERE Price >= ?", conn_pd, params=(200,))
    # print("\\nPrice >= 200 인 주식 정보:\\n", high_price_stocks)


# 예제 실행 후 생성된 DB 파일 삭제
# if os.path.exists(DB_FILE_PANDAS):
#     os.remove(DB_FILE_PANDAS)
#     print(f"'{DB_FILE_PANDAS}' 파일이 삭제되었습니다.")
"""
st.code(code_pandas_sqlite, language='python')

if st.checkbox(f"Pandas와 SQLite 연동 예시 실행 (로컬에 `{DB_FILENAME}_pd` 파일 생성/사용)", key="pandas_sqlite_page_5"):
    DB_FILENAME_PD_EX = f"{DB_FILENAME}_pd" # 예제용 파일 이름 구분
    conn_pd_ex = None # 연결 객체 초기화

    data_ex_pd = {
        'item_id': [1,2,3,4,5],
        'item_name': ['Keyboard', 'Mouse', 'Monitor', 'Webcam', 'Desk'],
        'category': ['Electronics', 'Electronics', 'Electronics', 'Electronics', 'Furniture'],
        'price_usd': [75, 25, 300, 50, 150]
    }
    df_items_ex = pd.DataFrame(data_ex_pd)
    
    st.write("데이터베이스에 저장할 예제 DataFrame (`df_items_ex`):")
    st.dataframe(df_items_ex)

    try:
        # SQLite 연결 (파일 DB 사용)
        conn_pd_ex = sqlite3.connect(DB_FILENAME_PD_EX)
        st.success(f"`{DB_FILENAME_PD_EX}` 데이터베이스에 연결되었습니다.")

        # DataFrame을 'items' 테이블로 저장
        df_items_ex.to_sql('items', conn_pd_ex, if_exists='replace', index=False)
        st.write("'items' 테이블에 DataFrame이 성공적으로 저장되었습니다.")

        # SQL 쿼리를 사용하여 데이터 조회 및 DataFrame으로 로드
        st.write("--- 'Electronics' 카테고리 아이템 조회 결과 ---")
        query = "SELECT item_name, price_usd FROM items WHERE category = ?"
        params = ('Electronics',)
        df_electronics = pd.read_sql_query(query, conn_pd_ex, params=params)
        st.dataframe(df_electronics)

        st.write("--- 가격이 100 USD 이상인 아이템 조회 결과 ---")
        df_expensive = pd.read_sql_query("SELECT * FROM items WHERE price_usd >= 100", conn_pd_ex)
        st.dataframe(df_expensive)

        st.caption(f"예제가 끝나면 생성된 `{DB_FILENAME_PD_EX}` 파일은 수동으로 삭제하시거나, 아래 버튼으로 삭제할 수 있습니다.")
        if st.button(f"`{DB_FILENAME_PD_EX}` 파일 삭제하기", key="delete_db_file_pandas"):
            if conn_pd_ex: conn_pd_ex.close()
            if os.path.exists(DB_FILENAME_PD_EX):
                os.remove(DB_FILENAME_PD_EX)
                st.success(f"`{DB_FILENAME_PD_EX}` 파일이 삭제되었습니다.")
            else:
                st.warning(f"`{DB_FILENAME_PD_EX}` 파일이 이미 존재하지 않습니다.")

    except sqlite3.Error as e:
        st.error(f"SQLite (Pandas 연동) 작업 중 오류 발생: {e}")
    except Exception as ex_gen: # 다른 일반적인 예외 처리
        st.error(f"처리 중 예외 발생: {ex_gen}")
    finally:
        if conn_pd_ex:
            conn_pd_ex.close()
            # st.write(f"`{DB_FILENAME_PD_EX}` 데이터베이스 연결이 안전하게 종료되었습니다.")


st.markdown("---")
st.markdown("""
SQLite는 간단한 데이터 저장 및 관리에 매우 유용하며, Pandas와의 뛰어난 호환성 덕분에 데이터 분석 파이프라인에 쉽게 통합될 수 있습니다.
더 크고 복잡한 시스템에서는 PostgreSQL, MySQL, SQL Server, Oracle 등과 같은 본격적인 서버 기반 RDBMS나, 필요에 따라 NoSQL 데이터베이스를 사용하게 됩니다.
이러한 데이터베이스들은 각각에 맞는 파이썬 라이브러리(예: `psycopg2` for PostgreSQL, `mysql.connector` for MySQL)와 SQLAlchemy 같은 ORM(Object-Relational Mapper)을 통해 파이썬과 연동할 수 있습니다.
""")