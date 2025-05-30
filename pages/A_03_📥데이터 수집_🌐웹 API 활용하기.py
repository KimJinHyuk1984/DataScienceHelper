# pages/3_🌐_웹_API_활용하기.py
import streamlit as st
import pandas as pd # JSON 응답을 DataFrame으로 변환 시 사용
import requests    # HTTP 요청을 보내기 위한 라이브러리
import json        # JSON 데이터를 다루기 위함
from utils.utils_collection import display_api_response # 유틸리티 함수 사용

st.header("3. 웹 API (Application Programming Interface) 활용하기")
st.markdown("""
웹 API는 애플리케이션(소프트웨어) 간의 상호작용을 위한 인터페이스입니다. 특정 웹 서비스가 제공하는 데이터를 정해진 규칙(프로토콜, 주로 HTTP)에 따라 요청하고 응답받을 수 있게 해줍니다.
이를 통해 다른 서비스의 데이터를 프로그래밍 방식으로 가져와 활용할 수 있습니다.

**주요 개념:**
-   **Endpoint:** API에 접근할 수 있는 특정 URL 주소입니다. 각 엔드포인트는 특정 리소스나 기능을 나타냅니다.
-   **HTTP Methods:** API 요청 시 사용되는 주요 HTTP 메소드입니다.
    -   `GET`: 서버로부터 정보를 요청합니다 (데이터 조회).
    -   `POST`: 서버에 새로운 데이터를 생성합니다.
    -   `PUT`: 서버의 기존 데이터를 수정합니다.
    -   `DELETE`: 서버의 데이터를 삭제합니다.
    (이 페이지에서는 주로 `GET` 요청을 다룹니다.)
-   **Request (요청):** 클라이언트가 서버로 보내는 메시지 (엔드포인트, 메소드, 헤더, 파라미터, 바디 등 포함).
-   **Response (응답):** 서버가 클라이언트의 요청에 대해 보내는 메시지 (상태 코드, 헤더, 바디(데이터) 등 포함).
-   **JSON (JavaScript Object Notation):** API 응답 데이터 형식으로 널리 사용되는 가벼운 텍스트 기반 형식입니다.
-   **API Key:** 많은 API 서비스는 무분별한 사용을 막고 사용자를 식별하기 위해 API 키를 발급합니다. 요청 시 이 키를 포함해야 하는 경우가 많습니다.
""")

st.info("💡 이 페이지의 예제에서는 인증이 필요 없는 공개 API를 사용합니다. 실제 API 사용 시에는 해당 API의 문서를 참조하여 인증 방법, 요청 제한 등을 확인해야 합니다.")
st.markdown("---")

# --- 3.1 `requests` 라이브러리 사용 ---
st.subheader("3.1 `requests` 라이브러리 사용하기")
st.markdown("""
파이썬에서 HTTP 요청을 보내고 응답을 받기 위해 가장 널리 사용되는 라이브러리는 `requests`입니다.
- `pip install requests` 명령으로 설치할 수 있습니다.
- `requests.get(url, params=None, headers=None, ...)`: GET 요청을 보냅니다.
- `response.status_code`: HTTP 상태 코드를 반환합니다 (200은 성공).
- `response.text`: 응답 내용을 문자열로 반환합니다.
- `response.json()`: 응답 내용이 JSON 형식일 경우, 파이썬 딕셔너리나 리스트로 변환하여 반환합니다.
""")

code_requests_get = """
import requests
import json # JSON 응답을 예쁘게 출력하기 위해

# 예제 API 엔드포인트 (JSONPlaceholder - 가짜 온라인 REST API)
# /posts/1은 첫 번째 게시글 정보를 요청
url = "https://jsonplaceholder.typicode.com/posts/1"

try:
    # GET 요청 보내기
    response = requests.get(url)

    # 요청 성공 여부 확인 (상태 코드가 200번대이면 성공)
    response.raise_for_status() # 200번대가 아니면 HTTPError 예외 발생

    # 응답 내용 확인
    # print(f"상태 코드: {response.status_code}")
    
    # JSON 응답 파싱
    # data = response.json() # 응답이 JSON 형식이라고 가정
    # print("JSON 데이터:")
    # print(json.dumps(data, indent=2, ensure_ascii=False)) # 예쁘게 출력 (ensure_ascii=False로 한글 유지)

except requests.exceptions.HTTPError as http_err:
    # print(f"HTTP 에러 발생: {http_err}")
    # print(f"응답 내용: {response.text if response else '응답 없음'}")
except requests.exceptions.RequestException as req_err:
    # print(f"요청 에러 발생: {req_err}")
except Exception as e:
    # print(f"기타 에러 발생: {e}")
    pass # Streamlit에서는 st.error 등으로 표시
"""
st.code(code_requests_get, language='python')

if st.checkbox("`requests.get()` 기본 예시 실행 (JSONPlaceholder API)", key="requests_get_page_3"):
    api_url_placeholder = "https://jsonplaceholder.typicode.com/todos/1" # 할 일 목록 첫 번째 아이템
    st.write(f"요청할 API URL: `{api_url_placeholder}`")
    
    try:
        response_placeholder = requests.get(api_url_placeholder, timeout=5) # 5초 타임아웃 설정
        # 유틸리티 함수를 사용하여 API 응답 표시
        display_api_response(response_placeholder, title="JSONPlaceholder API 응답 결과")
    except requests.exceptions.Timeout:
        st.error("요청 시간 초과: 서버에서 응답이 없거나 네트워크 연결을 확인해주세요.")
    except requests.exceptions.RequestException as e:
        st.error(f"API 요청 중 에러 발생: {e}")


st.markdown("---")

# --- 3.2 API 요청 파라미터 및 헤더 ---
st.subheader("3.2 API 요청 파라미터 및 헤더")
st.markdown("""
API 요청 시 추가 정보를 전달하기 위해 URL 파라미터(Query Parameters)나 HTTP 헤더(Headers)를 사용할 수 있습니다.
- **URL 파라미터:** `GET` 요청 시 URL 뒤에 `?key1=value1&key2=value2` 형태로 추가되어 서버에 특정 조건을 전달합니다. `requests` 라이브러리에서는 `params` 인자로 딕셔너리를 전달하여 쉽게 구성할 수 있습니다.
- **헤더:** 요청에 대한 메타데이터(예: 인증 토큰, Content-Type)를 전달합니다. `headers` 인자로 딕셔너리를 전달합니다.
""")

code_params_headers = """
import requests
import json

# 예제: 특정 사용자의 게시글 목록 가져오기 (JSONPlaceholder API)
base_url = "https://jsonplaceholder.typicode.com/posts"

# URL 파라미터 설정 (userId가 1인 게시글만 필터링)
query_params = {'userId': 1, '_limit': 3} # _limit은 결과 개수 제한 (JSONPlaceholder 비표준 파라미터)

# 헤더 설정 (예시: API 키가 필요한 경우)
# 실제 API 키는 여기에 직접 작성하지 마세요! (st.secrets 등 사용)
# headers = {
#     'Authorization': 'Bearer YOUR_API_KEY', # 인증 토큰 전달 예시
#     'Content-Type': 'application/json',    # 요청/응답 형식 지정 예시
#     'User-Agent': 'MyStreamlitApp/1.0'     # 사용자 에이전트 지정 예시
# }
# 이 예제 API는 인증 헤더가 필요 없습니다.

try:
    # GET 요청 시 params 인자로 쿼리 파라미터 전달
    response = requests.get(base_url, params=query_params) # headers=headers 추가 가능
    response.raise_for_status()
    
    # data = response.json()
    # print(f"userId=1인 게시글 (상위 {query_params.get('_limit', '모든')}개):")
    # print(json.dumps(data, indent=2, ensure_ascii=False))
    # print(f"실제 요청된 URL: {response.url}") # requests가 파라미터를 URL에 자동으로 추가해줌

except requests.exceptions.RequestException as e:
    # print(f"API 요청 에러: {e}")
    pass
"""
st.code(code_params_headers, language='python')

if st.checkbox("API 요청 파라미터 예시 실행", key="params_headers_page_3"):
    api_url_params = "https://jsonplaceholder.typicode.com/comments"
    # postId가 1인 댓글만 가져오도록 파라미터 설정
    params_ex = {'postId': 1, '_limit': 2} # _limit으로 결과 수 제한 (JSONPlaceholder 기능)
    
    st.write(f"요청할 API URL: `{api_url_params}`")
    st.write(f"URL 파라미터: `{params_ex}`")
    
    try:
        response_params_ex = requests.get(api_url_params, params=params_ex, timeout=5)
        display_api_response(response_params_ex, title=f"postId=1인 댓글 API 응답 결과 (상위 {params_ex.get('_limit','N')}개)")
        st.caption(f"실제 요청된 URL (파라미터 포함): `{response_params_ex.url}`")
    except requests.exceptions.Timeout:
        st.error("요청 시간 초과: 서버에서 응답이 없거나 네트워크 연결을 확인해주세요.")
    except requests.exceptions.RequestException as e:
        st.error(f"API 요청 중 에러 발생: {e}")


st.markdown("---")

# --- 3.3 API Key 및 인증 (중요: 보안) ---
st.subheader("3.3 API Key 및 인증")
st.markdown("""
많은 API는 인증을 요구하며, 주로 **API Key**를 사용합니다. API Key는 서비스 제공자가 발급하는 고유한 문자열로, 사용자를 식별하고 API 사용량을 추적하는 데 사용됩니다.

**API Key 사용 방법 (일반적인 경우):**
1.  **HTTP 헤더에 포함:** 가장 일반적인 방법으로, `Authorization` 헤더나 API 제공자가 지정한 특정 헤더(예: `X-Api-Key`)에 키 값을 넣어 전송합니다.
    ```python
    headers = {'Authorization': 'Bearer YOUR_API_KEY'}
    # 또는
    # headers = {'X-Api-Key': 'YOUR_API_KEY'}
    # response = requests.get(url, headers=headers)
    ```
2.  **URL 파라미터로 전달:** 일부 API는 URL에 API 키를 포함하도록 요구합니다 (예: `?apikey=YOUR_API_KEY`). 덜 안전한 방식으로 간주됩니다.

**🚨 중요: API Key 보안 관리**
-   **절대 코드에 API Key를 직접 하드코딩하지 마세요!** GitHub 등 공개된 장소에 코드를 올릴 경우 API Key가 노출될 수 있습니다.
-   **Streamlit 앱에서의 안전한 API Key 관리:** Streamlit의 **Secrets Management (`st.secrets`)** 기능을 사용하세요.
    1.  프로젝트 폴더에 `.streamlit/secrets.toml` 파일을 만듭니다.
    2.  `secrets.toml` 파일 안에 다음과 같이 API Key를 저장합니다:
        ```toml
        # .streamlit/secrets.toml
        MY_API_KEY = "여기에_실제_API_키_입력"
        ANOTHER_SERVICE_KEY = "다른_서비스_API_키"
        ```
    3.  Python 코드에서 `st.secrets`를 통해 접근합니다:
        ```python
        # api_key = st.secrets["MY_API_KEY"]
        # headers = {'Authorization': f'Bearer {api_key}'}
        ```
    4.  `.streamlit/secrets.toml` 파일은 **`.gitignore`에 추가**하여 Git 저장소에 포함되지 않도록 합니다.
    5.  Streamlit Community Cloud에 배포 시에는 앱 설정에서 Secrets를 직접 입력합니다.
-   **환경 변수 사용:** 로컬 개발 환경에서는 환경 변수를 통해 API Key를 관리할 수도 있습니다.
""")
code_secrets_example = """
import streamlit as st
import requests

# # --- Streamlit Secrets 사용 예시 (실행하려면 .streamlit/secrets.toml 설정 필요) ---
# # 가정: .streamlit/secrets.toml 파일에 SOME_API_KEY = "your_actual_key" 가 있다고 가정.
# # 이 예제는 실제 API 호출 없이 st.secrets 사용법만 보여줍니다.

# try:
#     # st.secrets 딕셔너리에서 API 키 가져오기
#     my_secret_api_key = st.secrets["SOME_API_KEY"]
#     st.success("secrets.toml에서 API 키를 성공적으로 로드했습니다! (실제 키는 출력하지 않음)")
#     # st.write(f"로드된 키 (일부만 표시): {my_secret_api_key[:4]}...{my_secret_api_key[-4:]}") # 실제 키는 절대 이렇게도 출력하지 마세요!

#     # # 실제 API 요청 시 사용 예
#     # api_url = "https://api.example.com/data"
#     # headers = {"Authorization": f"Bearer {my_secret_api_key}"}
#     # response = requests.get(api_url, headers=headers)
#     # # ... 응답 처리 ...

# except KeyError:
#     st.warning("'.streamlit/secrets.toml' 파일에 'SOME_API_KEY'가 정의되어 있지 않습니다. 예제를 실행하려면 설정이 필요합니다.")
# except Exception as e:
#     st.error(f"Secrets 처리 중 오류: {e}")
"""
st.code(code_secrets_example, language='python')
st.markdown("실제 API 키를 사용하는 예제는 로컬 환경에서 `secrets.toml` 파일을 직접 설정하고 테스트해보세요.")


st.markdown("---")
st.markdown("""
웹 API를 활용하면 방대한 양의 정형화된 데이터를 쉽게 수집할 수 있습니다.
항상 API 제공자의 문서를 꼼꼼히 읽고, 사용 정책(요청 제한, 인증 방법 등)을 준수하며 책임감 있게 사용해야 합니다.
""")