# pages/4_🕸️_웹_스크레이핑_기초.py
import streamlit as st
import pandas as pd
import requests # 웹 페이지 내용을 가져오기 위함
from bs4 import BeautifulSoup # HTML 내용을 파싱(분석)하기 위함
from utils.utils_collection import display_scraped_elements # 유틸리티 함수 사용 (선택 사항)

st.header("4. 웹 스크레이핑 (Web Scraping) 기초")
st.markdown("""
웹 스크레이핑은 웹사이트에서 HTML 형태로 제공되는 데이터를 프로그래밍 방식으로 추출하는 기술입니다.
API가 제공되지 않거나, API를 통해 얻을 수 없는 특정 정보가 필요할 때 사용될 수 있습니다.

**이 페이지에서 다루는 내용:**
- 웹 스크레이핑 시 반드시 지켜야 할 윤리적 및 법적 고려사항
- `requests` 라이브러리를 사용한 웹 페이지 HTML 내용 가져오기
- `BeautifulSoup4` 라이브러리를 사용한 HTML 내용 파싱 및 데이터 추출 기초
""")

st.error("""
**⚠️ 경고: 웹 스크레이핑의 윤리적 및 법적 책임 ⚠️**

웹 스크레이핑은 강력한 기술이지만, 잘못 사용될 경우 법적 문제나 윤리적 문제를 야기할 수 있습니다.
**아래 사항을 반드시 숙지하고 책임감 있게 사용해야 합니다:**

1.  **`robots.txt` 확인 및 준수:** 대부분의 웹사이트는 루트 디렉토리(예: `https://example.com/robots.txt`)에 `robots.txt` 파일을 두어 웹 크롤러의 접근 규칙을 명시합니다. 이 파일에서 `User-agent` 별로 `Allow` 또는 `Disallow` 된 경로를 확인하고, **`Disallow`된 경로는 절대 스크레이핑하지 마세요.**
2.  **서비스 이용 약관 (Terms of Service, ToS) 확인:** 웹사이트의 이용 약관에는 데이터 수집에 관한 정책이 명시되어 있을 수 있습니다. 약관을 위반하는 스크레이핑은 법적 조치를 받을 수 있습니다.
3.  **서버 부하 최소화:** 짧은 시간 내에 너무 많은 요청을 보내면 대상 웹사이트 서버에 과도한 부하를 줄 수 있습니다. 이는 서비스 거부(DoS) 공격으로 간주될 수 있으며, IP가 차단될 수 있습니다. 요청 사이에 적절한 시간 지연(`time.sleep()`)을 두고, 필요한 최소한의 페이지만 요청하세요.
4.  **개인정보 및 저작권 존중:** 개인을 식별할 수 있는 정보나 저작권이 있는 콘텐츠를 무단으로 수집하거나 활용해서는 안 됩니다.
5.  **데이터 활용 목적 명확화:** 수집한 데이터는 합법적이고 윤리적인 목적으로만 사용해야 합니다.

**이 페이지의 예제는 학습 목적으로만 제공되며, 실제 웹사이트에 적용하기 전에 반드시 해당 사이트의 정책을 확인하고 허용된 범위 내에서만 사용해야 합니다.**
""")
st.markdown("---")

# --- 4.1 웹 페이지 내용 가져오기 (`requests`) ---
st.subheader("4.1 웹 페이지 내용 가져오기 (`requests`)")
st.markdown("""
웹 스크레이핑의 첫 단계는 대상 웹 페이지의 HTML 내용을 가져오는 것입니다. 파이썬의 `requests` 라이브러리를 사용하여 HTTP GET 요청을 보낼 수 있습니다.
""")
code_fetch_html = """
import requests

# 스크레이핑할 대상 URL (예시)
# 실제로는 robots.txt와 ToS를 확인해야 합니다.
url = "https://books.toscrape.com/" # 스크레이핑 연습용으로 만들어진 사이트

try:
    # HTTP GET 요청 보내기
    response = requests.get(url, timeout=5) # 5초 타임아웃
    response.raise_for_status() # HTTP 에러 발생 시 예외를 일으킴 (4xx, 5xx 상태 코드)

    # HTML 내용 가져오기
    html_content = response.text
    # print(f"'{url}' 에서 HTML 내용 가져오기 성공 (일부만 표시):")
    # print(html_content[:500] + "...") # 너무 길어서 일부만 출력

except requests.exceptions.HTTPError as http_err:
    # print(f"HTTP 에러 발생: {http_err} (상태 코드: {response.status_code if 'response' in locals() else 'N/A'})")
    pass # Streamlit에서는 st.error 사용
except requests.exceptions.RequestException as req_err:
    # print(f"요청 에러 발생: {req_err}")
    pass
"""
st.code(code_fetch_html, language='python')

if st.checkbox("웹 페이지 HTML 내용 가져오기 예시 실행", key="fetch_html_page_4"):
    example_url_scrape = "http://books.toscrape.com/" # 스크레이핑 연습용 사이트
    st.write(f"대상 URL: `{example_url_scrape}`")
    st.caption("`books.toscrape.com`은 스크레이핑 연습을 위해 제공되는 웹사이트입니다.")
    
    try:
        response_scrape = requests.get(example_url_scrape, timeout=10)
        response_scrape.raise_for_status()
        st.success(f"'{example_url_scrape}' HTML 내용 가져오기 성공! (상태 코드: {response_scrape.status_code})")
        
        with st.expander("가져온 HTML 내용 보기 (상위 1000자)", expanded=False):
            st.code(response_scrape.text[:1000] + "...", language='html')
            
    except requests.exceptions.HTTPError as http_err:
        st.error(f"HTTP 에러 발생: {http_err} (상태 코드: {response_scrape.status_code if 'response_scrape' in locals() and hasattr(response_scrape, 'status_code') else 'N/A'})")
    except requests.exceptions.RequestException as req_err:
        st.error(f"요청 중 에러 발생: {req_err}")

st.markdown("---")

# --- 4.2 HTML 파싱 (`BeautifulSoup4`) ---
st.subheader("4.2 HTML 파싱 (`BeautifulSoup4`)")
st.markdown("""
가져온 HTML 문자열에서 원하는 정보를 추출하려면 HTML 구조를 분석(파싱)해야 합니다. `BeautifulSoup4` (보통 `bs4`로 임포트)는 이를 위한 강력하고 사용하기 쉬운 라이브러리입니다.
- `pip install beautifulsoup4 lxml` 명령으로 설치합니다. (`lxml`은 빠르고 안정적인 HTML 파서입니다.)
- `BeautifulSoup(html_content, 'html.parser')` 또는 `BeautifulSoup(html_content, 'lxml')`로 객체를 생성합니다.

**주요 메소드:**
- `soup.find('태그이름', attrs={'속성이름': '속성값', ...})`: 특정 조건을 만족하는 첫 번째 요소를 찾습니다.
- `soup.find_all('태그이름', attrs={'속성이름': '속성값', ...}, limit=None)`: 특정 조건을 만족하는 모든 요소를 리스트로 찾습니다.
    - `class_` 파라미터: HTML `class` 속성으로 검색 (예: `class_='my-class'`).
- `soup.select('CSS 선택자')`: CSS 선택자를 사용하여 요소를 찾습니다 (더 유연한 선택 가능).
- `element.get_text(strip=True)`: 요소 내부의 텍스트 내용만 추출합니다 (`strip=True`는 앞뒤 공백 제거).
- `element['속성이름']`: 요소의 특정 속성 값을 가져옵니다 (예: `a_tag['href']`는 `<a>` 태그의 `href` 속성값).
""")
st.warning("💡 `BeautifulSoup4`와 함께 `lxml` 파서를 사용하는 것이 일반적입니다. `pip install lxml`로 설치할 수 있습니다.")

code_parse_html = """
import requests
from bs4 import BeautifulSoup

url = "https.books.toscrape.com/" # 스크레이핑 연습용 사이트
# html_content = requests.get(url).text # 이미 가져왔다고 가정

# # 가상의 HTML 내용 (실제로는 위에서 가져온 html_content 사용)
# html_example = \"\"\"
# <html><head><title>My Page</title></head>
# <body><h1>A Big Heading</h1>
# <p class="content-text">This is a paragraph.</p>
# <p class="content-text" id="second-p">Another paragraph.</p>
# <a href="https://example.com">Click here</a>
# <ul><li>Item 1</li><li>Item 2</li></ul></body></html>
# \"\"\"

# BeautifulSoup 객체 생성 (lxml 파서 사용 권장)
# soup = BeautifulSoup(html_example, 'lxml')

# # 특정 태그 찾기
# title_tag = soup.find('title')
# # print(f"페이지 제목: {title_tag.get_text(strip=True) if title_tag else '제목 없음'}")

# h1_tag = soup.find('h1')
# # print(f"H1 태그 내용: {h1_tag.get_text(strip=True) if h1_tag else 'H1 없음'}")

# # 특정 클래스를 가진 모든 <p> 태그 찾기
# content_paragraphs = soup.find_all('p', class_='content-text')
# # print("\\n클래스가 'content-text'인 문단들:")
# # for i, p_tag in enumerate(content_paragraphs):
# #     print(f"  문단 {i+1}: {p_tag.get_text(strip=True)}")

# # CSS 선택자 사용 (예: id가 'second-p'인 요소)
# second_p_selector = soup.select_one('#second-p') # select는 리스트 반환, select_one은 단일 요소
# # print(f"\\nID가 'second-p'인 문단 내용: {second_p_selector.get_text(strip=True) if second_p_selector else 'ID 없음'}")

# # 링크(<a> 태그)의 href 속성 값 가져오기
# link_tag = soup.find('a')
# # if link_tag and 'href' in link_tag.attrs:
# #     print(f"\\n첫 번째 링크 URL: {link_tag['href']}")
"""
st.code(code_parse_html, language='python')

if st.checkbox("`BeautifulSoup4` 파싱 및 데이터 추출 예시 실행", key="parse_html_page_4"):
    scrape_target_url = "http://books.toscrape.com/"
    st.write(f"대상 URL: `{scrape_target_url}`")
    st.caption("`books.toscrape.com`의 첫 페이지에서 책 제목과 가격을 추출합니다.")

    try:
        response = requests.get(scrape_target_url, timeout=10)
        response.raise_for_status()
        html_to_parse = response.text
        
        soup = BeautifulSoup(html_to_parse, 'lxml') # lxml 파서 사용

        # 책 정보를 담고 있는 <article class="product_pod"> 태그들을 모두 찾음
        book_articles = soup.find_all('article', class_='product_pod')
        
        if book_articles:
            st.success(f"총 {len(book_articles)}개의 책 정보를 찾았습니다. (상위 5개만 표시)")
            scraped_data = []
            for i, article in enumerate(book_articles[:5]): # 상위 5개만 처리
                # 책 제목: <h3> 태그 안의 <a> 태그의 title 속성 또는 텍스트
                title_tag = article.find('h3').find('a')
                title = title_tag['title'] if title_tag and 'title' in title_tag.attrs else title_tag.get_text(strip=True) if title_tag else "제목 없음"
                
                # 책 가격: <p class="price_color"> 태그의 텍스트
                price_tag = article.find('p', class_='price_color')
                price = price_tag.get_text(strip=True) if price_tag else "가격 정보 없음"
                
                # (선택) 책 링크: <h3> 태그 안의 <a> 태그의 href 속성
                link = title_tag['href'] if title_tag and 'href' in title_tag.attrs else "링크 없음"
                full_link = requests.compat.urljoin(scrape_target_url, link) # 상대 경로를 절대 경로로 변환
                
                scraped_data.append({'Title': title, 'Price': price, 'Link': full_link})
            
            df_scraped = pd.DataFrame(scraped_data)
            st.dataframe(df_scraped)

            # 유틸리티 함수 사용 예시 (제목만 추출)
            # titles_elements = [article.find('h3').find('a') for article in book_articles if article.find('h3') and article.find('h3').find('a')]
            # display_scraped_elements(titles_elements, title="추출된 책 제목 (일부)", element_description="제목", max_elements_to_show=5)

        else:
            st.warning("책 정보를 담고 있는 `article.product_pod` 요소를 찾지 못했습니다.")

    except ImportError:
        st.error("`lxml` 파서가 필요합니다. `pip install lxml` 명령으로 설치해주세요.")
    except requests.exceptions.RequestException as e:
        st.error(f"웹 페이지 요청 중 오류 발생: {e}")
    except Exception as e:
        st.error(f"HTML 파싱 또는 데이터 추출 중 오류 발생: {e}")


st.markdown("---")
st.subheader("4.3 웹 스크레이핑의 한계와 주의사항 (재강조)")
st.markdown("""
- **정적 vs. 동적 웹사이트:** `requests`와 `BeautifulSoup`은 주로 정적인 HTML 내용을 가져와 파싱합니다. JavaScript를 통해 동적으로 내용이 로드되거나 변경되는 웹사이트(Single Page Applications 등)의 데이터는 이 방법만으로는 수집하기 어렵습니다. 이 경우 Selenium, Playwright와 같은 브라우저 자동화 도구가 필요하며, 이는 더 복잡한 기술입니다.
- **웹사이트 구조 변경:** 웹사이트의 HTML 구조는 언제든지 변경될 수 있습니다. 구조가 변경되면 기존 스크레이핑 코드가 더 이상 작동하지 않을 수 있으므로, 코드의 유지보수가 필요합니다.
- **CAPTCHA 및 안티-스크레이핑 기술:** 많은 웹사이트는 CAPTCHA, IP 기반 차단, 사용자 에이전트 검사 등 다양한 안티-스크레이핑 기술을 사용하여 자동화된 접근을 막습니다. 이러한 보호 장치를 우회하려는 시도는 서비스 약관 위반이며 피해야 합니다.
- **법적/윤리적 책임:** 앞서 강조했듯이, 항상 합법적이고 윤리적인 범위 내에서만 스크레이핑을 수행해야 합니다.

**학습 목적으로 스크레이핑을 연습할 때는 `books.toscrape.com`이나 `toscrape.com`과 같이 스크레이핑을 위해 명시적으로 제공된 웹사이트를 사용하는 것이 안전합니다.**
""")