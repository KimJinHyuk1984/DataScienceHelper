# utils_collection.py
import streamlit as st
import json
import requests # display_api_response 함수에서 requests.exceptions를 사용하기 위해 필요

def display_api_response(response, title="API 응답 결과"):
    """Streamlit에 API 응답 객체의 상태와 내용을 표시합니다."""
    st.subheader(title)
    try:
        st.write(f"- **상태 코드 (Status Code):** `{response.status_code}`")
        response.raise_for_status() # HTTP 에러 발생 시 예외를 일으킴 (4xx 또는 5xx 상태 코드)
        st.success("API 요청 성공!")
        
        content_type = response.headers.get('content-type', '').lower()
        
        if 'application/json' in content_type:
            try:
                json_data = response.json()
                st.write("- **응답 형식:** JSON")
                with st.expander("JSON 응답 내용 보기 (일부 또는 전체)", expanded=False):
                    st.json(json_data) # Streamlit이 JSON을 예쁘게 표시
            except json.JSONDecodeError:
                st.error("JSON 파싱 오류: 응답 내용이 유효한 JSON 형식이 아닙니다.")
                st.text_area("Raw 응답 내용 (상위 1000자)", response.text[:1000] + ("..." if len(response.text) > 1000 else ""), height=150)
        elif 'text/html' in content_type:
            st.write("- **응답 형식:** HTML")
            with st.expander("HTML 응답 내용 보기 (상위 1000자)", expanded=False):
                st.code(response.text[:1000] + ("..." if len(response.text) > 1000 else ""), language='html')
        elif 'text/plain' in content_type:
            st.write("- **응답 형식:** 일반 텍스트")
            with st.expander("응답 내용 보기 (상위 1000자)", expanded=False):
                st.text_area("응답 내용", response.text[:1000] + ("..." if len(response.text) > 1000 else ""), height=150)
        else:
            st.write(f"- **응답 형식:** {content_type} (알 수 없음)")
            with st.expander("Raw 응답 내용 보기 (상위 1000자)", expanded=False):
                st.text_area("Raw 응답 내용", response.text[:1000] + ("..." if len(response.text) > 1000 else ""), height=150)
            
    except requests.exceptions.HTTPError as http_err:
        st.error(f"HTTP 에러 발생: {http_err}")
        st.text_area("에러 응답 내용 (상위 1000자)", response.text[:1000] + ("..." if len(response.text) > 1000 else ""), height=150)
    except requests.exceptions.RequestException as req_err:
        st.error(f"요청 중 에러 발생: {req_err}")
    except Exception as e:
        st.error(f"응답 처리 중 예기치 않은 오류 발생: {e}")
    st.markdown("---")

def display_scraped_elements(elements, title="스크레이핑된 요소", element_description="요소", max_elements_to_show=10):
    """BeautifulSoup으로 찾은 요소들의 리스트를 Streamlit에 표시합니다."""
    st.subheader(title)
    if elements:
        num_found = len(elements)
        st.write(f"총 {num_found}개의 {element_description}를 찾았습니다. (최대 {max_elements_to_show}개 표시)")
        for i, elem in enumerate(elements[:max_elements_to_show]):
            text_content = elem.get_text(strip=True)
            display_text = (text_content[:150] + '...') if len(text_content) > 150 else text_content
            with st.expander(f"{element_description} #{i+1}: \"{display_text}\""):
                st.code(str(elem.prettify()), language='html') # HTML 구조 보기
        if num_found > max_elements_to_show:
            st.caption(f"... 외 {num_found - max_elements_to_show}개의 {element_description}가 더 있습니다.")
    else:
        st.warning(f"해당 선택자로 {element_description}를 찾지 못했습니다.")
    st.markdown("---")