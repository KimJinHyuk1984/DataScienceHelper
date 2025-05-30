# 🏠_통합_데이터과학_도우미_Home.py
import streamlit as st
import platform
import matplotlib.pyplot as plt # 한글 폰트 설정을 위해 임포트
import pandas as pd # 현재 시간 표시용

# -----------------------------------------------------------------------------
# 한글 폰트 설정 함수
# -----------------------------------------------------------------------------
def apply_korean_font():
    try:
        os_name = platform.system()
        font_name = "Malgun Gothic"
        if os_name == "Darwin":
            font_name = "AppleGothic"
        elif os_name == "Linux":
            font_name = "NanumGothic"
        plt.rcParams['font.family'] = font_name
        plt.rcParams['axes.unicode_minus'] = False
    except Exception as e:
        st.sidebar.warning(f"한글 폰트 설정 중 오류: {e}. 일부 그래프에서 한글이 깨질 수 있습니다.")
        pass

# -----------------------------------------------------------------------------
# Streamlit 페이지 설정
# -----------------------------------------------------------------------------
st.set_page_config(
    layout="wide",
    page_title="통합 데이터과학 도우미",
    page_icon="🎓"
)

# -----------------------------------------------------------------------------
# 앱 시작 시 한글 폰트 적용
# -----------------------------------------------------------------------------
apply_korean_font()


# --- 메인 페이지 내용 ---
st.title("🎓 통합 데이터과학 도우미")
st.markdown("""
안녕하세요! 데이터 과학 학습 여정에 오신 것을 환영합니다.
이 통합 도우미는 데이터 분석 및 머신러닝 학습에 필요한 다양한 파이썬 라이브러리 및 개념을 쉽고 효과적으로 익힐 수 있도록 설계되었습니다.

**👈 왼쪽 사이드바에서 학습하고 싶은 주제를 선택하여 탐색을 시작하세요.**

각 주제별 도우미는 상세한 설명, 실행 가능한 코드 예제, 그리고 시각적인 결과물을 제공하여 여러분의 학습을 돕습니다.
""")

st.sidebar.success("탐색할 주제를 선택하세요. 👆")

st.markdown("---")

st.subheader("제공되는 학습 도우미")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown( # 데이터 수집 도우미 카드
        """
        <div style="border: 1px solid #CD4BFF; border-radius: 5px; padding: 10px; text-align: center; margin-bottom: 10px; height: 120px; display: flex; flex-direction: column; justify-content: center;">
            📥 **데이터 수집 도우미**<br>
            <small>파일, 웹 API, 스크레이핑, DB 등 다양한 데이터 수집 방법을 안내합니다.</small>
        </div>
        """, unsafe_allow_html=True
    )
    st.markdown( # NumPy 도우미 카드
        """
        <div style="border: 1px solid #4B8BFF; border-radius: 5px; padding: 10px; text-align: center; margin-bottom: 10px; height: 120px; display: flex; flex-direction: column; justify-content: center;">
            📖 **NumPy 도우미**<br>
            <small>수치 연산의 핵심, NumPy 배열 생성, 조작, 함수 사용법을 익힙니다.</small>
        </div>
        """, unsafe_allow_html=True
    )

with col2:
    st.markdown( # Pandas 도우미 카드
        """
        <div style="border: 1px solid #FFCD4B; border-radius: 5px; padding: 10px; text-align: center; margin-bottom: 10px; height: 120px; display: flex; flex-direction: column; justify-content: center;">
            🐼 **Pandas 도우미**<br>
            <small>데이터 분석의 필수 도구, Pandas Series와 DataFrame 다루는 법을 학습합니다.</small>
        </div>
        """, unsafe_allow_html=True
    )
    st.markdown( # 데이터 시각화 도우미 카드
        """
        <div style="border: 1px solid #FF4B4B; border-radius: 5px; padding: 10px; text-align: center; margin-bottom: 10px; height: 120px; display: flex; flex-direction: column; justify-content: center;">
            📊 **데이터 시각화 도우미**<br>
            <small>Matplotlib, Seaborn, Plotly, Folium을 활용한 다양한 그래프 작성을 안내합니다.</small>
        </div>
        """, unsafe_allow_html=True
    )

with col3:
    st.markdown( # 머신러닝 도우미 카드
        """
        <div style="border: 1px solid #4BFFCD; border-radius: 5px; padding: 10px; text-align: center; margin-bottom: 10px; height: 120px; display: flex; flex-direction: column; justify-content: center;">
            🤖 **머신러닝 도우미**<br>
            <small>Scikit-learn 기반의 주요 머신러닝 모델과 평가 방법을 알아봅니다.</small>
        </div>
        """, unsafe_allow_html=True
    )

st.markdown("---")

st.subheader("학습 가이드")
st.markdown("""
-   **순차적 학습 (권장):** 각 도우미 내의 페이지들은 보통 기초부터 심화 순으로 구성되어 있습니다. 순서대로 학습하시면 개념을 체계적으로 이해하는 데 도움이 됩니다.
-   **코드 실행 및 수정:** 제공되는 코드 예제는 직접 실행해보고, 값을 바꾸거나 새로운 코드를 추가하며 실험해보는 것이 좋습니다. `Streamlit`의 실시간 업데이트 기능을 활용하세요.
-   **이론과 실습 병행:** 각 기능에 대한 설명을 읽고, 코드 예제를 통해 실제로 어떻게 동작하는지 확인하며 이론과 실습을 병행하세요.
-   **질문과 탐구:** 궁금한 점이 생기면 주저하지 말고 추가적인 자료를 찾아보거나 질문하세요. 스스로 탐구하는 과정이 학습에 매우 중요합니다.
""")

st.info("""
💡 **참고:** 이 통합 도우미는 데이터 과학의 주요 도구와 개념을 소개하기 위한 학습 보조 자료입니다.
각 라이브러리와 기법은 훨씬 더 방대하고 깊이 있는 내용을 담고 있으므로, 이 앱을 시작으로 더 넓은 학습으로 나아가시길 바랍니다.
모든 예제 코드는 교육적 목적으로 제공되며, 실제 프로젝트나 상업적 용도로 사용 시에는 해당 데이터 및 코드의 라이선스, 서비스 이용 약관 등을 반드시 확인해야 합니다.
""")

st.caption(f"현재 시간: {pd.Timestamp.now(tz='Asia/Seoul').strftime('%Y-%m-%d %H:%M:%S %Z')}")