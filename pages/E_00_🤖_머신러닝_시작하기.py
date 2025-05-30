# 🤖_머신러닝_도우미_Home.py
import streamlit as st
import matplotlib.pyplot as plt
import platform # 현재 운영체제 확인용

# 한글 폰트 설정 (Streamlit 앱 전체에 적용)
def apply_korean_font():
    try:
        # 사용자의 운영체제에 따라 폰트 경로 자동 지정 시도
        os_name = platform.system()
        font_name = "Malgun Gothic" # Windows 기본 한글 폰트

        if os_name == "Darwin": # MacOS
            font_name = "AppleGothic"
        elif os_name == "Linux":
            # Linux에서는 NanumGothic이 설치되어 있다고 가정
            # 실제 경로를 확인하거나, matplotlib 폰트 매니저에 추가해야 할 수 있음
            # 예: /usr/share/fonts/truetype/nanum/NanumGothic.ttf
            # 여기서는 간단히 이름만 지정, 시스템에 따라 경로 지정 필요할 수 있음
            # 가장 확실한 방법은 matplotlib.font_manager를 사용하는 것
            font_name = "NanumGothic" # 또는 다른 설치된 한글 폰트

        plt.rcParams['font.family'] = font_name
        plt.rcParams['axes.unicode_minus'] = False # 마이너스 부호 깨짐 방지
        print(f"적용된 한글 폰트: {font_name} (운영체제: {os_name})")

    except Exception as e:
        print(f"한글 폰트 설정 중 오류 발생: {e}")
        # 예외 발생 시 기본 폰트로 동작하도록 함
        pass

# 앱 시작 시 한글 폰트 적용
apply_korean_font()

# --- 페이지 설정 ---
st.set_page_config(
    layout="wide",
    page_title="머신러닝 도우미",
    page_icon="🤖"
)

st.title("🤖 머신러닝 도우미 with Scikit-learn")
st.markdown("""
안녕하세요! 이 앱은 파이썬의 대표적인 머신러닝 라이브러리인 **Scikit-learn**을 활용하여
주요 머신러닝 개념과 모델 사용법을 학습하는 데 도움을 드립니다.

**👈 왼쪽 사이드바에서 학습하고 싶은 머신러닝 주제를 선택하세요.**

각 페이지에서는 다음과 같은 내용을 확인할 수 있습니다:
-   **핵심 개념 및 알고리즘 설명:** 각 머신러닝 기법의 원리와 특징을 알아봅니다.
-   **Scikit-learn 코드 예시:** 실제 Scikit-learn 코드와 함께 상세한 주석을 통해 사용법을 익힐 수 있습니다.
-   **실행 결과 및 해석:** 모델 학습 결과, 평가지표 등을 직접 확인하며 이해도를 높입니다.

이 앱을 통해 머신러닝의 기초를 다지고, 다양한 문제에 Scikit-learn을 효과적으로 적용하는 능력을 키워보세요!
""")

st.sidebar.success("위에서 학습할 주제를 선택하세요.")

st.markdown("---")
st.subheader("Scikit-learn 이란?")
st.markdown("""
Scikit-learn은 파이썬을 위한 사용하기 쉬우면서도 강력한 오픈소스 머신러닝 라이브러리입니다.
다양한 분류, 회귀, 군집화, 차원 축소, 모델 선택, 데이터 전처리 알고리즘을 제공합니다.

**Scikit-learn의 주요 특징:**
-   **일관된 API:** `Estimator` 객체를 중심으로 `fit()`, `predict()`, `transform()` 등의 일관된 메소드를 제공하여 사용이 편리합니다.
-   **풍부한 알고리즘:** 최신 머신러닝 알고리즘부터 전통적인 기법까지 폭넓게 지원합니다.
-   **활발한 커뮤니티와 상세한 문서:** 방대한 사용자 커뮤니티와 잘 정리된 공식 문서를 통해 학습과 문제 해결이 용이합니다.
-   NumPy, SciPy, Pandas, Matplotlib 등 다른 파이썬 과학 컴퓨팅 라이브러리와 잘 통합됩니다.
""")

st.info("💡 이 앱의 예제는 Scikit-learn의 기본적인 사용법에 초점을 맞추고 있으며, 실제 머신러닝 프로젝트에서는 더 깊이 있는 이론 학습과 섬세한 모델 튜닝, 데이터 분석 과정이 필요합니다.")