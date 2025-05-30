# pages/1_📈_데이터_분포.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from utils.utils import get_sample_data # utils.py에서 샘플 데이터 함수 가져오기

st.set_page_config(layout="wide") # 페이지별로도 set_page_config를 호출할 수 있으나, 메인 앱에서 한 번만 하는 것이 일반적입니다.
                                 # 여기서는 각 페이지가 독립적으로 실행될 수 있도록 참고용으로 남겨둡니다.
                                 # 실제 멀티페이지 앱에서는 Home.py의 set_page_config가 적용됩니다.

st.header("1. 데이터 분포 확인 (단일 변수)")
st.markdown("""
하나의 연속형 또는 범주형 변수의 분포를 파악하고 싶을 때 사용합니다.
- **연속형 데이터:** 값의 범위, 중심 경향, 데이터의 퍼짐 정도, 이상치 등을 확인합니다.
- **범주형 데이터:** 각 범주에 속하는 데이터의 빈도수를 확인합니다.
""")

# --- 1.1 히스토그램 (Histogram) ---
st.subheader("1.1 히스토그램 (Histogram)")
st.markdown("""
연속형 변수의 분포를 막대 형태로 표현합니다. 데이터의 특정 구간에 몇 개의 관측치가 있는지 보여줍니다.
- **주요 사용 라이브러리:** `matplotlib.pyplot.hist()`, `seaborn.histplot()`, `plotly.express.histogram()`
- **언제 사용하나요?** 데이터 값의 빈도, 분포의 모양(대칭성, 첨도 등)을 파악할 때.
""")

# Matplotlib 예시
st.markdown("#### Matplotlib 히스토그램")
code_hist_mpl_dist = """
# matplotlib.pyplot을 plt라는 별칭으로 가져옵니다.
import matplotlib.pyplot as plt
# numpy를 np라는 별칭으로 가져옵니다. (샘플 데이터 생성용)
import numpy as np

# 샘플 데이터 생성: 평균 0, 표준편차 1의 정규분포를 따르는 난수 1000개
data = np.random.randn(1000)

# 새로운 그림(figure)과 축(axes)을 생성합니다. figsize로 크기 지정.
plt.figure(figsize=(8, 4))
# 히스토그램을 그립니다.
# data: 시각화할 데이터 배열
# bins: 막대의 개수 (구간 수).
# edgecolor: 각 막대의 테두리 색상
plt.hist(data, bins=30, edgecolor='black', color='skyblue')
# 차트 제목 설정
plt.title('Matplotlib Histogram')
# x축 레이블 설정
plt.xlabel('Value')
# y축 레이블 설정
plt.ylabel('Frequency')

# st.pyplot(plt.gcf()) # Streamlit 앱에 Matplotlib 그림을 표시합니다.
# plt.clf() # 다음 플롯을 위해 현재 그림을 초기화합니다.
"""
st.code(code_hist_mpl_dist, language='python')
if st.checkbox("Matplotlib 히스토그램 예시 보기", key="hist_mpl_dist_page"):
    sample_data_num_df = get_sample_data('numerical')
    data_for_hist = sample_data_num_df['A']
    fig, ax = plt.subplots(figsize=(8,4))
    ax.hist(data_for_hist, bins=20, edgecolor='black', color='skyblue')
    ax.set_title('Matplotlib Histogram Example (Column A)')
    ax.set_xlabel('Value of Column A')
    ax.set_ylabel('Frequency')
    st.pyplot(fig)

# Seaborn 예시
st.markdown("#### Seaborn 히스토그램")
code_hist_sns_dist = """
# seaborn을 sns라는 별칭으로 가져옵니다.
import seaborn as sns
# matplotlib.pyplot을 plt라는 별칭으로 가져옵니다. (차트 제목 등 추가 설정용)
import matplotlib.pyplot as plt
# numpy를 np라는 별칭으로 가져옵니다. (샘플 데이터 생성용)
import numpy as np

# 샘플 데이터 생성
data = np.random.randn(1000)

# 새로운 그림(figure)과 축(axes)을 생성합니다.
plt.figure(figsize=(8, 4))
# Seaborn으로 히스토그램(분포 플롯)을 그립니다.
# data: 시각화할 데이터 배열
# bins: 막대의 개수
# kde: Kernel Density Estimate (커널 밀도 추정) 곡선 표시 여부
# color: 막대 색상
sns.histplot(data, bins=30, kde=True, color='lightcoral')
# 차트 제목 설정
plt.title('Seaborn Histplot')
# x축 레이블 설정
plt.xlabel('Value')
# y축 레이블 설정
plt.ylabel('Frequency')

# st.pyplot(plt.gcf())
# plt.clf()
"""
st.code(code_hist_sns_dist, language='python')
if st.checkbox("Seaborn 히스토그램 예시 보기", key="hist_sns_dist_page"):
    sample_data_num_df = get_sample_data('numerical')
    data_for_hist = sample_data_num_df['A']
    fig, ax = plt.subplots(figsize=(8,4))
    sns.histplot(data_for_hist, bins=20, kde=True, ax=ax, color='lightcoral')
    ax.set_title('Seaborn Histplot Example (Column A)')
    ax.set_xlabel('Value of Column A')
    ax.set_ylabel('Frequency')
    st.pyplot(fig)

# Plotly Express 예시
st.markdown("#### Plotly Express 히스토그램")
code_hist_plotly_dist = """
# Plotly Express 라이브러리를 px라는 별칭으로 사용합니다.
import plotly.express as px
# pandas로 데이터프레임을 다루기 위해 import 합니다.
import pandas as pd
# numpy로 숫자 데이터를 생성하기 위해 import 합니다.
import numpy as np

# 샘플 데이터 생성
data_array = np.random.randn(1000)
# 생성된 배열을 'value'라는 컬럼을 가진 pandas DataFrame으로 변환합니다.
df = pd.DataFrame({'value': data_array})

# Plotly Express를 사용하여 히스토그램을 생성합니다.
fig = px.histogram(
    df,                    # 사용할 데이터프레임
    x="value",             # x축으로 사용할 컬럼 ('value')
    nbins=30,              # 막대의 개수 (구간 수)
    title="Plotly Express 히스토그램", # 차트 제목
    marginal="rug",        # x축 상단에 'rug plot' 추가 (옵션: "box", "violin")
    opacity=0.8            # 막대의 투명도 설정 (0.0 ~ 1.0)
)
# 차트 레이아웃 업데이트
fig.update_layout(
    xaxis_title_text='값 (Value)',  # x축 제목
    yaxis_title_text='빈도 (Frequency)', # y축 제목
    bargap=0.1                      # 막대 사이의 간격
)
# st.plotly_chart(fig, use_container_width=True)
"""
st.code(code_hist_plotly_dist, language='python')
if st.checkbox("Plotly 히스토그램 예시 보기", key="hist_plotly_dist_page"):
    sample_data_num_df = get_sample_data('numerical')
    fig = px.histogram(
        sample_data_num_df, x="A", nbins=30,
        title="Plotly Express 히스토그램 (컬럼 A)",
        marginal="box", color_discrete_sequence=['indianred']
    )
    fig.update_layout(xaxis_title_text='컬럼 A의 값', yaxis_title_text='빈도수')
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# --- 1.2 박스 플롯 (Box Plot) ---
st.subheader("1.2 박스 플롯 (Box Plot)")
st.markdown("""
데이터의 사분위수, 중앙값, 이상치 등을 시각적으로 표현합니다. 여러 그룹 간 분포를 비교할 때 유용합니다.
- **주요 사용 라이브러리:** `matplotlib.pyplot.boxplot()`, `seaborn.boxplot()`, `plotly.express.box()`
- **언제 사용하나요?** 데이터의 대략적인 분포, 이상치 확인, 그룹 간 분포 비교 시.
""")

# Matplotlib 예시
st.markdown("#### Matplotlib 박스 플롯")
code_box_mpl_dist = """
# matplotlib.pyplot을 plt라는 별칭으로 가져옵니다.
import matplotlib.pyplot as plt
# numpy를 np라는 별칭으로 가져옵니다.
import numpy as np

# 샘플 데이터 생성 (두 그룹)
data1 = np.random.normal(0, 1, 100) # 평균 0, 표준편차 1
data2 = np.random.normal(1, 1.5, 100) # 평균 1, 표준편차 1.5

# 새로운 그림과 축 생성
plt.figure(figsize=(8, 5))
# 박스 플롯을 그립니다.
# [data1, data2]: 시각화할 데이터 리스트 (각 요소가 하나의 박스가 됨)
# labels: 각 박스에 대한 레이블
plt.boxplot([data1, data2], labels=['Group1', 'Group2'], patch_artist=True) # patch_artist=True로 색상 채우기 가능
# 차트 제목 설정
plt.title('Matplotlib Boxplot')
# y축 레이블 설정
plt.ylabel('Value')

# st.pyplot(plt.gcf())
# plt.clf()
"""
st.code(code_box_mpl_dist, language='python')
if st.checkbox("Matplotlib 박스 플롯 예시 보기", key="box_mpl_dist_page"):
    sample_data_num_df = get_sample_data('numerical')
    data_for_boxplot = [sample_data_num_df['A'], sample_data_num_df['B']]
    labels = ['Column A', 'Column B']
    fig, ax = plt.subplots(figsize=(8,5))
    # patch_artist=True로 설정하면 박스 내부 색상을 변경할 수 있습니다.
    box_plot = ax.boxplot(data_for_boxplot, labels=labels, patch_artist=True, medianprops={'color':'black'})
    colors = ['lightblue', 'lightgreen'] # 각 박스에 대한 색상
    for patch, color in zip(box_plot['boxes'], colors):
        patch.set_facecolor(color) # 박스 색상 채우기
    ax.set_title('Matplotlib Boxplot Example (Columns A, B)')
    ax.set_ylabel('Value')
    st.pyplot(fig)

# Seaborn 예시
st.markdown("#### Seaborn 박스 플롯")
code_box_sns_dist = """
# seaborn을 sns라는 별칭으로 가져옵니다.
import seaborn as sns
# matplotlib.pyplot을 plt라는 별칭으로 가져옵니다.
import matplotlib.pyplot as plt
# pandas를 pd라는 별칭으로 가져옵니다. (데이터프레임 사용)
import pandas as pd
# numpy를 np라는 별칭으로 가져옵니다.
import numpy as np

# 샘플 데이터 생성 (데이터프레임 형태)
data_dict = {
    'Group1': np.random.normal(0, 1, 100),
    'Group2': np.random.normal(1, 1.5, 100)
}
df = pd.DataFrame(data_dict)

# 새로운 그림과 축 생성
plt.figure(figsize=(8, 5))
# Seaborn으로 박스 플롯을 그립니다.
# data: 사용할 데이터프레임
# palette: 색상 팔레트
sns.boxplot(data=df, palette="pastel")
# 차트 제목 설정
plt.title('Seaborn Boxplot')
# y축 레이블 설정
plt.ylabel('Value')

# st.pyplot(plt.gcf())
# plt.clf()
"""
st.code(code_box_sns_dist, language='python')
if st.checkbox("Seaborn 박스 플롯 예시 보기", key="box_sns_dist_page"):
    sample_data_num_df = get_sample_data('numerical')
    fig, ax = plt.subplots(figsize=(8,5))
    sns.boxplot(data=sample_data_num_df[['A', 'B']], ax=ax, palette="pastel")
    ax.set_title('Seaborn Boxplot Example (Columns A, B)')
    ax.set_ylabel('Value')
    st.pyplot(fig)

# Plotly Express 예시
st.markdown("#### Plotly Express 박스 플롯")
code_box_plotly_dist = """
# Plotly Express 라이브러리를 px라는 별칭으로 사용합니다.
import plotly.express as px
# pandas로 데이터프레임을 다루기 위해 import 합니다.
import pandas as pd
# numpy로 숫자 데이터를 생성/조작하기 위해 import 합니다.
import numpy as np

# 샘플 데이터 생성
data_dict = {
    'Group1': np.random.normal(0, 1, 100),
    'Group2': np.random.normal(1.5, 1.5, 100)
}
df = pd.DataFrame(data_dict)

# Plotly Express를 사용하여 박스 플롯을 생성합니다.
# DataFrame을 직접 전달하면 각 숫자형 컬럼에 대해 박스 플롯을 그립니다.
fig = px.box(
    df,                    # 사용할 데이터프레임
    points="all",          # 모든 데이터 포인트를 함께 표시 (옵션: "outliers", False)
    title="Plotly Express 박스 플롯" # 차트 제목
)
# 차트 레이아웃 업데이트
fig.update_layout(
    yaxis_title_text='값 (Value)' # y축 제목
)
# st.plotly_chart(fig, use_container_width=True)
"""
st.code(code_box_plotly_dist, language='python')
if st.checkbox("Plotly 박스 플롯 예시 보기", key="box_plotly_dist_page"):
    sample_data_num_df = get_sample_data('numerical')
    fig = px.box(
        sample_data_num_df[['A', 'B']], # 데이터프레임에서 'A'와 'B' 컬럼 선택
        points="outliers",        # 이상치(outliers)만 점으로 표시
        title="Plotly Express 박스 플롯 (컬럼 A, B)", # 차트 제목
        color_discrete_sequence=px.colors.qualitative.Set2 # 색상 팔레트 지정
    )
    fig.update_layout(yaxis_title_text='값')
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# --- 1.3 범주형 데이터 막대 그래프 (Bar Plot for Categorical Data) ---
st.subheader("1.3 범주형 데이터 막대 그래프 (Bar Plot for Categorical Data)")
st.markdown("""
범주형 데이터의 각 범주별 빈도수(count)를 막대로 표현합니다.
- **주요 사용 라이브러리:** `matplotlib.pyplot.bar()`, `seaborn.countplot()`, `plotly.express.bar()` (데이터 사전 집계 필요)
- **언제 사용하나요?** 범주별 항목 수 비교 시.
""")

# Matplotlib 예시
st.markdown("#### Matplotlib 범주형 막대 그래프")
code_bar_cat_mpl_dist = """
# matplotlib.pyplot을 plt라는 별칭으로 가져옵니다.
import matplotlib.pyplot as plt
# pandas를 pd라는 별칭으로 가져옵니다. (데이터 처리용)
import pandas as pd

# 샘플 데이터 생성
categories = ['A', 'B', 'C', 'A', 'B', 'A', 'D', 'B', 'C', 'A']
df = pd.DataFrame({'Category': categories})

# 각 범주별 빈도수 계산
counts = df['Category'].value_counts()

# 새로운 그림과 축 생성
plt.figure(figsize=(8, 5))
# 막대 그래프를 그립니다.
# counts.index: x축 값 (범주 이름)
# counts.values: y축 값 (빈도수)
plt.bar(counts.index, counts.values, color=['skyblue', 'lightcoral', 'lightgreen', 'gold'])
# 차트 제목 설정
plt.title('Matplotlib Bar Plot (Categorical Counts)')
# x축 레이블 설정
plt.xlabel('Category')
# y축 레이블 설정
plt.ylabel('Frequency')

# st.pyplot(plt.gcf())
# plt.clf()
"""
st.code(code_bar_cat_mpl_dist, language='python')
if st.checkbox("Matplotlib 범주형 막대 그래프 예시 보기", key="bar_cat_mpl_dist_page"):
    sample_data_cat_df = get_sample_data('categorical')
    counts = sample_data_cat_df['Category'].value_counts()
    fig, ax = plt.subplots(figsize=(8,5))
    ax.bar(counts.index, counts.values, color=['#66c2a5','#fc8d62','#8da0cb','#e78ac3']) # 색상 지정
    ax.set_title('Matplotlib Bar Plot (Categorical Counts)')
    ax.set_xlabel('Category')
    ax.set_ylabel('Frequency')
    st.pyplot(fig)

# Seaborn 예시
st.markdown("#### Seaborn 범주형 막대 그래프 (countplot)")
code_bar_cat_sns_dist = """
# seaborn을 sns라는 별칭으로 가져옵니다.
import seaborn as sns
# matplotlib.pyplot을 plt라는 별칭으로 가져옵니다.
import matplotlib.pyplot as plt
# pandas를 pd라는 별칭으로 가져옵니다.
import pandas as pd

# 샘플 데이터 생성
categories = ['A', 'B', 'C', 'A', 'B', 'A', 'D', 'B', 'C', 'A']
df = pd.DataFrame({'Category': categories})

# 새로운 그림과 축 생성
plt.figure(figsize=(8, 5))
# Seaborn으로 countplot을 그립니다. (자동으로 빈도수 계산)
# x: x축으로 사용할 데이터프레임의 컬럼 이름
# data: 사용할 데이터프레임
# palette: 색상 팔레트
# order: 막대 순서 지정 (선택 사항)
sns.countplot(x='Category', data=df, palette='viridis', order=df['Category'].value_counts().index)
# 차트 제목 설정
plt.title('Seaborn Count Plot')
# x축 레이블 설정
plt.xlabel('Category')
# y축 레이블 설정
plt.ylabel('Frequency')

# st.pyplot(plt.gcf())
# plt.clf()
"""
st.code(code_bar_cat_sns_dist, language='python')
if st.checkbox("Seaborn 범주형 막대 그래프 예시 보기", key="bar_cat_sns_dist_page"):
    sample_data_cat_df = get_sample_data('categorical')
    fig, ax = plt.subplots(figsize=(8,5))
    # value_counts().index를 사용하여 빈도수가 높은 순으로 정렬
    order = sample_data_cat_df['Category'].value_counts().index
    sns.countplot(x='Category', data=sample_data_cat_df, ax=ax, palette='Set2', order=order)
    ax.set_title('Seaborn Count Plot')
    ax.set_xlabel('Category')
    ax.set_ylabel('Frequency')
    st.pyplot(fig)

# Plotly Express 예시
st.markdown("#### Plotly Express 범주형 막대 그래프")
code_bar_cat_plotly_dist = """
# Plotly Express 라이브러리를 px라는 별칭으로 사용합니다.
import plotly.express as px
# pandas로 데이터프레임을 다루기 위해 import 합니다.
import pandas as pd

# 샘플 데이터 생성
categories = ['A', 'B', 'C', 'A', 'B', 'A', 'D', 'B', 'C', 'A']
df = pd.DataFrame({'Category': categories})

# 각 범주별 빈도수 계산 (Plotly bar는 집계된 데이터를 주로 사용)
category_counts_df = df['Category'].value_counts().reset_index()
# 컬럼 이름 변경: 기존 인덱스 컬럼 -> 'Category', 값 컬럼 -> 'Count'
category_counts_df.columns = ['Category', 'Count']

# Plotly Express를 사용하여 막대 그래프를 생성합니다.
fig = px.bar(
    category_counts_df,    # 사용할 데이터프레임 (범주별 빈도수)
    x='Category',          # x축으로 사용할 컬럼 (범주)
    y='Count',             # y축으로 사용할 컬럼 (빈도수)
    title="Plotly Express 범주형 데이터 빈도수 막대 그래프", # 차트 제목
    color='Category',      # 각 막대의 색상을 'Category' 값에 따라 다르게 지정
    text_auto=True         # 막대 위에 값(빈도수)을 자동으로 표시
)
# 차트 레이아웃 업데이트
fig.update_layout(
    xaxis_title_text='범주 (Category)', # x축 제목
    yaxis_title_text='빈도 (Frequency)'  # y축 제목
)
# st.plotly_chart(fig, use_container_width=True)
"""
st.code(code_bar_cat_plotly_dist, language='python')
if st.checkbox("Plotly 범주형 막대 그래프 예시 보기", key="bar_cat_plotly_dist_page"):
    sample_data_cat_df = get_sample_data('categorical')
    category_counts_df = sample_data_cat_df['Category'].value_counts().reset_index()
    category_counts_df.columns = ['Category', '빈도수']

    fig = px.bar(
        category_counts_df, x='Category', y='빈도수',
        title="Plotly Express 범주형 데이터 빈도수",
        color='Category', text_auto=True, # text_auto는 막대 위에 값을 표시
        color_discrete_map={'X':'#1f77b4', 'Y':'#ff7f0e', 'Z':'#2ca02c', 'W':'#d62728'} # 특정 카테고리 색상 매핑
    )
    fig.update_layout(xaxis_title_text='카테고리', yaxis_title_text='빈도수')
    st.plotly_chart(fig, use_container_width=True)