# pages/5_🧩_전체_대비_부분.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns # 여기서는 직접 사용하지 않지만, 일반적인 임포트
import plotly.express as px
from utils.utils import get_sample_data

st.header("5. 전체 중 부분의 비율 표시")
st.markdown("""
전체에 대한 각 부분의 비율이나 백분율을 나타낼 때 사용합니다.
""")

# --- 5.1 파이 차트 (Pie Chart) ---
st.subheader("5.1 파이 차트 (Pie Chart)")
st.markdown("""
전체에 대한 각 부분의 비율을 부채꼴 모양으로 표현합니다. 항목 수가 적을 때 (일반적으로 5개 이하) 효과적입니다.
- **주요 사용 라이브러리:** `matplotlib.pyplot.pie()`, `plotly.express.pie()`
- **언제 사용하나요?** 몇 개의 범주가 전체에서 차지하는 비율을 명확히 보여주고 싶을 때.
- **주의점:** 항목이 너무 많거나 비율 차이가 미미하면 가독성이 떨어질 수 있습니다. 이 경우 막대 그래프가 더 나을 수 있습니다.
""")

# Matplotlib 예시
st.markdown("#### Matplotlib 파이 차트")
code_pie_mpl_comp = """
# matplotlib.pyplot을 plt라는 별칭으로 가져옵니다.
import matplotlib.pyplot as plt

# 샘플 데이터
labels = 'Frogs', 'Hogs', 'Dogs', 'Logs' # 각 부분의 레이블
sizes = [15, 30, 45, 10]                # 각 부분의 크기 (비율)
explode = (0, 0.1, 0, 0)                # 특정 조각을 떼어내는 정도 (0은 붙어있음)

# 새로운 그림과 축 생성 (정사각형으로 만드는 것이 좋음)
fig, ax = plt.subplots(figsize=(7, 7))
# 파이 차트를 그립니다.
# sizes: 각 조각의 크기
# explode: 조각을 떼어내는 정도
# labels: 각 조각의 레이블
# autopct: 각 조각에 표시될 백분율 형식 (예: '%1.1f%%'는 소수점 첫째 자리까지)
# shadow: 그림자 효과
# startangle: 첫 번째 조각이 시작되는 각도 (0도는 x축 양의 방향)
# colors: (선택 사항) 각 조각의 색상 리스트
ax.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
       shadow=True, startangle=90, colors=plt.cm.Pastel1.colors) # Pastel1 색상맵 사용
# 파이 차트가 원형으로 보이도록 축 비율을 동일하게 설정
ax.axis('equal')
# 차트 제목 설정
plt.title('Matplotlib Pie Chart')

# st.pyplot(fig)
# plt.clf() # fig, ax 방식으로 사용했으므로 plt.clf() 불필요. fig 객체를 사용.
"""
st.code(code_pie_mpl_comp, language='python')
if st.checkbox("Matplotlib 파이 차트 예시 보기", key="pie_mpl_comp_page"):
    sample_cat_df = get_sample_data('categorical')
    counts = sample_cat_df['Category'].value_counts() # 카테고리별 빈도수 계산

    labels = counts.index
    sizes = counts.values
    # 가장 큰 조각을 약간 떼어내도록 설정
    explode_values = [0.05] * len(labels) # 기본값
    if len(labels) > 0:
        explode_values[np.argmax(sizes)] = 0.1 # 가장 큰 값의 인덱스를 찾아 explode 값 변경

    fig, ax = plt.subplots(figsize=(7,7))
    ax.pie(sizes, explode=explode_values, labels=labels, autopct='%1.1f%%',
           shadow=False, startangle=90, colors=plt.cm.Set3.colors[:len(labels)]) # Set3 색상맵 사용
    ax.axis('equal')
    ax.set_title('Matplotlib Pie Chart (Category Frequencies)')
    st.pyplot(fig)

# Plotly Express 예시
st.markdown("#### Plotly Express 파이 차트")
code_pie_plotly_comp = """
# Plotly Express 라이브러리를 px라는 별칭으로 사용합니다.
import plotly.express as px
# pandas로 데이터프레임을 다루기 위해 import 합니다.
import pandas as pd

# 샘플 데이터 (데이터프레임 형태)
data = {
    'Category': ['Frogs', 'Hogs', 'Dogs', 'Logs'],
    'Value': [15, 30, 45, 10]
}
df_pie = pd.DataFrame(data)

# Plotly Express를 사용하여 파이 차트를 생성합니다.
fig = px.pie(
    df_pie,                # 사용할 데이터프레임
    names='Category',      # 각 조각의 이름을 나타내는 컬럼
    values='Value',        # 각 조각의 크기를 나타내는 컬럼
    title="Plotly Express 파이 차트", # 차트 제목
    hole=0.3,              # 가운데 구멍 크기 (도넛 차트 효과, 0~1)
    color_discrete_sequence=px.colors.sequential.RdBu # 색상 시퀀스
)
# 차트 레이아웃 업데이트 (선택 사항)
# fig.update_traces(textposition='inside', textinfo='percent+label') # 조각 내부에 퍼센트와 레이블 표시
fig.update_layout(legend_title_text='범주') # 범례 제목

# st.plotly_chart(fig, use_container_width=True)
"""
st.code(code_pie_plotly_comp, language='python')
if st.checkbox("Plotly 파이 차트 예시 보기", key="pie_plotly_comp_page"):
    sample_cat_df = get_sample_data('categorical')
    counts_df = sample_cat_df['Category'].value_counts().reset_index()
    counts_df.columns = ['Category', 'Count'] # 컬럼 이름 변경

    fig = px.pie(
        counts_df, names='Category', values='Count',
        title='Plotly Express Pie Chart (Category Frequencies)',
        hole=0.3, # 도넛 차트 형태
        color_discrete_sequence=px.colors.qualitative.Pastel # 부드러운 색상 사용
    )
    # 조각 위에 퍼센트와 레이블 함께 표시, 글자 크기 조정
    fig.update_traces(textposition='outside', textinfo='percent+label', insidetextfont=dict(size=10))
    fig.update_layout(legend_title_text='카테고리', uniformtext_minsize=10, uniformtext_mode='hide') # 텍스트 크기 일관성
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# --- 5.2 누적 막대 그래프 (Stacked Bar Chart) ---
st.subheader("5.2 누적 막대 그래프 (Stacked Bar Chart)")
st.markdown("""
각 막대 내부를 여러 부분으로 나누어, 전체 값과 함께 각 부분의 크기를 보여줍니다.
- **주요 사용 라이브러리:** `matplotlib.pyplot.bar()` (bottom 파라미터 활용), `pandas.DataFrame.plot(kind='bar', stacked=True)`, `plotly.express.bar()` (barmode='stack' 또는 color 사용)
- **언제 사용하나요?** 각 범주 내에서 하위 항목들의 구성 비율 또는 절대량을 함께 비교하고 싶을 때.
""")

# Matplotlib 예시
st.markdown("#### Matplotlib 누적 막대 그래프")
code_stacked_bar_mpl_comp = """
# matplotlib.pyplot을 plt라는 별칭으로 가져옵니다.
import matplotlib.pyplot as plt
# numpy를 np라는 별칭으로 가져옵니다.
import numpy as np

# 샘플 데이터
labels = ['G1', 'G2', 'G3', 'G4', 'G5']    # 주 그룹 레이블
category1_means = np.array([20, 35, 30, 35, 27]) # 첫 번째 하위 카테고리 값
category2_means = np.array([25, 32, 34, 20, 25]) # 두 번째 하위 카테고리 값
category3_means = np.array([15, 20, 25, 18, 22]) # 세 번째 하위 카테고리 값

# 새로운 그림과 축 생성
plt.figure(figsize=(10, 6))
# 첫 번째 카테고리 막대 (바닥부터 시작)
plt.bar(labels, category1_means, label='Category 1', color='skyblue')
# 두 번째 카테고리 막대 (첫 번째 카테고리 막대 위부터 시작)
plt.bar(labels, category2_means, bottom=category1_means, label='Category 2', color='salmon')
# 세 번째 카테고리 막대 (첫 번째 + 두 번째 카테고리 막대 위부터 시작)
plt.bar(labels, category3_means, bottom=category1_means + category2_means, label='Category 3', color='lightgreen')

# y축 레이블, 차트 제목, 범례 설정
plt.ylabel('Total Value')
plt.xlabel('Group')
plt.title('Matplotlib Stacked Bar Chart')
plt.legend(loc='upper right')

# st.pyplot(plt.gcf())
# plt.clf()
"""
st.code(code_stacked_bar_mpl_comp, language='python')
if st.checkbox("Matplotlib 누적 막대 그래프 예시 보기", key="stacked_bar_mpl_comp_page"):
    labels = ['Quarter 1', 'Quarter 2', 'Quarter 3', 'Quarter 4']
    product_A_sales = np.array([100, 150, 120, 180])
    product_B_sales = np.array([80, 90, 110, 130])
    product_C_sales = np.array([120, 100, 140, 100])

    fig, ax = plt.subplots(figsize=(10,6))
    ax.bar(labels, product_A_sales, label='Product A', color='skyblue')
    ax.bar(labels, product_B_sales, bottom=product_A_sales, label='Product B', color='salmon')
    ax.bar(labels, product_C_sales, bottom=product_A_sales + product_B_sales, label='Product C', color='lightgreen')
    ax.set_ylabel('Total Sales')
    ax.set_xlabel('Quarter')
    ax.set_title('Matplotlib Stacked Bar Chart (Quarterly Sales by Product)')
    ax.legend(loc='upper left')
    st.pyplot(fig)


# Pandas Plotting 예시 (Matplotlib 기반)
st.markdown("#### Pandas `plot(kind='bar', stacked=True)`")
code_stacked_bar_pd_comp = """
# pandas를 pd라는 별칭으로 가져옵니다.
import pandas as pd
# matplotlib.pyplot을 plt라는 별칭으로 가져옵니다. (차트 제목 등 추가 설정)
import matplotlib.pyplot as plt
# numpy를 np라는 별칭으로 가져옵니다.
import numpy as np

# 샘플 데이터 (wide-form 데이터가 Pandas plotting에 적합)
data = {
    'Product A': np.array([100, 150, 120, 180]),
    'Product B': np.array([80, 90, 110, 130]),
    'Product C': np.array([120, 100, 140, 100])
}
index_labels = ['Quarter 1', 'Quarter 2', 'Quarter 3', 'Quarter 4']
df_wide = pd.DataFrame(data, index=index_labels)

# Pandas DataFrame의 plot 메소드를 사용하여 누적 막대 그래프를 그립니다.
# kind='bar': 막대 그래프 종류
# stacked=True: 누적 형태로 표시
# figsize: 그림 크기
# colormap: 사용할 색상 맵
ax = df_wide.plot(kind='bar', stacked=True, figsize=(10, 6), colormap='Spectral')
# 차트 제목, 축 레이블 설정
plt.title('Pandas Stacked Bar Chart')
plt.ylabel('Total Sales')
plt.xlabel('Quarter')
# x축 눈금 회전 (선택 사항)
plt.xticks(rotation=0) # 회전 없음
# 범례 위치 조정
plt.legend(title='Product', loc='upper left')
# 레이아웃 자동 조정
plt.tight_layout()

# st.pyplot(plt.gcf())
# plt.clf() # DataFrame.plot()이 Figure 객체를 반환하므로, 해당 객체로 관리하거나 plt.clf() 사용
"""
st.code(code_stacked_bar_pd_comp, language='python')
if st.checkbox("Pandas 누적 막대 그래프 예시 보기", key="stacked_bar_pd_comp_page"):
    data = {
        'Product A': np.array([100, 150, 120, 180]),
        'Product B': np.array([80, 90, 110, 130]),
        'Product C': np.array([120, 100, 140, 100])
    }
    index_labels = ['Quarter 1', 'Quarter 2', 'Quarter 3', 'Quarter 4']
    df_wide = pd.DataFrame(data, index=index_labels)

    fig, ax = plt.subplots(figsize=(10,6)) # Figure와 Axes를 먼저 생성
    df_wide.plot(kind='bar', stacked=True, colormap='Spectral', ax=ax) # 생성된 Axes에 그림
    ax.set_title('Pandas Stacked Bar Chart (Quarterly Sales)')
    ax.set_ylabel('Total Sales')
    ax.set_xlabel('Quarter')
    ax.tick_params(axis='x', rotation=0) # x축 눈금 회전 없음
    ax.legend(title='Product', loc='best')
    plt.tight_layout()
    st.pyplot(fig)


# Plotly Express 예시
st.markdown("#### Plotly Express 누적 막대 그래프")
code_stacked_bar_plotly_comp = """
# Plotly Express 라이브러리를 px라는 별칭으로 사용합니다.
import plotly.express as px
# pandas로 데이터프레임을 다루기 위해 import 합니다.
import pandas as pd

# 샘플 데이터 (long-form 데이터가 Plotly Express에 적합)
data = {
    'Quarter': ['Q1', 'Q1', 'Q1', 'Q2', 'Q2', 'Q2', 'Q3', 'Q3', 'Q3', 'Q4', 'Q4', 'Q4'],
    'Product': ['A', 'B', 'C', 'A', 'B', 'C', 'A', 'B', 'C', 'A', 'B', 'C'],
    'Sales': [100, 80, 120, 150, 90, 100, 120, 110, 140, 180, 130, 100]
}
df_long_stacked = pd.DataFrame(data)

# Plotly Express를 사용하여 누적 막대 그래프를 생성합니다.
# x, y: x축, y축으로 사용할 컬럼
# color: 누적될 하위 그룹을 나타내는 컬럼
# barmode='stack'는 기본값이지만 명시적으로 사용 가능
fig = px.bar(
    df_long_stacked,
    x='Quarter',
    y='Sales',
    color='Product',       # 'Product' 별로 누적
    title="Plotly Express 누적 막대 그래프",
    labels={'Sales': '총 판매량', 'Quarter': '분기', 'Product': '제품군'},
    # text_auto='.2s'      # 막대 위에 값 표시 (선택 사항, 너무 많은 값은 가독성 저해)
    color_discrete_map={'A':'#636EFA', 'B':'#EF553B', 'C':'#00CC96'} # 색상 직접 지정
)
# 차트 레이아웃 업데이트 (선택 사항)
fig.update_layout(
    legend_title_text='제품군',
    xaxis_categoryorder='array', # x축 순서 지정 (리스트 전달)
    xaxis_categoryarray=['Q1', 'Q2', 'Q3', 'Q4']
)

# st.plotly_chart(fig, use_container_width=True)
"""
st.code(code_stacked_bar_plotly_comp, language='python')
if st.checkbox("Plotly 누적 막대 그래프 예시 보기", key="stacked_bar_plotly_comp_page"):
    data_long = {
        'Quarter': ['Q1', 'Q1', 'Q1', 'Q2', 'Q2', 'Q2', 'Q3', 'Q3', 'Q3', 'Q4', 'Q4', 'Q4'],
        'Product': ['A', 'B', 'C', 'A', 'B', 'C', 'A', 'B', 'C', 'A', 'B', 'C'],
        'Sales': [100, 80, 120, 150, 90, 100, 120, 110, 140, 180, 130, 100]
    }
    df_long_stacked = pd.DataFrame(data_long)
    fig = px.bar(
        df_long_stacked, x='Quarter', y='Sales', color='Product',
        title="Plotly Express Stacked Bar Chart (Quarterly Sales by Product)",
        labels={'Sales': '총 판매량', 'Quarter': '분기', 'Product': '제품'},
        text_auto=True, # 각 세그먼트에 값 표시
        color_discrete_sequence=px.colors.carto.Pastel # 색상 팔레트
    )
    # x축 순서 고정
    fig.update_layout(
        xaxis_categoryorder='array',
        xaxis_categoryarray=['Q1', 'Q2', 'Q3', 'Q4'],
        legend_title_text='제품'
    )
    # 텍스트 폰트 크기 및 색상 조정 (선택 사항)
    fig.update_traces(textfont_size=10, textangle=0, textposition="inside", insidetextanchor='middle')
    st.plotly_chart(fig, use_container_width=True)