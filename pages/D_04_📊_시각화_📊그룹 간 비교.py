# pages/4_📊_그룹_간_비교.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from utils.utils import get_sample_data

st.header("4. 여러 그룹 간 비교")
st.markdown("""
여러 그룹(범주)에 따른 수치형 데이터의 차이를 비교할 때 사용합니다.
""")

# --- 4.1 그룹화된 막대 그래프 (Grouped Bar Plot) ---
st.subheader("4.1 그룹화된 막대 그래프 (Grouped Bar Plot)")
st.markdown("""
여러 그룹에 대해 각 그룹 내의 하위 범주별 값을 비교할 때 유용합니다.
- **주요 사용 라이브러리:** `matplotlib.pyplot.bar()` (수동 위치 조정 필요), `seaborn.barplot()` (hue 옵션 사용), `plotly.express.bar()` (barmode='group')
- **언제 사용하나요?** 두 개의 범주형 변수에 따른 수치형 변수의 값을 비교할 때.
""")

# Matplotlib 예시
st.markdown("#### Matplotlib 그룹화된 막대 그래프")
code_grouped_bar_mpl_comp = """
# matplotlib.pyplot을 plt라는 별칭으로 가져옵니다.
import matplotlib.pyplot as plt
# numpy를 np라는 별칭으로 가져옵니다.
import numpy as np

# 샘플 데이터
labels = ['G1', 'G2', 'G3', 'G4', 'G5'] # 주 그룹 레이블
men_means = [20, 34, 30, 35, 27]     # 남성 그룹 평균값
women_means = [25, 32, 34, 20, 25]   # 여성 그룹 평균값

x = np.arange(len(labels))  # 각 레이블의 위치
width = 0.35  # 막대의 너비

# 새로운 그림과 축 생성
fig, ax = plt.subplots(figsize=(10, 6))
# 남성 그룹 막대 생성
rects1 = ax.bar(x - width/2, men_means, width, label='Men', color='cornflowerblue')
# 여성 그룹 막대 생성
rects2 = ax.bar(x + width/2, women_means, width, label='Women', color='lightcoral')

# y축 레이블, 차트 제목, x축 눈금 레이블, 범례 설정
ax.set_ylabel('Scores')
ax.set_title('Scores by group and gender (Matplotlib)')
ax.set_xticks(x) # x축 눈금 위치 설정
ax.set_xticklabels(labels) # x축 눈금 레이블 설정
ax.legend() # 범례 표시

# 각 막대 위에 값 표시 (선택 사항)
ax.bar_label(rects1, padding=3, fontsize=8)
ax.bar_label(rects2, padding=3, fontsize=8)

# 레이아웃 자동 조정
fig.tight_layout()

# st.pyplot(fig)
# plt.clf()
"""
st.code(code_grouped_bar_mpl_comp, language='python')
if st.checkbox("Matplotlib 그룹화된 막대 그래프 예시 보기", key="grouped_bar_mpl_comp_page"):
    labels = ['Q1', 'Q2', 'Q3', 'Q4']
    product_A_sales = [150, 180, 220, 200]
    product_B_sales = [120, 160, 190, 170]
    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10,6))
    rects1 = ax.bar(x - width/2, product_A_sales, width, label='Product A', color='cornflowerblue')
    rects2 = ax.bar(x + width/2, product_B_sales, width, label='Product B', color='lightcoral')
    ax.set_ylabel('Sales')
    ax.set_title('Quarterly Sales by Product (Matplotlib)')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.bar_label(rects1, padding=3, fontsize=8)
    ax.bar_label(rects2, padding=3, fontsize=8)
    fig.tight_layout()
    st.pyplot(fig)

# Seaborn 예시
st.markdown("#### Seaborn 그룹화된 막대 그래프")
code_grouped_bar_sns_comp = """
# seaborn을 sns라는 별칭으로 가져옵니다.
import seaborn as sns
# matplotlib.pyplot을 plt라는 별칭으로 가져옵니다.
import matplotlib.pyplot as plt
# pandas를 pd라는 별칭으로 가져옵니다.
import pandas as pd

# 샘플 데이터 (long-form 데이터가 Seaborn에 적합)
data = {
    'Quarter': ['Q1', 'Q1', 'Q2', 'Q2', 'Q3', 'Q3', 'Q4', 'Q4'],
    'Product': ['A', 'B', 'A', 'B', 'A', 'B', 'A', 'B'],
    'Sales': [150, 120, 180, 160, 220, 190, 200, 170]
}
df_sales = pd.DataFrame(data)

# 새로운 그림과 축 생성
plt.figure(figsize=(10, 6))
# Seaborn으로 그룹화된 막대 그래프를 그립니다.
# x, y: x축, y축으로 사용할 데이터프레임의 컬럼 이름
# hue: 막대를 그룹화할 기준이 되는 컬럼
# data: 사용할 데이터프레임
# palette: 색상 팔레트
sns.barplot(x='Quarter', y='Sales', hue='Product', data=df_sales, palette='pastel', dodge=True) # dodge=True가 그룹화
# 차트 제목 설정
plt.title('Seaborn Grouped Bar Plot')
# y축 레이블 설정
plt.ylabel('Sales')
# x축 레이블 설정
plt.xlabel('Quarter')
# 범례 제목 변경 (선택 사항)
plt.legend(title='Product Type')

# st.pyplot(plt.gcf())
# plt.clf()
"""
st.code(code_grouped_bar_sns_comp, language='python')
if st.checkbox("Seaborn 그룹화된 막대 그래프 예시 보기", key="grouped_bar_sns_comp_page"):
    # 샘플 데이터 (utils.py의 get_sample_data 활용)
    # get_sample_data('mixed')는 Group(Alpha,Beta,Gamma), Metric1, Metric2 컬럼을 가짐
    # 여기서는 Group을 주 그룹, Metric 종류를 서브 그룹으로 가정하고 데이터 변형
    sample_df = get_sample_data('mixed')
    # Metric1과 Metric2를 long-form으로 변환
    plot_df = sample_df.melt(id_vars='Group', value_vars=['Metric1', 'Metric2'], var_name='Metric_Type', value_name='Value')

    fig, ax = plt.subplots(figsize=(10,6))
    sns.barplot(x='Group', y='Value', hue='Metric_Type', data=plot_df, palette='Set2', ax=ax)
    ax.set_title('Seaborn Grouped Bar Plot (Metrics by Group)')
    ax.set_ylabel('Average Value')
    ax.set_xlabel('Main Group')
    ax.legend(title='Metric Type')
    st.pyplot(fig)

# Plotly Express 예시
st.markdown("#### Plotly Express 그룹화된 막대 그래프")
code_grouped_bar_plotly_comp = """
# Plotly Express 라이브러리를 px라는 별칭으로 사용합니다.
import plotly.express as px
# pandas로 데이터프레임을 다루기 위해 import 합니다.
import pandas as pd

# 샘플 데이터 (long-form 데이터가 Plotly에 적합)
data = {
    'Quarter': ['Q1', 'Q1', 'Q2', 'Q2', 'Q3', 'Q3', 'Q4', 'Q4'],
    'Product': ['A', 'B', 'A', 'B', 'A', 'B', 'A', 'B'],
    'Sales': [150, 120, 180, 160, 220, 190, 200, 170]
}
df_sales = pd.DataFrame(data)

# Plotly Express를 사용하여 그룹화된 막대 그래프를 생성합니다.
fig = px.bar(
    df_sales,              # 사용할 데이터프레임 (long-form)
    x='Quarter',           # x축으로 사용할 컬럼 (주 그룹)
    y='Sales',             # y축으로 사용할 컬럼 (값)
    color='Product',       # 막대의 색상을 'Product' 값에 따라 다르게 지정 (그룹화 역할)
    barmode='group',       # 막대 그룹화 모드 ('group', 'stack', 'overlay')
    title="Plotly Express 그룹화된 막대 그래프", # 차트 제목
    labels={'Sales': '판매량', 'Quarter': '분기', 'Product': '제품'}, # 레이블 변경
    text_auto=True         # 각 막대 위에 값을 자동으로 표시
)
# 차트 레이아웃 업데이트 (선택 사항)
fig.update_layout(
    legend_title_text='제품 종류', # 범례 제목
    bargap=0.2,           # 그룹 내 막대 간 간격
    bargroupgap=0.1       # 그룹 간 간격
)

# st.plotly_chart(fig, use_container_width=True)
"""
st.code(code_grouped_bar_plotly_comp, language='python')
if st.checkbox("Plotly 그룹화된 막대 그래프 예시 보기", key="grouped_bar_plotly_comp_page"):
    sample_df = get_sample_data('mixed')
    plot_df = sample_df.melt(id_vars='Group', value_vars=['Metric1', 'Metric2'], var_name='Metric_Type', value_name='Value')
    # Plotly는 그룹별 평균을 직접 계산하지 않으므로, 필요시 집계
    # 여기서는 barplot이 기본적으로 값의 합계를 표시하거나, 데이터 그대로 표시. 평균은 별도 계산 후 사용.
    # 예시에서는 집계 없이 각 row를 그리는 것보다, 그룹별 평균을 사용하는 것이 일반적.
    # 여기서는 get_sample_data가 이미 집계되지 않은 데이터이므로, 집계가 필요.
    # 하지만 barplot은 자동집계 기능이 없으므로, 데이터를 그대로 사용하거나, 집계된 데이터를 넣어야 함.
    # 현재 plot_df는 이미 long form이므로, 각 Group, Metric_Type 조합의 Value를 사용. (만약 평균을 원하면 groupby 후 사용)
    # 여기서는 설명을 위해, 각 Group/Metric_Type 조합의 Value를 그대로 사용 (값이 여러 개면 중첩될 수 있음. 이 경우 집계 필요)
    # 더 정확한 비교를 위해서는 groupby로 평균/합계 계산 후 사용해야 함.
    # 예시 단순화를 위해, 제공된 데이터가 이미 적절히 집계되었다고 가정.
    # 실제로는 다음과 같이 집계: agg_df = plot_df.groupby(['Group', 'Metric_Type'])['Value'].mean().reset_index()

    fig = px.bar(
        plot_df, # 실제로는 집계된 데이터를 사용하는 것이 좋음. agg_df 사용.
        x='Group', y='Value', color='Metric_Type', barmode='group',
        title="Plotly Express 그룹화된 막대 그래프 (Metrics by Group)",
        labels={'Value': '평균 값 (집계 가정)', 'Group': '주 그룹', 'Metric_Type': '메트릭 종류'},
        text_auto='.2s' # 값 표시 형식 (예: 1.2k, 3.4M)
    )
    fig.update_layout(legend_title_text='메트릭 종류', bargap=0.15, bargroupgap=0.1)
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# --- 4.2 그룹별 박스 플롯 (Box Plot by Group) ---
st.subheader("4.2 그룹별 박스 플롯 (Box Plot by Group)")
st.markdown("""
여러 그룹(범주형 변수)에 따른 수치형 변수의 분포를 박스 플롯으로 비교합니다.
(이 내용은 "1. 데이터 분포 확인"의 박스 플롯과 유사하나, 그룹 간 비교에 초점을 맞춥니다.)
- **주요 사용 라이브러리:** `matplotlib.pyplot.boxplot()` (데이터 수동 준비), `seaborn.boxplot()` (x, y 지정), `plotly.express.box()` (x, y 지정)
- **언제 사용하나요?** 각 그룹별 데이터의 중앙값, 사분위 범위, 이상치 등을 비교하여 분포 차이를 확인할 때.
""")

# Matplotlib 예시
st.markdown("#### Matplotlib 그룹별 박스 플롯")
code_group_box_mpl_comp = """
# matplotlib.pyplot을 plt라는 별칭으로 가져옵니다.
import matplotlib.pyplot as plt
# numpy를 np라는 별칭으로 가져옵니다.
import numpy as np
# pandas를 pd라는 별칭으로 가져옵니다.
import pandas as pd

# 샘플 데이터 생성 (여러 그룹에 대한 데이터)
np.random.seed(123)
groups = ['A', 'B', 'C', 'D']
data_by_group = {group: np.random.normal(loc=i*2, scale=1.5, size=50) for i, group in enumerate(groups)}
# Matplotlib boxplot은 리스트의 리스트 또는 값들의 리스트를 입력으로 받음
data_list = [data_by_group[group] for group in groups]

# 새로운 그림과 축 생성
plt.figure(figsize=(10, 6))
# 그룹별 박스 플롯을 그립니다.
bp = plt.boxplot(data_list, labels=groups, patch_artist=True, medianprops={'color':'black'})
# 각 박스에 색상 적용
colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightsalmon']
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
# 차트 제목 설정
plt.title('Matplotlib Box Plot by Group')
# y축 레이블 설정
plt.ylabel('Value')
# x축 레이블 설정
plt.xlabel('Group')

# st.pyplot(plt.gcf())
# plt.clf()
"""
st.code(code_group_box_mpl_comp, language='python')
if st.checkbox("Matplotlib 그룹별 박스 플롯 예시 보기", key="group_box_mpl_comp_page"):
    sample_df = get_sample_data('mixed') # Group, Metric1, Metric2
    # Metric1에 대해 그룹별로 데이터를 리스트로 준비
    groups = sample_df['Group'].unique()
    data_to_plot = [sample_df[sample_df['Group'] == g]['Metric1'] for g in groups]

    fig, ax = plt.subplots(figsize=(10,6))
    bp = ax.boxplot(data_to_plot, labels=groups, patch_artist=True, medianprops=dict(color='black'))
    colors = plt.cm.Pastel1(np.linspace(0, 1, len(groups))) # 그룹 수에 맞춰 색상 자동 생성
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    ax.set_title('Matplotlib Box Plot (Metric1 by Group)')
    ax.set_ylabel('Metric1 Value')
    ax.set_xlabel('Group')
    st.pyplot(fig)

# Seaborn 예시
st.markdown("#### Seaborn 그룹별 박스 플롯")
code_group_box_sns_comp = """
# seaborn을 sns라는 별칭으로 가져옵니다.
import seaborn as sns
# matplotlib.pyplot을 plt라는 별칭으로 가져옵니다.
import matplotlib.pyplot as plt
# pandas를 pd라는 별칭으로 가져옵니다.
import pandas as pd
# numpy를 np라는 별칭으로 가져옵니다.
import numpy as np

# 샘플 데이터 생성 (long-form 데이터가 Seaborn에 적합)
np.random.seed(123)
n_obs_per_group = 50
df_list = []
for i, group_name in enumerate(['Alpha', 'Beta', 'Gamma', 'Delta']):
    df_list.append(pd.DataFrame({
        'Group': group_name,
        'Value': np.random.normal(loc=i*2, scale=1.5, size=n_obs_per_group)
    }))
df_long = pd.concat(df_list, ignore_index=True)

# 새로운 그림과 축 생성
plt.figure(figsize=(10, 6))
# Seaborn으로 그룹별 박스 플롯을 그립니다.
# x: 그룹을 나타내는 컬럼
# y: 값을 나타내는 컬럼
# data: 사용할 데이터프레임
# palette: 색상 팔레트
sns.boxplot(x='Group', y='Value', data=df_long, palette='Spectral')
# 차트 제목 설정
plt.title('Seaborn Box Plot by Group')
# y축 레이블 설정
plt.ylabel('Value')
# x축 레이블 설정
plt.xlabel('Group')

# st.pyplot(plt.gcf())
# plt.clf()
"""
st.code(code_group_box_sns_comp, language='python')
if st.checkbox("Seaborn 그룹별 박스 플롯 예시 보기", key="group_box_sns_comp_page"):
    sample_df = get_sample_data('mixed') # Group, Metric1
    fig, ax = plt.subplots(figsize=(10,6))
    sns.boxplot(x='Group', y='Metric1', data=sample_df, palette='Spectral', ax=ax)
    ax.set_title('Seaborn Box Plot (Metric1 by Group)')
    ax.set_ylabel('Metric1 Value')
    ax.set_xlabel('Group')
    st.pyplot(fig)

# Plotly Express 예시
st.markdown("#### Plotly Express 그룹별 박스 플롯")
code_group_box_plotly_comp = """
# Plotly Express 라이브러리를 px라는 별칭으로 사용합니다.
import plotly.express as px
# pandas로 데이터프레임을 다루기 위해 import 합니다.
import pandas as pd
# numpy로 숫자 데이터를 생성/조작하기 위해 import 합니다.
import numpy as np

# 샘플 데이터 생성 (long-form 데이터가 Plotly에 적합)
np.random.seed(123)
n_obs_per_group = 50
df_list = []
for i, group_name in enumerate(['Alpha', 'Beta', 'Gamma', 'Delta']):
    df_list.append(pd.DataFrame({
        'Group': group_name,
        'Value': np.random.normal(loc=i*2, scale=1.5, size=n_obs_per_group)
    }))
df_long = pd.concat(df_list, ignore_index=True)

# Plotly Express를 사용하여 그룹별 박스 플롯을 생성합니다.
fig = px.box(
    df_long,               # 사용할 데이터프레임 (long-form)
    x='Group',             # x축으로 사용할 컬럼 (그룹)
    y='Value',             # y축으로 사용할 컬럼 (값)
    color='Group',         # 각 박스의 색상을 'Group' 값에 따라 다르게 지정
    points='all',          # 모든 데이터 포인트 표시 (옵션: "outliers", False)
    notched=True,          # 중앙값 신뢰구간을 노치 형태로 표시 (선택 사항)
    title="Plotly Express 그룹별 박스 플롯", # 차트 제목
    labels={'Value': '측정값', 'Group': '그룹 구분'} # 레이블 변경
)
# 차트 레이아웃 업데이트 (선택 사항)
fig.update_layout(legend_title_text='그룹')

# st.plotly_chart(fig, use_container_width=True)
"""
st.code(code_group_box_plotly_comp, language='python')
if st.checkbox("Plotly 그룹별 박스 플롯 예시 보기", key="group_box_plotly_comp_page"):
    sample_df = get_sample_data('mixed') # Group, Metric1
    fig = px.box(
        sample_df, x='Group', y='Metric1', color='Group',
        points='all', # 모든 점 표시
        notched=True, # 중앙값 신뢰구간 노치 표시
        title="Plotly Express Box Plot (Metric1 by Group)",
        labels={'Metric1': 'Metric1 값', 'Group': '그룹'}
    )
    fig.update_layout(legend_title_text='그룹')
    st.plotly_chart(fig, use_container_width=True)