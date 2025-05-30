# pages/2_🔗_변수_간_관계.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from utils.utils import get_sample_data
from pandas.plotting import scatter_matrix # Matplotlib 기반 페어플롯용

st.header("2. 변수 간 관계 파악 (두 변수 이상)")
st.markdown("""
두 개 이상의 변수들 사이에 어떤 관계가 있는지 (상관 관계, 패턴 등) 파악하고 싶을 때 사용합니다.
""")

# --- 2.1 산점도 (Scatter Plot) ---
st.subheader("2.1 산점도 (Scatter Plot)")
st.markdown("""
두 연속형 변수 간의 관계를 점으로 표현합니다. 점들의 분포를 통해 변수 간의 상관관계, 군집 등을 파악할 수 있습니다.
- **주요 사용 라이브러리:** `matplotlib.pyplot.scatter()`, `seaborn.scatterplot()`, `plotly.express.scatter()`
- **언제 사용하나요?** 두 수치형 변수 사이의 관계, 패턴, 이상치 등을 보고 싶을 때.
""")

# Matplotlib 예시
st.markdown("#### Matplotlib 산점도")
code_scatter_mpl_rel = """
# matplotlib.pyplot을 plt라는 별칭으로 가져옵니다.
import matplotlib.pyplot as plt
# numpy를 np라는 별칭으로 가져옵니다.
import numpy as np

# 샘플 데이터 생성
np.random.seed(42) # 재현성을 위한 시드 설정
x = np.random.rand(50) * 10  # 0과 10 사이의 난수 50개
y = 2 * x + 5 + np.random.randn(50) * 3 # y는 x와 선형 관계 + 노이즈

# 새로운 그림과 축 생성
plt.figure(figsize=(8, 5))
# 산점도를 그립니다.
# x: x축 데이터
# y: y축 데이터
# s: 점 크기
# c: 점 색상
# alpha: 점 투명도
plt.scatter(x, y, s=50, c='dodgerblue', alpha=0.7, edgecolors='w', linewidth=0.5)
# 차트 제목 설정
plt.title('Matplotlib Scatter Plot')
# x축 레이블 설정
plt.xlabel('X_variable')
# y축 레이블 설정
plt.ylabel('Y_variable')
# 그리드 추가 (선택 사항)
plt.grid(True, linestyle='--', alpha=0.7)

# st.pyplot(plt.gcf())
# plt.clf()
"""
st.code(code_scatter_mpl_rel, language='python')
if st.checkbox("Matplotlib 산점도 예시 보기", key="scatter_mpl_rel_page"):
    sample_data_num_df = get_sample_data('numerical')
    fig, ax = plt.subplots(figsize=(8,5))
    ax.scatter(sample_data_num_df['A'], sample_data_num_df['B'], s=50, c='dodgerblue', alpha=0.7, edgecolors='w', linewidth=0.5)
    ax.set_title('Matplotlib Scatter Plot (A vs B)')
    ax.set_xlabel('Column A')
    ax.set_ylabel('Column B')
    ax.grid(True, linestyle='--', alpha=0.7)
    st.pyplot(fig)

# Seaborn 예시
st.markdown("#### Seaborn 산점도")
code_scatter_sns_rel = """
# seaborn을 sns라는 별칭으로 가져옵니다.
import seaborn as sns
# matplotlib.pyplot을 plt라는 별칭으로 가져옵니다.
import matplotlib.pyplot as plt
# pandas를 pd라는 별칭으로 가져옵니다.
import pandas as pd
# numpy를 np라는 별칭으로 가져옵니다.
import numpy as np

# 샘플 데이터 생성
np.random.seed(42)
x_vals = np.random.rand(50) * 10
y_vals = 2 * x_vals + 5 + np.random.randn(50) * 3
category = np.random.choice(['Type1', 'Type2'], 50)
size_vals = np.random.rand(50) * 100
df = pd.DataFrame({'X_Val': x_vals, 'Y_Val': y_vals, 'Category': category, 'Size': size_vals})

# 새로운 그림과 축 생성
plt.figure(figsize=(9, 6))
# Seaborn으로 산점도를 그립니다.
# x, y: x축, y축으로 사용할 데이터프레임의 컬럼 이름
# data: 사용할 데이터프레임
# hue: 점의 색상을 구분할 카테고리형 컬럼
# size: 점의 크기를 구분할 수치형 컬럼
# style: 점의 모양을 구분할 카테고리형 컬럼 (선택 사항)
# palette: 색상 팔레트
# alpha: 투명도
sns.scatterplot(x='X_Val', y='Y_Val', hue='Category', size='Size', style='Category', data=df, palette='Set1', alpha=0.8, sizes=(20, 200))
# 차트 제목 설정
plt.title('Seaborn Scatter Plot')
# x축 레이블 설정
plt.xlabel('X_variable')
# y축 레이블 설정
plt.ylabel('Y_variable')
# 범례 위치 조정 (선택 사항)
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))

# st.pyplot(plt.gcf())
# plt.clf() # plt.tight_layout() 대신에, 혹은 함께 사용하여 레이아웃 정리
"""
st.code(code_scatter_sns_rel, language='python')
if st.checkbox("Seaborn 산점도 예시 보기", key="scatter_sns_rel_page"):
    sample_data_num_df = get_sample_data('numerical')
    sample_data_cat_df = get_sample_data('categorical') # 추가 정보용
    # 두 데이터프레임 합치기 (인덱스 기준, 실제 사용 시엔 적절한 병합 필요)
    plot_df = pd.concat([sample_data_num_df, sample_data_cat_df['Category']], axis=1)

    fig, ax = plt.subplots(figsize=(9,6))
    sns.scatterplot(x='A', y='B', hue='Category', size='C', style='Category', data=plot_df, palette='Set2', ax=ax, alpha=0.8, sizes=(30,300))
    ax.set_title('Seaborn Scatter Plot (A vs B, Hue/Style by Category, Size by C)')
    ax.set_xlabel('Column A')
    ax.set_ylabel('Column B')
    ax.legend(title='Category & C Size', loc='upper left', bbox_to_anchor=(1,1)) # 범례 위치 및 제목
    plt.tight_layout() # 레이아웃 자동 조정
    st.pyplot(fig)

# Plotly Express 예시
st.markdown("#### Plotly Express 산점도")
code_scatter_plotly_rel = """
# Plotly Express 라이브러리를 px라는 별칭으로 사용합니다.
import plotly.express as px
# pandas로 데이터프레임을 다루기 위해 import 합니다.
import pandas as pd
# numpy로 숫자 데이터를 생성/조작하기 위해 import 합니다.
import numpy as np

# 샘플 데이터 생성
np.random.seed(42)
x_vals = np.random.rand(50) * 10
y_vals = 2 * x_vals + 5 + np.random.randn(50) * 3
category_vals = np.random.choice(['Alpha', 'Beta'], 50)
size_vals = np.random.rand(50) * 20 # 점 크기 조정을 위한 값
df = pd.DataFrame({'X_Val': x_vals, 'Y_Val': y_vals, 'Group': category_vals, 'PointSize': size_vals})

# Plotly Express를 사용하여 산점도를 생성합니다.
fig = px.scatter(
    df,                    # 사용할 데이터프레임
    x="X_Val",             # x축으로 사용할 컬럼
    y="Y_Val",             # y축으로 사용할 컬럼
    color="Group",         # 점의 색상을 'Group' 값에 따라 다르게 지정
    size="PointSize",      # 점의 크기를 'PointSize' 값에 따라 다르게 지정
    symbol="Group",        # 점의 모양을 'Group' 값에 따라 다르게 지정 (선택 사항)
    hover_name="Group",    # 마우스 오버 시 표시될 이름
    title="Plotly Express 산점도", # 차트 제목
    labels={'X_Val': 'X축 값', 'Y_Val': 'Y축 값', 'Group': '그룹 구분'} # 축 및 범례 레이블 변경
)
# 차트 레이아웃 업데이트 (선택 사항)
fig.update_traces(marker=dict(line=dict(width=0.5, color='DarkSlateGrey'))) # 마커 테두리 추가
fig.update_layout(legend_title_text='범례 제목') # 범례 제목 변경

# st.plotly_chart(fig, use_container_width=True)
"""
st.code(code_scatter_plotly_rel, language='python')
if st.checkbox("Plotly 산점도 예시 보기", key="scatter_plotly_rel_page"):
    sample_data_num_df = get_sample_data('numerical')
    sample_data_cat_df = get_sample_data('categorical')
    plot_df = pd.concat([sample_data_num_df, sample_data_cat_df['Category']], axis=1)

    fig = px.scatter(
        plot_df, x="A", y="B", color="Category", size="C",
        symbol="Category", # 모양도 카테고리별로 다르게
        title="Plotly Express 산점도 (A vs B)",
        labels={'A': '컬럼 A 값', 'B': '컬럼 B 값', 'Category': '범주', 'C': '컬럼 C (크기)'},
        hover_data={'A':':.2f', 'B':':.2f', 'C':':.2f', 'Category':True} # 마우스오버 시 표시될 정보 및 형식
    )
    fig.update_traces(marker=dict(line=dict(width=0.5, color='DarkSlateGrey'))) # 마커 테두리
    fig.update_layout(legend_title_text='범주 / 크기(C)')
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# --- 2.2 라인 플롯 (Line Plot) ---
st.subheader("2.2 라인 플롯 (Line Plot)")
st.markdown("""
주로 시간에 따른 연속형 변수의 변화 추세를 보거나, 순서가 있는 데이터 포인트들을 연결하여 패턴을 파악할 때 사용합니다.
- **주요 사용 라이브러리:** `matplotlib.pyplot.plot()`, `seaborn.lineplot()`, `plotly.express.line()`
- **언제 사용하나요?** 시계열 데이터, 순서형 데이터의 추세 파악. 두 변수 간 관계가 순차적일 때.
""")

# Matplotlib 예시
st.markdown("#### Matplotlib 라인 플롯")
code_line_mpl_rel = """
# matplotlib.pyplot을 plt라는 별칭으로 가져옵니다.
import matplotlib.pyplot as plt
# numpy를 np라는 별칭으로 가져옵니다.
import numpy as np

# 샘플 데이터 생성
x = np.linspace(0, 10, 100) # 0부터 10까지 100개의 점 생성 (x축)
y1 = np.sin(x)              # sin(x) (첫 번째 y축 데이터)
y2 = np.cos(x)              # cos(x) (두 번째 y축 데이터)

# 새로운 그림과 축 생성
plt.figure(figsize=(10, 5))
# 첫 번째 라인 플롯
plt.plot(x, y1, label='sin(x)', color='blue', linestyle='-', marker='o', markersize=3)
# 두 번째 라인 플롯
plt.plot(x, y2, label='cos(x)', color='red', linestyle='--', marker='x', markersize=5)
# 차트 제목 설정
plt.title('Matplotlib Line Plot')
# x축 레이블 설정
plt.xlabel('X_value')
# y축 레이블 설정
plt.ylabel('Y_value')
# 범례 표시
plt.legend()
# 그리드 추가
plt.grid(True, linestyle=':', alpha=0.6)

# st.pyplot(plt.gcf())
# plt.clf()
"""
st.code(code_line_mpl_rel, language='python')
if st.checkbox("Matplotlib 라인 플롯 예시 보기", key="line_mpl_rel_page"):
    sample_data_ts_df = get_sample_data('timeseries')
    fig, ax = plt.subplots(figsize=(10,5))
    ax.plot(sample_data_ts_df.index, sample_data_ts_df['StockA'], label='StockA', color='blue', linestyle='-', marker='.', markersize=4)
    ax.plot(sample_data_ts_df.index, sample_data_ts_df['StockB'], label='StockB', color='red', linestyle='--', marker='.', markersize=4)
    ax.set_title('Matplotlib Line Plot (Time Series)')
    ax.set_xlabel('Date')
    ax.set_ylabel('Stock Price')
    ax.legend()
    ax.grid(True, linestyle=':', alpha=0.6)
    fig.autofmt_xdate() # x축 날짜 레이블 자동 포맷팅
    st.pyplot(fig)

# Seaborn 예시
st.markdown("#### Seaborn 라인 플롯")
code_line_sns_rel = """
# seaborn을 sns라는 별칭으로 가져옵니다.
import seaborn as sns
# matplotlib.pyplot을 plt라는 별칭으로 가져옵니다.
import matplotlib.pyplot as plt
# pandas를 pd라는 별칭으로 가져옵니다.
import pandas as pd
# numpy를 np라는 별칭으로 가져옵니다.
import numpy as np

# 샘플 데이터 생성 (시계열 데이터 형태)
dates = pd.to_datetime(pd.date_range('2023-01-01', periods=100))
data_A = np.random.randn(100).cumsum() + 50
data_B = np.random.randn(100).cumsum() + 60
df_line = pd.DataFrame({'Date': dates, 'Value_A': data_A, 'Value_B': data_B})
# Seaborn lineplot은 long-form 데이터를 선호합니다.
df_melted = df_line.melt(id_vars='Date', var_name='Series', value_name='Value')


# 새로운 그림과 축 생성
plt.figure(figsize=(10, 5))
# Seaborn으로 라인 플롯을 그립니다.
# x, y: x축, y축으로 사용할 데이터프레임의 컬럼 이름
# data: 사용할 데이터프레임 (long-form 권장)
# hue: 라인을 구분할 카테고리형 컬럼
# style: 라인 스타일을 구분할 카테고리형 컬럼 (선택 사항)
# markers: 데이터 포인트에 마커 표시 여부
sns.lineplot(x='Date', y='Value', hue='Series', style='Series', data=df_melted, markers=True, dashes=False)
# 차트 제목 설정
plt.title('Seaborn Line Plot')
# x축 레이블 설정
plt.xlabel('Date')
# y축 레이블 설정
plt.ylabel('Value')
# x축 날짜 레이블 회전
plt.xticks(rotation=45)
# 레이아웃 자동 조정
plt.tight_layout()

# st.pyplot(plt.gcf())
# plt.clf()
"""
st.code(code_line_sns_rel, language='python')
if st.checkbox("Seaborn 라인 플롯 예시 보기", key="line_sns_rel_page"):
    sample_data_ts_df = get_sample_data('timeseries').reset_index()
    sample_data_ts_df = sample_data_ts_df.rename(columns={'index': 'Date'})
    df_melted = sample_data_ts_df.melt(id_vars='Date', var_name='Stock_Symbol', value_name='Price')

    fig, ax = plt.subplots(figsize=(10,5))
    sns.lineplot(x='Date', y='Price', hue='Stock_Symbol', style='Stock_Symbol', data=df_melted, ax=ax, markers=True, dashes=False)
    ax.set_title('Seaborn Line Plot (Time Series)')
    ax.set_xlabel('Date')
    ax.set_ylabel('Stock Price')
    plt.xticks(rotation=30, ha='right') # x축 눈금 회전 및 정렬
    plt.tight_layout()
    st.pyplot(fig)

# Plotly Express 예시
st.markdown("#### Plotly Express 라인 플롯")
code_line_plotly_rel = """
# Plotly Express 라이브러리를 px라는 별칭으로 사용합니다.
import plotly.express as px
# pandas로 데이터프레임과 시계열 데이터를 다루기 위해 import 합니다.
import pandas as pd
# numpy로 숫자 데이터를 생성/조작하기 위해 import 합니다.
import numpy as np

# 샘플 데이터 (시계열) 생성
dates = pd.to_datetime(pd.date_range('2023-01-01', periods=100))
data_A = np.random.randn(100).cumsum() + 50
data_B = np.random.randn(100).cumsum() + 60
df_line = pd.DataFrame({'Date': dates, 'Value_A': data_A, 'Value_B': data_B})
# Plotly Express도 long-form 데이터를 선호합니다.
df_melted = df_line.melt(id_vars='Date', var_name='Series_Name', value_name='Value_Data')

# Plotly Express를 사용하여 라인 플롯을 생성합니다.
fig = px.line(
    df_melted,             # 사용할 데이터프레임 (long-form)
    x='Date',              # x축으로 사용할 컬럼 (날짜)
    y='Value_Data',        # y축으로 사용할 컬럼 (값)
    color='Series_Name',   # 라인의 색상을 'Series_Name' 값에 따라 다르게 지정
    symbol='Series_Name',  # 각 데이터 포인트의 마커 모양을 'Series_Name' 값에 따라 다르게 지정
    title="Plotly Express 라인 플롯", # 차트 제목
    labels={'Value_Data': '측정값', 'Series_Name': '데이터 시리즈'} # 축 및 범례 레이블 변경
)
# 차트 레이아웃 업데이트 (선택 사항)
fig.update_traces(marker=dict(size=5)) # 마커 크기 일괄 조정
fig.update_layout(xaxis_title_text='날짜', yaxis_title_text='값')
# x축 날짜 형식 지정 (선택 사항)
# fig.update_xaxes(tickformat="%b %d, %Y") # 예: Jan 01, 2023

# st.plotly_chart(fig, use_container_width=True)
"""
st.code(code_line_plotly_rel, language='python')
if st.checkbox("Plotly 라인 플롯 예시 보기", key="line_plotly_rel_page"):
    sample_data_ts_df = get_sample_data('timeseries').reset_index()
    sample_data_ts_df = sample_data_ts_df.rename(columns={'index': 'Date'})
    df_melted = sample_data_ts_df.melt(id_vars='Date', var_name='Stock_Symbol', value_name='Price')

    fig = px.line(
        df_melted, x='Date', y='Price', color='Stock_Symbol', symbol='Stock_Symbol',
        title="Plotly Express 라인 플롯 (Time Series)",
        labels={'Price': '주가', 'Stock_Symbol': '주식 심볼', 'Date': '날짜'}
    )
    fig.update_traces(marker=dict(size=5), connectgaps=True) # connectgaps는 결측값이 있어도 라인 연결
    fig.update_layout(xaxis_title_text='날짜', yaxis_title_text='주가')
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# --- 2.3 히트맵 (Heatmap) ---
st.subheader("2.3 히트맵 (Heatmap - 상관계수 행렬 예시)")
st.markdown("""
숫자 데이터를 색상으로 표현하여 매트릭스 형태로 보여줍니다. 주로 변수 간 상관계수 행렬이나 혼동 행렬 등을 시각화할 때 사용됩니다.
- **주요 사용 라이브러리:** `matplotlib.pyplot.imshow()`, `seaborn.heatmap()`, `plotly.express.imshow()`
- **언제 사용하나요?** 여러 변수 간의 상관관계, 매트릭스 형태 데이터의 패턴 파악.
""")

# Matplotlib 예시
st.markdown("#### Matplotlib 히트맵 (imshow)")
code_heatmap_mpl_rel = """
# matplotlib.pyplot을 plt라는 별칭으로 가져옵니다.
import matplotlib.pyplot as plt
# numpy를 np라는 별칭으로 가져옵니다.
import numpy as np
# pandas를 pd라는 별칭으로 가져옵니다.
import pandas as pd

# 샘플 데이터프레임 생성
np.random.seed(0)
df = pd.DataFrame(np.random.rand(5, 5), columns=[f'Var{i+1}' for i in range(5)])
# 상관계수 행렬 계산 (실제 사용 시에는 의미 있는 데이터를 사용)
corr_matrix = df.corr()

# 새로운 그림과 축 생성
fig, ax = plt.subplots(figsize=(7, 6))
# imshow를 사용하여 히트맵을 그립니다.
# corr_matrix: 시각화할 2D 배열
# cmap: 색상 맵 (예: 'coolwarm', 'viridis', 'YlGnBu')
# interpolation: 보간 방법 (데이터 포인트 사이의 색상 처리)
cax = ax.imshow(corr_matrix, cmap='coolwarm', interpolation='nearest')
# 컬러바 추가
fig.colorbar(cax, label='Correlation Coefficient')
# 차트 제목 설정
ax.set_title('Matplotlib Heatmap (Correlation Matrix)')
# x축, y축 눈금 및 레이블 설정
ax.set_xticks(np.arange(len(corr_matrix.columns)))
ax.set_yticks(np.arange(len(corr_matrix.columns)))
ax.set_xticklabels(corr_matrix.columns, rotation=45, ha="right")
ax.set_yticklabels(corr_matrix.columns)

# 각 셀에 값 표시 (선택 사항)
for i in range(len(corr_matrix.index)):
    for j in range(len(corr_matrix.columns)):
        text = ax.text(j, i, f"{corr_matrix.iloc[i, j]:.2f}",
                       ha="center", va="center", color="black", fontsize=8)
# 레이아웃 자동 조정
plt.tight_layout()
# st.pyplot(fig)
# plt.clf()
"""
st.code(code_heatmap_mpl_rel, language='python')
if st.checkbox("Matplotlib 히트맵 예시 보기", key="heatmap_mpl_rel_page"):
    sample_data_num_df = get_sample_data('numerical')
    corr_matrix = sample_data_num_df.corr()
    fig, ax = plt.subplots(figsize=(7,6))
    cax = ax.imshow(corr_matrix, cmap='coolwarm', interpolation='nearest', vmin=-1, vmax=1)
    fig.colorbar(cax, label='Correlation Coefficient')
    ax.set_title('Matplotlib Heatmap (Correlation: A, B, C)')
    ax.set_xticks(np.arange(len(corr_matrix.columns)))
    ax.set_yticks(np.arange(len(corr_matrix.columns)))
    ax.set_xticklabels(corr_matrix.columns, rotation=45, ha="right")
    ax.set_yticklabels(corr_matrix.columns)
    for i in range(len(corr_matrix.index)): # 값 표시
        for j in range(len(corr_matrix.columns)):
            ax.text(j, i, f"{corr_matrix.iloc[i, j]:.2f}", ha="center", va="center", color="w" if abs(corr_matrix.iloc[i,j]) > 0.5 else "black", fontsize=8)
    plt.tight_layout()
    st.pyplot(fig)

# Seaborn 예시
st.markdown("#### Seaborn 히트맵")
code_heatmap_sns_rel = """
# seaborn을 sns라는 별칭으로 가져옵니다.
import seaborn as sns
# matplotlib.pyplot을 plt라는 별칭으로 가져옵니다.
import matplotlib.pyplot as plt
# pandas를 pd라는 별칭으로 가져옵니다.
import pandas as pd
# numpy를 np라는 별칭으로 가져옵니다.
import numpy as np

# 샘플 데이터프레임 생성
np.random.seed(0)
df = pd.DataFrame(np.random.rand(5, 5), columns=[f'Var{i+1}' for i in range(5)])
corr_matrix = df.corr()

# 새로운 그림과 축 생성
plt.figure(figsize=(7, 6))
# Seaborn으로 히트맵을 그립니다.
# corr_matrix: 시각화할 2D 배열
# annot: 셀 안에 값 표시 여부 (True/False 또는 값 포맷 문자열)
# cmap: 색상 맵
# fmt: annot이 True일 때 값의 포맷 (예: ".2f"는 소수점 둘째 자리)
# linewidths: 셀 사이의 선 두께
# linecolor: 셀 사이의 선 색상
sns.heatmap(corr_matrix, annot=True, cmap='viridis', fmt=".2f", linewidths=.5, linecolor='lightgray', cbar_kws={'label': 'Correlation'})
# 차트 제목 설정
plt.title('Seaborn Heatmap (Correlation Matrix)')
# 레이아웃 자동 조정
plt.tight_layout()

# st.pyplot(plt.gcf())
# plt.clf()
"""
st.code(code_heatmap_sns_rel, language='python')
if st.checkbox("Seaborn 히트맵 예시 보기", key="heatmap_sns_rel_page"):
    sample_data_num_df = get_sample_data('numerical')
    corr_matrix = sample_data_num_df.corr()
    fig, ax = plt.subplots(figsize=(7,6))
    sns.heatmap(corr_matrix, annot=True, cmap='viridis', fmt=".2f", ax=ax, linewidths=.5, linecolor='lightgray', cbar_kws={'label': 'Correlation'})
    ax.set_title('Seaborn Heatmap (Correlation: A, B, C)')
    plt.tight_layout()
    st.pyplot(fig)

# Plotly Express 예시
st.markdown("#### Plotly Express 히트맵 (imshow)")
code_heatmap_plotly_rel = """
# Plotly Express 라이브러리를 px라는 별칭으로 사용합니다.
import plotly.express as px
# pandas로 데이터프레임을 다루기 위해 import 합니다.
import pandas as pd
# numpy로 숫자 데이터를 생성/조작하기 위해 import 합니다.
import numpy as np

# 샘플 데이터프레임 생성
np.random.seed(0)
df = pd.DataFrame(np.random.rand(5, 5), columns=[f'Var{i+1}' for i in range(5)])
corr_matrix = df.corr()

# Plotly Express를 사용하여 히트맵을 생성합니다.
fig = px.imshow(
    corr_matrix,           # 시각화할 매트릭스 (상관계수 행렬)
    text_auto=True,        # 각 셀에 값을 자동으로 표시 (True 또는 소수점 형식 ".2f" 등)
    aspect="auto",         # 셀의 가로세로 비율을 자동으로 조정 ("equal"로 하면 정사각형)
    color_continuous_scale='RdBu_r', # 색상 스케일 (빨강-파랑 반전, -1에서 1 범위에 적합)
    labels=dict(color="Correlation"), # 컬러바 레이블 설정
    title="Plotly Express 상관계수 히트맵" # 차트 제목
)
# x축 레이블 회전 (선택 사항)
# fig.update_xaxes(tickangle=-45)
# y축 레이블을 매트릭스의 인덱스 이름으로 설정
fig.update_yaxes(tickvals=np.arange(len(corr_matrix.index)), ticktext=corr_matrix.index)
# x축 레이블을 매트릭스의 컬럼 이름으로 설정
fig.update_xaxes(tickvals=np.arange(len(corr_matrix.columns)), ticktext=corr_matrix.columns)


# st.plotly_chart(fig, use_container_width=True)
"""
st.code(code_heatmap_plotly_rel, language='python')
if st.checkbox("Plotly 히트맵 예시 보기", key="heatmap_plotly_rel_page"):
    sample_data_num_df = get_sample_data('numerical')
    corr_matrix = sample_data_num_df.corr()
    fig = px.imshow(
        corr_matrix, text_auto=".2f", aspect="auto",
        color_continuous_scale='RdBu_r', # 상관계수 시각화에 적합한 Red-Blue 스케일 (중간 0)
        range_color=[-1,1], # 색상 범위 고정
        labels=dict(color="상관계수"),
        title="Plotly Express 상관계수 히트맵 (A, B, C)"
    )
    # Plotly imshow는 기본적으로 축 레이블이 인덱스 숫자로 표시되므로, 명시적으로 컬럼/인덱스 이름으로 설정
    fig.update_xaxes(tickvals=np.arange(len(corr_matrix.columns)), ticktext=corr_matrix.columns)
    fig.update_yaxes(tickvals=np.arange(len(corr_matrix.index)), ticktext=corr_matrix.index)
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# --- 2.4 페어 플롯 (Pair Plot / Scatter Matrix) ---
st.subheader("2.4 페어 플롯 (Pair Plot / Scatter Matrix)")
st.markdown("""
데이터프레임 내 여러 수치형 변수들 간의 모든 가능한 조합에 대해 산점도를 그리고, 각 변수 자체의 분포는 히스토그램이나 KDE 플롯으로 대각선에 표시합니다.
- **주요 사용 라이브러리:** `seaborn.pairplot()`, `pandas.plotting.scatter_matrix()`, `plotly.express.scatter_matrix()`
- **언제 사용하나요?** 다변량 데이터에서 변수 간의 전반적인 관계와 각 변수의 분포를 한눈에 파악하고 싶을 때.
""")

# Pandas plotting (Matplotlib 기반) 예시
st.markdown("#### Pandas `scatter_matrix` (Matplotlib 기반)")
code_pairplot_pd_rel = """
# pandas.plotting에서 scatter_matrix를 가져옵니다.
from pandas.plotting import scatter_matrix
# matplotlib.pyplot을 plt라는 별칭으로 가져옵니다. (차트 표시 및 추가 설정용)
import matplotlib.pyplot as plt
# pandas를 pd라는 별칭으로 가져옵니다.
import pandas as pd
# numpy를 np라는 별칭으로 가져옵니다.
import numpy as np

# 샘플 데이터프레임 생성
np.random.seed(1)
df = pd.DataFrame(np.random.randn(100, 4), columns=['A', 'B', 'C', 'D'])

# 새로운 그림과 축 배열을 생성합니다.
# scatter_matrix는 Figure 객체와 Axes 배열을 반환합니다.
fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(10, 10)) # subplot 직접 생성은 불필요
                                                            # scatter_matrix가 알아서 처리

# scatter_matrix를 사용하여 페어 플롯을 그립니다.
# df: 사용할 데이터프레임
# alpha: 점 투명도
# figsize: 그림 크기
# diagonal: 대각선에 그릴 플롯 종류 ('hist' 또는 'kde')
# s: 점 크기
# c: 점 색상 (단일 색상 지정)
scatter_matrix(df, alpha=0.7, figsize=(10, 10), diagonal='kde', s=30, c='cornflowerblue', ax=axes)
# 전체 제목 설정 (fig.suptitle 권장)
fig = plt.gcf() # scatter_matrix가 만든 현재 Figure 가져오기
fig.suptitle('Pandas Scatter Matrix', y=1.02, fontsize=16) # y=1.02로 제목 위치 조정

# st.pyplot(fig)
# plt.clf()
"""
st.code(code_pairplot_pd_rel, language='python')
if st.checkbox("Pandas `scatter_matrix` 예시 보기", key="pairplot_pd_rel_page"):
    sample_data_num_df = get_sample_data('numerical')
    st.write("`scatter_matrix`는 변수가 많거나 데이터가 크면 생성에 시간이 걸릴 수 있습니다.")
    # scatter_matrix는 Figure를 직접 반환하지 않으므로, plt.figure()로 생성 후 ax 전달은 부적합.
    # 대신, scatter_matrix 호출 후 plt.gcf()로 현재 Figure를 가져옵니다.
    fig_sm = plt.figure(figsize=(9, 9)) # 새 Figure 생성 (선택사항, scatter_matrix도 내부적으로 만듦)
    # scatter_matrix 함수는 Axes 배열을 반환합니다.
    # 직접 ax를 전달하려면 scatter_matrix 내부가 이를 어떻게 처리하는지 확인해야 합니다.
    # 간단하게는 ax=None으로 두고, scatter_matrix가 생성한 Figure를 사용합니다.
    axes_array = scatter_matrix(sample_data_num_df[['A', 'B', 'C']], alpha=0.7, figsize=(9, 9), diagonal='kde', s=30, c='cornflowerblue')
    # 모든 subplot의 x, y 레이블 폰트 크기 조정 (선택 사항)
    for ax_row in axes_array:
        for ax in ax_row:
            ax.xaxis.label.set_size(8)
            ax.yaxis.label.set_size(8)
            ax.tick_params(axis='both', which='major', labelsize=6)

    # 현재 Figure를 가져와 제목 설정
    current_fig = plt.gcf()
    current_fig.suptitle('Pandas Scatter Matrix (A, B, C)', y=0.95, fontsize=14) # y 값 조정으로 제목 위치 변경
    st.pyplot(current_fig)


# Seaborn 예시
st.markdown("#### Seaborn `pairplot`")
code_pairplot_sns_rel = """
# seaborn을 sns라는 별칭으로 가져옵니다.
import seaborn as sns
# matplotlib.pyplot을 plt라는 별칭으로 가져옵니다. (차트 제목 등 추가 설정용)
import matplotlib.pyplot as plt
# pandas를 pd라는 별칭으로 가져옵니다.
import pandas as pd
# numpy를 np라는 별칭으로 가져옵니다.
import numpy as np

# 샘플 데이터프레임 생성
np.random.seed(1)
df = pd.DataFrame(np.random.randn(100, 4), columns=['A', 'B', 'C', 'D'])
df['Category'] = np.random.choice(['TypeX', 'TypeY'], 100)

# Seaborn으로 페어 플롯을 그립니다.
# df: 사용할 데이터프레임
# hue: 점의 색상을 구분할 카테고리형 컬럼
# diag_kind: 대각선에 그릴 플롯 종류 ('hist', 'kde', None)
# markers: hue 카테고리별 마커 모양
# palette: 색상 팔레트
# plot_kws: 산점도 부분에 전달할 추가 인자 (예: alpha, s)
g = sns.pairplot(df, hue='Category', diag_kind='kde', markers=['o', 's'], palette='husl', plot_kws={'alpha':0.6, 's':40, 'edgecolor':'k'})
# 전체 제목 설정
g.fig.suptitle('Seaborn Pairplot', y=1.02, fontsize=16) # y=1.02로 제목 위치 조정

# st.pyplot(g.fig) # pairplot은 Figure 객체를 g.fig로 접근
# plt.clf() # pairplot은 Figure를 반환하므로, g.fig를 clear하거나 plt.close(g.fig) 사용
"""
st.code(code_pairplot_sns_rel, language='python')
if st.checkbox("Seaborn `pairplot` 예시 보기", key="pairplot_sns_rel_page"):
    sample_data_mixed_df = get_sample_data('mixed')
    st.write("`pairplot`은 변수가 많거나 데이터가 크면 생성에 시간이 걸릴 수 있습니다.")
    g = sns.pairplot(sample_data_mixed_df[['Metric1', 'Metric2', 'Group']], hue='Group', diag_kind='kde', palette='husl', markers=['o', 'X', 'D'], plot_kws={'alpha':0.7, 's':50, 'edgecolor':'gray'})
    g.fig.suptitle('Seaborn Pairplot (Metrics by Group)', y=1.02)
    st.pyplot(g.fig)

# Plotly Express 예시
st.markdown("#### Plotly Express `scatter_matrix`")
code_pairplot_plotly_rel = """
# Plotly Express 라이브러리를 px라는 별칭으로 사용합니다.
import plotly.express as px
# pandas로 데이터프레임을 다루기 위해 import 합니다.
import pandas as pd
# numpy로 숫자 데이터를 생성/조작하기 위해 import 합니다.
import numpy as np

# 샘플 데이터프레임 생성
np.random.seed(1)
df = pd.DataFrame(np.random.randn(100, 4), columns=['A', 'B', 'C', 'D'])
df['Category'] = np.random.choice(['TypeX', 'TypeY'], 100)

# Plotly Express를 사용하여 페어 플롯(산점도 행렬)을 생성합니다.
fig = px.scatter_matrix(
    df,                    # 사용할 데이터프레임
    dimensions=['A', 'B', 'C', 'D'], # 산점도 행렬에 포함할 숫자형 컬럼들
    color='Category',      # 점의 색상을 'Category' 값에 따라 다르게 지정
    symbol='Category',     # 점의 모양을 'Category' 값에 따라 다르게 지정
    title="Plotly Express 페어 플롯 (Scatter Matrix)", # 차트 제목
    labels={col: col.replace('_', ' ') for col in df.columns}, # 레이블 공백 처리
    height=700,            # 차트 높이
    width=700              # 차트 너비
)
# 대각선 그래프, 상/하단 삼각형 표시 여부 등 미세 조정
# fig.update_traces(diagonal_visible=False) # 대각선 히스토그램/KDE 숨기기
# fig.update_traces(showupperhalf=False)   # 상단 삼각형 숨기기
# fig.update_layout(legend_orientation="h", legend_yanchor="bottom", legend_y=1.02, legend_xanchor="right", legend_x=1) # 범례 위치

# st.plotly_chart(fig, use_container_width=True) # 너비는 use_container_width로, 높이는 fig.update_layout(height=...)
"""
st.code(code_pairplot_plotly_rel, language='python')
if st.checkbox("Plotly `scatter_matrix` 예시 보기", key="pairplot_plotly_rel_page"):
    sample_data_mixed_df = get_sample_data('mixed')
    st.write("Plotly `scatter_matrix`는 변수가 많거나 데이터가 크면 생성에 시간이 걸릴 수 있습니다.")
    fig = px.scatter_matrix(
        sample_data_mixed_df,
        dimensions=['Metric1', 'Metric2'],
        color='Group', symbol='Group',
        title="Plotly Express Scatter Matrix (Metrics by Group)",
        labels={col: col.replace('_', ' ') for col in sample_data_mixed_df.columns}, # 컬럼 이름 공백 처리
        height=600 # 높이 지정 (너비는 use_container_width로 자동 조절)
    )
    # 대각선 그래프 표시, 상/하단 삼각형 표시 여부 등 미세 조정
    fig.update_traces(diagonal_visible=True) # 대각선 분포 표시 (기본값)
    fig.update_layout(legend_title_text='그룹')
    st.plotly_chart(fig, use_container_width=True)