# pages/3_⏳_시간에_따른_변화.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from utils.utils import get_sample_data

st.header("3. 시간에 따른 변화 표시 (시계열)")
st.markdown("""
시간의 흐름에 따라 데이터가 어떻게 변하는지 나타낼 때 사용합니다. 주가, 기온 변화, 판매량 추이 등이 대표적인 예입니다.
**라인 플롯(Line Plot)**이 가장 일반적으로 사용됩니다.
""")

# --- 3.1 시계열 라인 플롯 (Time Series Line Plot) ---
st.subheader("3.1 시계열 라인 플롯 (Time Series Line Plot)")
st.markdown("""
시간을 x축으로, 관측값을 y축으로 하여 데이터 포인트를 선으로 연결합니다.
- **주요 사용 라이브러리:** `matplotlib.pyplot.plot()`, `seaborn.lineplot()`, `plotly.express.line()`
- **언제 사용하나요?** 특정 기간 동안의 데이터 변화 추세, 주기성, 계절성 등을 파악할 때.
""")

# Matplotlib 예시
st.markdown("#### Matplotlib 시계열 라인 플롯")
code_ts_line_mpl_time = """
# matplotlib.pyplot을 plt라는 별칭으로 가져옵니다.
import matplotlib.pyplot as plt
# pandas를 pd라는 별칭으로 가져옵니다. (시계열 데이터 생성용)
import pandas as pd
# numpy를 np라는 별칭으로 가져옵니다. (랜덤 데이터 생성용)
import numpy as np

# 샘플 시계열 데이터 생성
date_rng = pd.date_range(start='2023-01-01', end='2023-03-31', freq='D')
data = pd.DataFrame(date_rng, columns=['date'])
data['value'] = np.random.randn(len(date_rng)).cumsum() + 10 # 랜덤 워크 데이터

# 새로운 그림과 축 생성
plt.figure(figsize=(12, 6))
# 라인 플롯을 그립니다.
plt.plot(data['date'], data['value'], marker='.', linestyle='-', color='teal', label='Value over Time')
# 차트 제목 설정
plt.title('Matplotlib Time Series Line Plot')
# x축 레이블 설정
plt.xlabel('Date')
# y축 레이블 설정
plt.ylabel('Value')
# 범례 표시
plt.legend()
# 그리드 추가
plt.grid(True, linestyle='--', alpha=0.5)
# x축 날짜 레이블 자동 포맷팅 및 회전
plt.gcf().autofmt_xdate()

# st.pyplot(plt.gcf())
# plt.clf()
"""
st.code(code_ts_line_mpl_time, language='python')
if st.checkbox("Matplotlib 시계열 라인 플롯 예시 보기", key="ts_line_mpl_time_page"):
    sample_data_ts_df = get_sample_data('timeseries') # StockA, StockB 컬럼과 DatetimeIndex
    fig, ax = plt.subplots(figsize=(12,6))
    ax.plot(sample_data_ts_df.index, sample_data_ts_df['StockA'], marker='.', linestyle='-', color='teal', label='Stock A')
    ax.plot(sample_data_ts_df.index, sample_data_ts_df['StockB'], marker='x', linestyle='--', color='tomato', label='Stock B')
    ax.set_title('Matplotlib Time Series Line Plot')
    ax.set_xlabel('Date')
    ax.set_ylabel('Stock Price')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.5)
    fig.autofmt_xdate() # x축 날짜 레이블 자동 포맷팅
    st.pyplot(fig)

# Seaborn 예시
st.markdown("#### Seaborn 시계열 라인 플롯")
code_ts_line_sns_time = """
# seaborn을 sns라는 별칭으로 가져옵니다.
import seaborn as sns
# matplotlib.pyplot을 plt라는 별칭으로 가져옵니다.
import matplotlib.pyplot as plt
# pandas를 pd라는 별칭으로 가져옵니다.
import pandas as pd
# numpy를 np라는 별칭으로 가져옵니다.
import numpy as np

# 샘플 시계열 데이터 생성
date_rng = pd.date_range(start='2023-01-01', end='2023-03-31', freq='D')
df_ts = pd.DataFrame(date_rng, columns=['date'])
df_ts['value_A'] = np.random.randn(len(date_rng)).cumsum() + 10
df_ts['value_B'] = np.random.randn(len(date_rng)).cumsum() + 15
# Seaborn은 long-form 데이터를 선호
df_melted = df_ts.melt(id_vars='date', var_name='series', value_name='value')

# 새로운 그림과 축 생성
plt.figure(figsize=(12, 6))
# Seaborn으로 라인 플롯을 그립니다.
sns.lineplot(x='date', y='value', hue='series', style='series', data=df_melted, markers=True, dashes=False)
# 차트 제목 설정
plt.title('Seaborn Time Series Line Plot')
# x축 레이블 설정
plt.xlabel('Date')
# y축 레이블 설정
plt.ylabel('Value')
# x축 날짜 레이블 회전 및 정렬
plt.xticks(rotation=30, ha='right')
# 레이아웃 자동 조정
plt.tight_layout()

# st.pyplot(plt.gcf())
# plt.clf()
"""
st.code(code_ts_line_sns_time, language='python')
if st.checkbox("Seaborn 시계열 라인 플롯 예시 보기", key="ts_line_sns_time_page"):
    sample_data_ts_df = get_sample_data('timeseries').reset_index() # DatetimeIndex를 'index' 컬럼으로
    sample_data_ts_df = sample_data_ts_df.rename(columns={'index': 'Date'}) # 컬럼명 변경
    df_melted = sample_data_ts_df.melt(id_vars='Date', var_name='Stock_Symbol', value_name='Price')

    fig, ax = plt.subplots(figsize=(12,6))
    sns.lineplot(x='Date', y='Price', hue='Stock_Symbol', style='Stock_Symbol', data=df_melted, ax=ax, markers=True, dashes=False)
    ax.set_title('Seaborn Time Series Line Plot')
    ax.set_xlabel('Date')
    ax.set_ylabel('Stock Price')
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    st.pyplot(fig)

# Plotly Express 예시
st.markdown("#### Plotly Express 시계열 라인 플롯")
code_ts_line_plotly_time = """
# Plotly Express 라이브러리를 px라는 별칭으로 사용합니다.
import plotly.express as px
# pandas로 데이터프레임과 시계열 데이터를 다루기 위해 import 합니다.
import pandas as pd
# numpy로 숫자 데이터를 생성/조작하기 위해 import 합니다.
import numpy as np

# 샘플 시계열 데이터 생성
date_rng = pd.date_range(start='2023-01-01', end='2023-03-31', freq='D')
df_ts = pd.DataFrame(date_rng, columns=['date'])
df_ts['value_A'] = np.random.randn(len(date_rng)).cumsum() + 10
df_ts['value_B'] = np.random.randn(len(date_rng)).cumsum() + 15
# Plotly Express도 long-form 데이터를 선호
df_melted = df_ts.melt(id_vars='date', var_name='series_id', value_name='measurement')

# Plotly Express를 사용하여 라인 플롯을 생성합니다.
fig = px.line(
    df_melted,             # 사용할 데이터프레임 (long-form)
    x='date',              # x축으로 사용할 날짜/시간 컬럼
    y='measurement',       # y축으로 사용할 값 컬럼
    color='series_id',     # 라인의 색상을 'series_id' 값에 따라 다르게 지정
    symbol='series_id',    # 각 데이터 포인트의 마커 모양을 'series_id' 값에 따라 다르게 지정
    title="Plotly Express 시계열 라인 플롯", # 차트 제목
    labels={'measurement': '측정값', 'series_id': '시리즈 구분', 'date':'날짜'} # 축 및 범례 레이블 변경
)
# 차트 레이아웃 업데이트 (선택 사항)
fig.update_traces(marker=dict(size=4), connectgaps=False) # connectgaps=True는 결측값 있어도 라인 연결
fig.update_layout(
    xaxis_title_text='날짜',
    yaxis_title_text='값',
    xaxis_rangeslider_visible=True # x축 아래에 범위 슬라이더 추가 (인터랙티브)
)
# x축 날짜 형식 지정 (선택 사항)
# fig.update_xaxes(dtick="M1", tickformat="%b\n%Y") # 월 단위 눈금, "월\n연도" 형식

# st.plotly_chart(fig, use_container_width=True)
"""
st.code(code_ts_line_plotly_time, language='python')
if st.checkbox("Plotly 시계열 라인 플롯 예시 보기", key="ts_line_plotly_time_page"):
    sample_data_ts_df = get_sample_data('timeseries').reset_index()
    sample_data_ts_df = sample_data_ts_df.rename(columns={'index': 'Date'})
    df_melted = sample_data_ts_df.melt(id_vars='Date', var_name='Stock_Symbol', value_name='Price')

    fig = px.line(
        df_melted, x='Date', y='Price', color='Stock_Symbol', symbol='Stock_Symbol',
        title="Plotly Express Time Series Line Plot",
        labels={'Price': '주가', 'Stock_Symbol': '주식', 'Date': '날짜'}
    )
    fig.update_traces(marker=dict(size=4), connectgaps=True) # connectgaps는 NaN이 있어도 선을 연결
    fig.update_layout(
        xaxis_title_text='날짜', yaxis_title_text='주가',
        xaxis_rangeslider_visible=True # x축 하단에 범위 선택 슬라이더 표시
    )
    st.plotly_chart(fig, use_container_width=True)