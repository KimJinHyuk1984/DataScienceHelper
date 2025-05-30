# pages/6_🌍_지리_정보.py
import streamlit as st
import pandas as pd
import numpy as np
# matplotlib, seaborn은 지리 정보 시각화에 직접적으로 많이 쓰이지 않으므로 주석 처리
# import matplotlib.pyplot as plt
# import seaborn as sns
import plotly.express as px
from utils.utils import get_sample_data # 여기서는 get_sample_data를 직접 사용하지 않을 수 있음

st.header("6. 지리적 데이터 시각화 (간단 소개)")
st.markdown("""
지도 위에 데이터를 표현하여 지리적 패턴이나 분포를 파악합니다.
이 분야는 `folium`, `geopandas` 등과 함께 `plotly.express`의 지도 관련 기능을 사용하는 것이 일반적입니다.
여기서는 **Plotly Express**를 중심으로 간단한 예시를 소개합니다.
""")

# --- 6.1 단계 구분도 (Choropleth Map) ---
st.subheader("6.1 단계 구분도 (Choropleth Map)")
st.markdown("""
지역(국가, 주, 도시 등)별로 특정 데이터 값의 크기를 색상의 농도나 종류로 표현합니다.
- **주요 사용 라이브러리:** `plotly.express.choropleth()`, `folium` + `geopandas`
- **언제 사용하나요?** 지역별 인구 밀도, 선거 결과, 1인당 GDP 등 지역 단위로 집계된 데이터를 비교할 때.
""")
code_choropleth_plotly_geo = """
# Plotly Express 라이브러리를 px라는 별칭으로 가져옵니다.
import plotly.express as px
# pandas로 데이터프레임을 다루기 위해 import 합니다. (데이터 로드 및 처리)
import pandas as pd

# Plotly Express에 내장된 gapminder 데이터를 사용합니다.
# 이 데이터는 국가별 연도별 기대 수명(lifeExp), 인구(pop), GDP(gdpPercap) 등의 정보를 포함합니다.
# 각 국가를 식별하기 위한 'iso_alpha' (ISO 3166-1 alpha-3 국가 코드) 컬럼도 있습니다.
df_gapminder = px.data.gapminder()

# 2007년도 데이터만 필터링합니다.
df_2007 = df_gapminder.query("year == 2007")

# 단계 구분도(Choropleth Map)를 생성합니다.
fig_choropleth = px.choropleth(
    df_2007,                                  # 사용할 데이터프레임
    locations="iso_alpha",                    # 국가(지역)를 식별하는 컬럼 이름 (ISO Alpha-3 코드)
    color="lifeExp",                          # 색상으로 표현할 데이터 컬럼 (여기서는 기대 수명)
    hover_name="country",                     # 마우스를 올렸을 때 표시될 국가(지역) 이름 컬럼
    color_continuous_scale=px.colors.sequential.Plasma, # 연속적인 값에 대한 색상 스케일
    title="2007년 국가별 기대 수명 (Choropleth Map)",     # 차트 제목
    projection="natural earth"                # 지도 투영 방식 (다양한 방식 사용 가능)
)

# 차트 레이아웃 업데이트 (선택 사항)
fig_choropleth.update_layout(
    margin={"r":0,"t":50,"l":0,"b":0}, # 차트 여백 조정 (오른쪽, 위, 왼쪽, 아래)
    geo=dict(
        showframe=False,           # 지도 프레임(테두리) 숨기기
        showcoastlines=True,       # 해안선 표시
        bgcolor='rgba(0,0,0,0)'    # 배경색 투명하게 (선택 사항)
    )
)

# Streamlit 앱에 Plotly 차트를 표시합니다.
# st.plotly_chart(fig_choropleth, use_container_width=True)
"""
st.code(code_choropleth_plotly_geo, language='python')
if st.checkbox("Plotly 단계 구분도 예시 보기", key="choropleth_plotly_geo_page"):
    try:
        df_gapminder = px.data.gapminder()
        df_2007 = df_gapminder.query("year == 2007")
        fig_choropleth = px.choropleth(
            df_2007, locations="iso_alpha", color="lifeExp",
            hover_name="country", color_continuous_scale=px.colors.sequential.Viridis, # 다른 색상 스케일
            title="2007년 국가별 기대 수명 (Plotly Express)",
            projection="robinson" # 다른 지도 투영 방식
        )
        fig_choropleth.update_layout(
            margin={"r":0,"t":50,"l":0,"b":0},
            geo=dict(showframe=False, showcoastlines=True, landcolor='lightgray', oceancolor='lightblue')
        )
        st.plotly_chart(fig_choropleth, use_container_width=True)
    except Exception as e:
        st.error(f"Plotly 단계 구분도 예시를 로드하는 중 오류 발생: {e}")
        st.info("이 예시를 실행하려면 `plotly` 라이브러리가 설치되어 있어야 합니다: `pip install plotly`")

st.markdown("---")

# --- 6.2 포인트 맵 (Point Map / Scatter Map) ---
st.subheader("6.2 포인트 맵 (Point Map / Scatter Map)")
st.markdown("""
지도 위에 특정 위치(위도, 경도)를 점으로 표시합니다. 점의 크기나 색상으로 추가 정보를 나타낼 수도 있습니다.
- **주요 사용 라이브러리:** `plotly.express.scatter_geo()`, `plotly.express.scatter_mapbox()`
- **언제 사용하나요?** 특정 사건 발생 위치, 매장 위치, 지진 발생 지점 등 지리적 좌표를 가진 데이터를 시각화할 때.
  - `scatter_geo`: 간단한 지리적 배경 위에 점 표시.
  - `scatter_mapbox`: 상세한 타일맵(OpenStreetMap 등) 배경 위에 점 표시 (Mapbox API 토큰 필요할 수 있음 - 공개 스타일은 무료).
""")

# Plotly Express scatter_geo 예시
st.markdown("#### Plotly Express `scatter_geo`")
code_scatter_geo_plotly_geo = """
# Plotly Express 라이브러리를 px라는 별칭으로 가져옵니다.
import plotly.express as px
# pandas로 데이터프레임을 다루기 위해 import 합니다.
import pandas as pd

# 샘플 데이터 생성 (도시별 인구 데이터 가정)
# 실제 데이터에서는 정확한 위도(lat), 경도(lon) 정보가 필요합니다.
cities_data = {
    'city': ['Seoul', 'New York', 'Paris', 'Tokyo', 'London'],
    'lat': [37.5665, 40.7128, 48.8566, 35.6895, 51.5074],
    'lon': [126.9780, -74.0060, 2.3522, 139.6917, -0.1278],
    'population': [9776000, 8399000, 2141000, 13960000, 8982000],
    'country': ['South Korea', 'USA', 'France', 'Japan', 'UK']
}
df_cities = pd.DataFrame(cities_data)

# scatter_geo를 사용하여 지리적 산점도를 생성합니다.
fig_scatter_geo = px.scatter_geo(
    df_cities,             # 사용할 데이터프레임
    lat='lat',             # 위도를 나타내는 컬럼
    lon='lon',             # 경도를 나타내는 컬럼
    size='population',     # 점의 크기를 인구 수에 따라 다르게 지정
    color='country',       # 점의 색상을 국가에 따라 다르게 지정
    hover_name='city',     # 마우스를 올렸을 때 표시될 도시 이름
    projection='orthographic', # 지도 투영 방식 (지구본 모양)
    title='주요 도시 인구 (scatter_geo)'
)
# 차트 레이아웃 업데이트 (선택 사항)
fig_scatter_geo.update_layout(margin={"r":0,"t":30,"l":0,"b":0})
fig_scatter_geo.update_geos(landcolor="lightgreen", oceancolor="lightblue", showcountries=True, countrycolor="RebeccaPurple")


# st.plotly_chart(fig_scatter_geo, use_container_width=True)
"""
st.code(code_scatter_geo_plotly_geo, language='python')
if st.checkbox("Plotly `scatter_geo` 예시 보기", key="scatter_geo_plotly_geo_page"):
    cities_data = {
        'city': ['Seoul', 'New York', 'Paris', 'Tokyo', 'London', 'Beijing', 'Moscow', 'Sydney'],
        'lat': [37.5665, 40.7128, 48.8566, 35.6895, 51.5074, 39.9042, 55.7558, -33.8688],
        'lon': [126.9780, -74.0060, 2.3522, 139.6917, -0.1278, 116.4074, 37.6173, 151.2093],
        'population_mil': [9.7, 8.4, 2.1, 13.9, 8.9, 21.5, 12.5, 5.3], # 백만 단위
        'country': ['South Korea', 'USA', 'France', 'Japan', 'UK', 'China', 'Russia', 'Australia']
    }
    df_cities = pd.DataFrame(cities_data)
    fig_scatter_geo = px.scatter_geo(
        df_cities, lat='lat', lon='lon', size='population_mil', color='country',
        hover_name='city', projection='natural earth',
        title='주요 도시 인구 (단위: 백만)',
        size_max=30 # 최대 점 크기
    )
    fig_scatter_geo.update_layout(margin={"r":0,"t":40,"l":0,"b":0})
    fig_scatter_geo.update_geos(landcolor="rgb(217, 217, 217)", subunitcolor="rgb(255,255,255)")
    st.plotly_chart(fig_scatter_geo, use_container_width=True)


# Plotly Express scatter_mapbox 예시
st.markdown("#### Plotly Express `scatter_mapbox`")
code_scatter_mapbox_plotly_geo = """
# Plotly Express 라이브러리를 px라는 별칭으로 사용합니다.
import plotly.express as px
# pandas로 데이터프레임을 다루기 위해 import 합니다.
import pandas as pd

# Plotly Express에 내장된 carshare 데이터를 사용합니다.
# 이 데이터는 차량 공유 서비스의 픽업 위치(위도: centroid_lat, 경도: centroid_lon) 정보를 포함합니다.
df_carshare = px.data.carshare()

# scatter_mapbox를 사용하여 타일맵 배경의 산점도를 생성합니다.
fig_scatter_mapbox = px.scatter_mapbox(
    df_carshare,                             # 사용할 데이터프레임
    lat="centroid_lat",                      # 위도를 나타내는 컬럼
    lon="centroid_lon",                      # 경도를 나타내는 컬럼
    color="peak_hour",                       # 색상으로 표현할 데이터 컬럼 (피크 시간 여부, 0 또는 1)
    size="car_hours",                        # 점의 크기로 표현할 데이터 컬럼 (차량 사용 시간)
    color_continuous_scale=px.colors.cyclical.IceFire, # 연속형 색상 스케일 (peak_hour가 숫자형일 경우)
                                                      # 만약 peak_hour가 범주형이라면 color_discrete_map 또는 color_discrete_sequence 사용
    size_max=15,                             # 점의 최대 크기 설정
    zoom=10,                                 # 초기 지도의 확대/축소 레벨
    mapbox_style="carto-positron",           # Mapbox 배경 지도 스타일 (공개 스타일 중 하나)
                                             # (옵션: "open-street-map", "white-bg", "stamen-terrain", 등)
    title="차량 공유 데이터 포인트 맵 (scatter_mapbox)" # 차트 제목
)
# 차트 레이아웃 업데이트 (선택 사항)
fig_scatter_mapbox.update_layout(
    margin={"r":0,"t":30,"l":0,"b":0},
    mapbox_accesstoken="YOUR_MAPBOX_ACCESS_TOKEN" # 비공개 스타일 사용 시 필요. 공개 스타일은 불필요.
                                                 # Streamlit 공유 앱에서는 토큰 관리에 유의해야 함.
)


# st.plotly_chart(fig_scatter_mapbox, use_container_width=True)
"""
st.code(code_scatter_mapbox_plotly_geo, language='python')
if st.checkbox("Plotly `scatter_mapbox` 예시 보기", key="scatter_mapbox_plotly_geo_page"):
    try:
        df_carshare = px.data.carshare()
        fig_scatter_mapbox = px.scatter_mapbox(
            df_carshare, lat="centroid_lat", lon="centroid_lon",
            color="peak_hour", # peak_hour는 0 또는 1의 값을 가짐
            size="car_hours",
            color_continuous_scale=px.colors.diverging.Portland, # 0과 1을 구분하는 색상 스케일
            # 만약 peak_hour를 범주형으로 취급하고 싶다면, df_carshare['peak_hour'] = df_carshare['peak_hour'].astype(str) 후 사용
            # color_discrete_map={'0': 'blue', '1': 'red'},
            size_max=15, zoom=10,
            mapbox_style="open-street-map", # 공개 스타일 사용
            title="차량 공유 데이터 포인트 맵 (Plotly Express)"
        )
        fig_scatter_mapbox.update_layout(margin={"r":0,"t":40,"l":0,"b":0})
        # Mapbox 토큰이 필수는 아니지만, 일부 고급 기능이나 사용량 제한 해제에 필요할 수 있습니다.
        # 공개 스타일 사용 시에는 대부분 토큰 없이 동작합니다.
        st.plotly_chart(fig_scatter_mapbox, use_container_width=True)
    except Exception as e:
        st.error(f"Plotly scatter_mapbox 예시를 로드하는 중 오류 발생: {e}")
        st.info("이 예시를 실행하려면 `plotly` 라이브러리가 설치되어 있어야 하며, 네트워크 연결이 필요할 수 있습니다.")

st.warning("⚠️ `scatter_mapbox` 사용 시, 복잡한 지도 스타일이나 많은 데이터를 로드할 경우 Mapbox API 토큰이 필요하거나 성능에 영향을 줄 수 있습니다. 공개된 기본 스타일은 대부분 토큰 없이 사용 가능합니다.")