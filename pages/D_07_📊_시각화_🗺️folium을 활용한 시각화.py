# pages/7_🗺️_Folium_지도.py
import streamlit as st
import folium
from streamlit_folium import st_folium # Streamlit에서 Folium을 사용하기 위한 라이브러리
import pandas as pd
import numpy as np # 데이터 생성 및 처리에 사용
import branca.colormap as cm # 컬러맵 사용을 위해 (선택 사항)

# 페이지 설정 (메인 앱에서 한 번만 하는 것이 일반적)
# st.set_page_config(layout="wide")

st.header("7. Folium을 이용한 인터랙티브 지도 시각화")
st.markdown("""
`folium`은 Python 생태계에서 Leaflet.js 라이브러리를 활용하여 다양한 인터랙티브 지도를 손쉽게 만들 수 있도록 지원하는 강력한 도구입니다.
데이터를 지도 위에 시각화하여 지리적 패턴과 인사이트를 효과적으로 전달할 수 있습니다.

이 페이지에서는 `folium`의 기본 사용법과 Streamlit 환경에서의 통합 방법을 소개합니다.
`streamlit-folium` 라이브러리를 사용하면 Streamlit 앱 내에 Folium 지도를 쉽게 임베드하고 상호작용할 수 있습니다.
""")

st.info("💡 예제를 실행하려면 `folium`과 `streamlit-folium` 라이브러리가 설치되어 있어야 합니다: `pip install folium streamlit-folium`")

# --- 7.1 기본 지도 생성 ---
st.subheader("7.1 기본 지도 생성")
st.markdown("""
`folium.Map()` 함수를 사용하여 기본적인 지도를 생성할 수 있습니다. `location` 파라미터로 지도의 초기 중심 좌표를, `zoom_start`로 초기 확대/축소 레벨을 설정합니다.
""")

code_basic_map = """
# folium 라이브러리를 가져옵니다.
import folium
# Streamlit에서 folium 지도를 표시하기 위해 streamlit_folium에서 st_folium을 가져옵니다.
from streamlit_folium import st_folium

# 지도의 중심이 될 위도와 경도를 지정합니다. (예: 서울 시청)
latitude = 37.5665
longitude = 126.9780

# folium.Map 객체를 생성합니다.
# location: [위도, 경도] 형태의 리스트로, 지도의 초기 중심점을 설정합니다.
# zoom_start: 초기 확대/축소 레벨을 설정합니다. 숫자가 클수록 확대됩니다.
# tiles: 지도 타일의 종류를 설정합니다. (기본값: "OpenStreetMap", 그 외 "Stamen Terrain", "Stamen Toner", "CartoDB positron" 등)
m = folium.Map(location=[latitude, longitude], zoom_start=12, tiles="OpenStreetMap")

# 생성된 지도를 Streamlit 앱에 표시합니다.
# st_folium은 Folium Map 객체를 받아 Streamlit 컴포넌트로 렌더링합니다.
# width, height 파라미터로 지도의 크기를 조절할 수 있습니다.
map_output = st_folium(m, width=700, height=500)

# map_output에는 지도와의 상호작용 결과(예: 클릭한 마커 정보)가 담길 수 있습니다.
# st.write(map_output) # 필요한 경우 상호작용 결과 확인
"""
st.code(code_basic_map, language='python')

if st.checkbox("기본 Folium 지도 예시 보기", key="folium_basic_map_page"):
    # 지도의 중심이 될 위도와 경도를 지정합니다. (예: 서울 남산타워)
    namsan_lat, namsan_lon = 37.5512, 126.9882
    # folium.Map 객체를 생성합니다.
    m_basic = folium.Map(location=[namsan_lat, namsan_lon], zoom_start=13, tiles="CartoDB positron")
    # 생성된 지도를 Streamlit 앱에 표시합니다.
    st.markdown("##### 서울 남산타워 중심의 기본 지도:")
    st_data = st_folium(m_basic, width=700, height=400)
    # st.write("지도 상호작용 데이터:", st_data) # 클릭, 이동 등의 이벤트 데이터 확인 가능

st.markdown("---")

# --- 7.2 마커 추가하기 ---
st.subheader("7.2 마커(Markers) 추가하기")
st.markdown("""
지도 위에 특정 지점을 표시하기 위해 마커를 추가할 수 있습니다. `folium.Marker()`를 사용하며, `popup`과 `tooltip`으로 추가 정보를 제공할 수 있습니다.
""")
code_markers = """
import folium
from streamlit_folium import st_folium

# 기본 지도 생성 (예: 대한민국 중심)
map_with_markers = folium.Map(location=[36.5, 127.5], zoom_start=7)

# 마커 정보 리스트 (위도, 경도, 팝업 내용, 툴팁 내용, 아이콘)
marker_locations = [
    {'location': [37.5665, 126.9780], 'popup': '<b>서울 시청</b><br>대한민국의 수도', 'tooltip': 'Seoul City Hall', 'icon': folium.Icon(color='blue', icon='info-sign')},
    {'location': [35.1796, 129.0756], 'popup': '<i>부산 해운대</i>', 'tooltip': 'Haeundae, Busan', 'icon': folium.Icon(color='red', icon='cloud')},
    {'location': [35.8714, 128.6014], 'popup': '대구', 'tooltip': 'Daegu', 'icon': folium.Icon(color='green', prefix='fa', icon='tree')} # FontAwesome 아이콘 사용
]

# 각 위치에 마커 추가
for marker_info in marker_locations:
    folium.Marker(
        location=marker_info['location'], # 마커의 위치 [위도, 경도]
        popup=folium.Popup(marker_info['popup'], max_width=300), # 클릭 시 나타나는 팝업창 (HTML 지원)
        tooltip=marker_info['tooltip'],   # 마우스 오버 시 나타나는 툴팁
        icon=marker_info.get('icon')      # 마커 아이콘 (folium.Icon 객체)
    ).add_to(map_with_markers) # 생성된 마커를 지도 객체에 추가

# Streamlit에 표시
# st_folium(map_with_markers, width=700, height=500)
"""
st.code(code_markers, language='python')

if st.checkbox("Folium 마커 예시 보기", key="folium_markers_page"):
    map_with_markers = folium.Map(location=[36.0, 127.8], zoom_start=7, tiles="OpenStreetMap")
    marker_locations = [
        {'location': [37.5665, 126.9780], 'popup': '<b>서울 시청</b><br>Seoul City Hall', 'tooltip': '서울', 'icon': folium.Icon(color='blue', icon='glyphicon-home', prefix='glyphicon')},
        {'location': [35.1796, 129.0756], 'popup': '<i>부산 해운대</i><br>Haeundae Beach', 'tooltip': '부산', 'icon': folium.Icon(color='red', icon='umbrella-beach', prefix='fa')}, # FontAwesome 아이콘
        {'location': [33.4996, 126.5312], 'popup': '제주 국제공항', 'tooltip': '제주', 'icon': folium.Icon(color='green', icon='plane', prefix='fa')}
    ]
    for marker_info in marker_locations:
        folium.Marker(
            location=marker_info['location'],
            popup=folium.Popup(marker_info['popup'], max_width=250),
            tooltip=marker_info['tooltip'],
            icon=marker_info.get('icon')
        ).add_to(map_with_markers)
    st.markdown("##### 주요 도시 마커 지도:")
    st_folium(map_with_markers, width=700, height=450)

st.markdown("---")

# --- 7.3 원형 마커 및 도형 추가 ---
st.subheader("7.3 원형 마커(Circle Markers) 및 도형 추가")
st.markdown("""
`folium.Circle()`이나 `folium.CircleMarker()`를 사용하여 원형 마커를, `folium.Polygon()` 등으로 다각형을 지도에 추가할 수 있습니다.
- `Circle`: 반경(radius)을 미터(meter) 단위로 지정. 지도 확대/축소에 따라 크기 변경.
- `CircleMarker`: 반경(radius)을 픽셀(pixel) 단위로 지정. 지도 확대/축소에도 크기 고정.
""")
code_circles_shapes = """
import folium
from streamlit_folium import st_folium

# 기본 지도 생성 (예: 경기도 수원)
map_with_circles = folium.Map(location=[37.2636, 127.0286], zoom_start=10)

# CircleMarker 예시 (픽셀 단위 반경)
folium.CircleMarker(
    location=[37.2636, 127.0286], # 중심 위치 (수원시청)
    radius=15,                    # 원의 반경 (픽셀 단위)
    color='crimson',              # 원의 테두리 색상
    fill=True,                    # 내부 채우기 여부
    fill_color='crimson',         # 내부 채우기 색상
    fill_opacity=0.6,             # 내부 채우기 투명도
    popup='수원시청 (CircleMarker)',
    tooltip='고정 크기 원'
).add_to(map_with_circles)

# Circle 예시 (미터 단위 반경)
folium.Circle(
    location=[37.3390, 127.2050], # 중심 위치 (에버랜드 근처)
    radius=5000,                  # 원의 반경 (미터 단위, 5km)
    color='blue',                 # 원의 테두리 색상
    fill=True,
    fill_color='#3186cc',
    fill_opacity=0.3,
    popup='에버랜드 반경 5km (Circle)',
    tooltip='실제 크기 원'
).add_to(map_with_circles)

# Polygon (다각형) 예시
polygon_points = [ # 다각형을 구성하는 꼭짓점들의 [위도, 경도] 리스트
    [37.4000, 127.1000], # 성남시 부근1
    [37.3500, 127.1500], # 성남시 부근2
    [37.3000, 127.1000], # 성남시 부근3
    [37.3500, 127.0500]  # 성남시 부근4
]
folium.Polygon(
    locations=polygon_points,
    color='green',
    fill=True,
    fill_color='darkgreen',
    fill_opacity=0.4,
    popup='임의의 다각형 영역',
    tooltip='Polygon Area'
).add_to(map_with_circles)


# Streamlit에 표시
# st_folium(map_with_circles, width=700, height=500)
"""
st.code(code_circles_shapes, language='python')

if st.checkbox("Folium 원형 마커 및 도형 예시 보기", key="folium_circles_page"):
    map_with_circles = folium.Map(location=[37.28, 127.1], zoom_start=9)
    folium.CircleMarker(
        location=[37.2636, 127.0286], radius=12, color='red', fill=True, fill_color='pink', fill_opacity=0.7,
        popup='수원시청 (CircleMarker)', tooltip='고정 크기'
    ).add_to(map_with_circles)
    folium.Circle(
        location=[37.413294, 127.269348], radius=10000, color='purple', fill=True, fill_color='plum', fill_opacity=0.4,
        popup='광주시 반경 10km (Circle)', tooltip='실제 크기 (10km)'
    ).add_to(map_with_circles)

    polygon_points = [[37.2, 127.0], [37.1, 127.1], [37.0, 127.0], [37.1, 126.9]]
    folium.Polygon(locations=polygon_points, color='orange', fill=True, fill_color='gold', fill_opacity=0.5,
                   popup='임의의 Polygon', tooltip='다각형').add_to(map_with_circles)

    st.markdown("##### 원형 마커 및 도형 지도:")
    st_folium(map_with_circles, width=700, height=500)

st.markdown("---")

# --- 7.4 단계 구분도 (Choropleth Map) ---
st.subheader("7.4 단계 구분도 (Choropleth Map)")
st.markdown("""
`folium.Choropleth()`를 사용하여 단계 구분도를 만들 수 있습니다. 이는 특정 지역 경계(GeoJSON 데이터)에 따라 통계 데이터를 색상으로 표현하는 지도입니다.
- **`geo_data`**: 지역 경계 정보를 담은 GeoJSON 파일 경로 또는 URL, 또는 GeoJSON 형태의 문자열/딕셔너리.
- **`data`**: Pandas DataFrame으로, 지역을 식별하는 키 컬럼과 시각화할 값 컬럼을 포함해야 합니다.
- **`columns`**: `[키_컬럼, 값_컬럼]` 리스트. 키 컬럼은 GeoJSON의 feature id와 매칭되어야 합니다.
- **`key_on`**: GeoJSON 데이터 내에서 지역을 식별하는 키 경로 (예: `feature.id` 또는 `feature.properties.name`).
""")

code_choropleth_folium = """
import folium
from streamlit_folium import st_folium
import pandas as pd
import requests # URL에서 GeoJSON 데이터를 가져오기 위함

# 샘플 GeoJSON 데이터 URL (미국 주 경계)
# 실제 사용 시에는 프로젝트에 맞는 GeoJSON 파일을 준비해야 합니다.
geojson_url = "https.raw.githubusercontent.com/python-visualization/folium/main/examples/data/us-states.json"

# GeoJSON 데이터 로드 (URL에서 직접 가져오기)
# response = requests.get(geojson_url)
# geo_json_data = response.json()
# 만약 로컬 파일이라면:
# with open('path/to/your/us-states.json', 'r') as f:
#    geo_json_data = json.load(f)
# 여기서는 예시를 위해 URL을 그대로 사용 (folium.Choropleth 함수가 URL 직접 처리 가능)


# 샘플 통계 데이터 생성 (Pandas DataFrame)
# 실제 데이터에서는 각 주의 ID (GeoJSON의 feature.id와 일치)와 해당 값을 가져와야 합니다.
# 미국 주 이름 또는 FIPS 코드를 사용할 수 있습니다. (위 GeoJSON은 FIPS 코드를 id로 사용)
us_state_data_dict = {
    'State_FIPS': ['01', '02', '04', '05', '06', '08', '09', '10', '12', '13', # ... (나머지 주 FIPS 코드)
                   '48', '49', '50', '51', '53', '54', '55', '56'],
    'Unemployment_Rate': [5.7, 6.3, 6.0, 5.1, 7.5, 5.0, 6.3, 6.5, 6.3, 6.1, # ... (각 주에 대한 값)
                          5.2, 3.5, 3.7, 4.9, 5.8, 6.5, 4.6, 4.0]
}
# 모든 FIPS 코드에 대한 데이터를 준비해야 제대로 표시됨 (예시에서는 일부만 포함)
# 여기서는 간단히 몇 개만 표시되도록 랜덤 데이터 생성
all_states_fips = ["01","02","04","05","06","08","09","10","11","12","13","15","16","17","18","19","20","21","22","23","24","25","26","27","28","29","30","31","32","33","34","35","36","37","38","39","40","41","42","44","45","46","47","48","49","50","51","53","54","55","56"]
unemployment_data = pd.DataFrame({
    'State_FIPS': all_states_fips,
    'Unemployment_Rate': np.random.uniform(2.0, 10.0, size=len(all_states_fips))
})


# 기본 지도 생성 (미국 중심)
map_choropleth = folium.Map(location=[39.8283, -98.5795], zoom_start=4, tiles="CartoDB positron")

# 단계 구분도 추가
folium.Choropleth(
    geo_data=geojson_url,       # GeoJSON 데이터 (URL 또는 파일 경로, 또는 객체)
    name='choropleth_us_states', # 레이어 이름
    data=unemployment_data,     # Pandas DataFrame 형태의 통계 데이터
    columns=['State_FIPS', 'Unemployment_Rate'], # DataFrame에서 사용할 [키_컬럼, 값_컬럼]
    key_on='feature.id',        # GeoJSON 데이터에서 data의 키_컬럼과 매칭될 경로 (여기서는 FIPS 코드)
    fill_color='YlGnBu',        # 채우기 색상 팔레트 (YlGn, YlGnBu, PuRd, RdYlGn 등)
    fill_opacity=0.7,           # 채우기 투명도
    line_opacity=0.2,           # 경계선 투명도
    legend_name='Unemployment Rate (%)', # 범례 제목
    highlight=True              # 마우스 오버 시 해당 지역 하이라이트
).add_to(map_choropleth)

# 레이어 컨트롤 추가 (여러 레이어가 있을 경우 유용)
folium.LayerControl().add_to(map_choropleth)

# Streamlit에 표시
# st_folium(map_choropleth, width=700, height=500)
"""
st.code(code_choropleth_folium, language='python')

if st.checkbox("Folium 단계 구분도 예시 보기 (미국 주별 실업률)", key="folium_choropleth_page"):
    st.markdown("##### 미국 주별 임의 실업률 단계 구분도:")
    st.caption("데이터 로딩 및 지도 생성에 약간의 시간이 소요될 수 있습니다.")
    try:
        # 샘플 GeoJSON 데이터 URL (미국 주 경계)
        geojson_url = "https://raw.githubusercontent.com/python-visualization/folium/main/examples/data/us-states.json"
        # 모든 FIPS 코드 리스트 (실제 GeoJSON 파일의 id와 일치해야 함)
        all_states_fips = ["01","02","04","05","06","08","09","10","11","12","13","15","16","17","18","19","20","21","22","23","24","25","26","27","28","29","30","31","32","33","34","35","36","37","38","39","40","41","42","44","45","46","47","48","49","50","51","53","54","55","56"]

        # 임의의 실업률 데이터 생성
        np.random.seed(42) # 재현성을 위한 시드
        unemployment_data = pd.DataFrame({
            'State_FIPS': all_states_fips, # GeoJSON의 feature.id와 매칭될 키
            'Unemployment_Rate': np.random.uniform(2.0, 10.0, size=len(all_states_fips)).round(1) # 값
        })

        # 기본 지도 생성 (미국 중심)
        map_choropleth = folium.Map(location=[39.8283, -98.5795], zoom_start=4, tiles="CartoDB positron")

        # 단계 구분도 추가
        choro = folium.Choropleth(
            geo_data=geojson_url,
            name='US Unemployment Rate',
            data=unemployment_data,
            columns=['State_FIPS', 'Unemployment_Rate'],
            key_on='feature.id', # GeoJSON 파일의 각 feature가 'id' 라는 속성으로 FIPS 코드를 가짐
            fill_color='YlOrRd', # 색상 팔레트
            fill_opacity=0.7,
            line_opacity=0.3,
            legend_name='Unemployment Rate (%)',
            highlight=True, # 마우스 올리면 강조 표시
            nan_fill_color='lightgray' # 데이터 없는 지역 색상
        ).add_to(map_choropleth)

        # 툴팁 추가 (선택 사항, 각 지역에 마우스 오버 시 정보 표시)
        # GeoJSON의 'name' 속성을 사용하려면 GeoJSON 구조를 확인해야 함.
        # 이 예시의 GeoJSON은 feature.properties.name 으로 주 이름을 가짐.
        # folium.features.GeoJsonTooltip(fields=['name', 'Unemployment_Rate'], aliases=['State:', 'Unemployment Rate (%):'], labels=True, sticky=False).add_to(choro.geojson)
        # 위 방식은 복잡할 수 있으므로, 간단히 범례만 사용.

        # 레이어 컨트롤 추가
        folium.LayerControl().add_to(map_choropleth)

        # Streamlit에 표시
        st_folium(map_choropleth, width=700, height=500, returned_objects=[]) # returned_objects=[] 로 지도 상호작용 데이터 반환 안함

    except Exception as e:
        st.error(f"Folium 단계 구분도 생성 중 오류 발생: {e}")
        st.info("인터넷 연결을 확인하거나 GeoJSON URL이 유효한지 확인해주세요.")


st.markdown("---")
st.markdown("`folium`은 이 외에도 히트맵, 클러스터링, 다양한 플러그인 연동 등 풍부한 기능을 제공합니다. 공식 문서나 예제를 참고하여 더 많은 활용법을 익혀보세요.")