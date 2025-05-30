# utils.py
# 하위 페이지에서 공통으로 사용할 샘플 데이터 생성 함수
import pandas as pd
import numpy as np
import streamlit as st # @st.cache_data를 사용하기 위해 필요

@st.cache_data # 데이터 로딩/생성 함수는 캐싱하면 성능에 도움
def get_sample_data(type='numerical'):
    """
    시각화 예시에 사용할 샘플 데이터를 생성합니다.
    Args:
        type (str): 생성할 데이터의 종류 ('numerical', 'categorical', 'timeseries', 'mixed')
    Returns:
        pandas.DataFrame: 생성된 샘플 데이터프레임
    """
    np.random.seed(0) # 재현성을 위해 시드 고정
    if type == 'numerical':
        return pd.DataFrame({
            'A': np.random.randn(100), # 평균 0, 표준편차 1의 정규분포 난수 100개
            'B': np.random.randn(100) + 1, # 평균 1, 표준편차 1의 정규분포 난수 100개
            'C': np.random.rand(100) * 10  # 0과 1 사이의 균일분포 난수 100개에 10을 곱함
        })
    elif type == 'categorical':
        return pd.DataFrame({
            'Category': np.random.choice(['X', 'Y', 'Z', 'W'], 100, p=[0.4, 0.3, 0.2, 0.1]), # X,Y,Z,W 중 하나를 100번 선택 (확률 지정)
            'Value': np.random.randint(1, 100, 100) # 1과 99 사이의 정수 난수 100개
        })
    elif type == 'timeseries':
        dates = pd.date_range('20230101', periods=100, freq='D') # 2023년 1월 1일부터 100일간의 날짜 생성
        return pd.DataFrame(
            np.random.randn(100, 2).cumsum(axis=0) + 50, # 누적합 랜덤 워크 데이터 (2개 시리즈)
            index=dates,
            columns=['StockA', 'StockB']
        )
    elif type == 'mixed':
        return pd.DataFrame({
            'Group': np.random.choice(['Alpha', 'Beta', 'Gamma'], 100), # Alpha, Beta, Gamma 중 하나를 100번 선택
            'Metric1': np.random.rand(100) * 100, # 0과 100 사이의 실수 난수
            'Metric2': np.random.randn(100) * 50 + 20 # 평균 20, 표준편차 50 정도의 정규분포 난수
        })
    return pd.DataFrame()